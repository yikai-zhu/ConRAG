import asyncio
import logging
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import json_repair
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from conrag.common import TRACE_CONTEXT, clean_text
from conrag.planning import NOT_FOUND, PlanStep, normalize_plan, render_plan
from conrag.prompts import (
    FINAL_ANSWER_PROMPT,
    QUESTION_DECOMPOSITION_PROMPT,
    SINGLE_QUESTION_ANSWER_PROMPT,
    Prompt,
)

if TYPE_CHECKING:
    from conrag.clients import EmbeddingProvider, InferenceClient
    from conrag.config import Config
    from conrag.graph import Entity, Relation

logger = logging.getLogger(__name__)


type ScoreMap = dict[str, float]
type JsonObject = dict[str, object]


class RetrievalStore(Protocol):
    def search_nodes(
        self,
        query: NDArray[np.float32],
        top_k: int,
    ) -> list[tuple[Entity, float]]: ...

    def search_relations(
        self,
        query: NDArray[np.float32],
        top_k: int,
    ) -> list[tuple[Relation, float]]: ...

    def search_chunks(
        self,
        query: NDArray[np.float32],
        top_k: int,
    ) -> list[tuple[str, float]]: ...


@dataclass(slots=True, frozen=True)
class Hit:
    chunk_id: str
    score: float


@dataclass(slots=True)
class EntityCandidateEvidence:
    similarity_sum: float = 0.0
    specificity_mass: float = 0.0

    def add(self, similarity: float, specificity: float) -> None:
        self.similarity_sum += similarity
        self.specificity_mass += similarity * specificity

    @property
    def weighted_specificity(self) -> float:
        return self.specificity_mass / self.similarity_sum


@dataclass(slots=True, frozen=True)
class Decomposition:
    acquired_information: str
    plan: list[PlanStep]


@dataclass(slots=True, frozen=True)
class StepResult:
    answer: str
    acquired_information: str


class RetrievalEngine:
    def __init__(
        self,
        config: Config,
        llm: InferenceClient,
        embeddings: EmbeddingProvider,
        store: RetrievalStore,
        graph: nx.MultiDiGraph,
        chunks: dict[str, str],
    ) -> None:
        self.config = config
        self.llm = llm
        self.embeddings = embeddings
        self.store = store
        self.graph = graph
        self.chunks = chunks

    async def answer(self, question: str) -> str:
        decomposition = await self._decompose_question(question)
        memory = [decomposition.acquired_information] if decomposition.acquired_information else []

        step_answers = await self._execute_plan(question, decomposition.plan, memory)
        return await self._answer_from_plan(question, decomposition.plan, step_answers, memory)

    async def _decompose_question(self, question: str) -> Decomposition:
        context = await self._retrieve_context(question)
        fallback_plan = single_step_plan(question)
        payload = await self._json_call(
            QUESTION_DECOMPOSITION_PROMPT,
            {"question": question, "evidence": context},
            {"acquired_information": "", "plan": fallback_plan},
        )
        return Decomposition(
            acquired_information=clean_optional(payload.get("acquired_information")),
            plan=normalize_plan(payload.get("plan", [])) or fallback_plan,
        )

    async def _execute_plan(self, question: str, plan: list[PlanStep], memory: list[str]) -> dict[int, str]:
        plan = normalize_plan(plan)
        if not plan:
            return {}

        base_memory = tuple(memory)
        events = {step["id"]: asyncio.Event() for step in plan}
        answers: dict[int, str] = {}
        new_memory: dict[int, str] = {}

        async with asyncio.TaskGroup() as group:
            for step in plan:
                group.create_task(
                    self._execute_step(question, step, base_memory, events, answers, new_memory),
                    name=f"plan-step-{step['id']}",
                )

        for step in plan:
            if text := new_memory.get(step["id"]):
                memory.append(text)
        return answers

    async def _execute_step(
        self,
        question: str,
        step: PlanStep,
        base_memory: tuple[str, ...],
        events: dict[int, asyncio.Event],
        answers: dict[int, str],
        new_memory: dict[int, str],
    ) -> None:
        step_id = step["id"]
        sub_question = step["sub_question"]
        with TRACE_CONTEXT.set(f"{TRACE_CONTEXT.get()}/step:{step_id}"):
            try:
                await self._wait_for_dependencies(step, events)
                sub_question = render_sub_question(step, answers)
                step_memory = collect_step_memory(step, base_memory, new_memory)

                result = await self._answer_step(question, sub_question, step_memory)
                answers[step_id] = result.answer
                if result.acquired_information:
                    new_memory[step_id] = result.acquired_information
            except Exception:
                logger.exception("Plan step failed: step_id=%s question=%s", step_id, sub_question)
                answers[step_id] = NOT_FOUND
            finally:
                events[step_id].set()

    async def _wait_for_dependencies(self, step: PlanStep, events: dict[int, asyncio.Event]) -> None:
        for dep_id in step["dependencies"]:
            await events[dep_id].wait()

    async def _answer_step(self, question: str, sub_question: str, memory: tuple[str, ...]) -> StepResult:
        context = await self._retrieve_context(sub_question)
        payload = await self._json_call(
            SINGLE_QUESTION_ANSWER_PROMPT,
            {
                "original_question": question,
                "acquired_information": format_memory(memory),
                "sub_question": sub_question,
                "evidence": context,
            },
            {"answer": NOT_FOUND, "acquired_information": ""},
        )
        return StepResult(
            answer=clean_answer(payload.get("answer")),
            acquired_information=clean_optional(payload.get("acquired_information")),
        )

    async def _answer_from_plan(
        self,
        question: str,
        plan: list[PlanStep],
        answers: dict[int, str],
        memory: list[str],
    ) -> str:
        evidence = "\n\n".join(
            item
            for item in (
                render_plan(plan, answers),
                "\n".join(f"- {item}" for item in memory),
            )
            if item
        )
        try:
            raw = await self.llm.infer(
                instructions=FINAL_ANSWER_PROMPT["instructions"],
                input_text=FINAL_ANSWER_PROMPT["input"].format(
                    question=question,
                    evidence=evidence,
                ),
            )
            return clean_answer(raw)
        except Exception:
            logger.exception("Final answer generation failed: question=%s", question[:80])
            return NOT_FOUND

    async def _retrieve_context(self, text: str) -> str:
        query = await self.embeddings.encode_async(text, "query")
        return self._render_context(await self._retrieve(query, self.config.final_top_k))

    async def _retrieve(self, query: NDArray[np.float32], top_k: int) -> list[Hit]:
        async with asyncio.TaskGroup() as group:
            relation_task = group.create_task(
                asyncio.to_thread(self._relation_scores, query),
                name="retrieve-relation",
            )
            entity_task = group.create_task(
                asyncio.to_thread(self.store.search_nodes, query, self.config.candidate_top_k),
                name="retrieve-entity",
            )
            text_task = group.create_task(
                asyncio.to_thread(self.store.search_chunks, query, self.config.candidate_top_k),
                name="retrieve-text",
            )

        relation_scores = normalize_scores(relation_task.result())
        entity_anchors = entity_task.result()
        entity_evidence = await asyncio.to_thread(self._entity_evidence, entity_anchors)
        entity_scores = specificity_weighted_entity_scores(entity_evidence)
        text_scores = normalize_scores(dict(text_task.result()))

        hits = [
            Hit(
                chunk_id=chunk_id,
                score=ratsat_fusion_score(
                    relation_scores.get(chunk_id, 0.0),
                    entity_scores.get(chunk_id, 0.0),
                    text_scores.get(chunk_id, 0.0),
                ),
            )
            for chunk_id in set(relation_scores) | set(entity_scores) | set(text_scores)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _relation_scores(self, query: NDArray[np.float32]) -> ScoreMap:
        scores: dict[str, list[float]] = defaultdict(list)
        for relation, score in self.store.search_relations(query, self.config.candidate_top_k):
            edge_data = self.graph.get_edge_data(relation.source_id, relation.target_id, key=relation.key)
            if not edge_data:
                continue
            for chunk_id in source_chunks(edge_data):
                scores[chunk_id].append(score)

        relation_scores: ScoreMap = {}
        for chunk_id, values in scores.items():
            pooled_score = bounded_residual_relation_score(values)
            if pooled_score > 0.0:
                relation_scores[chunk_id] = pooled_score
        return relation_scores

    def _entity_evidence(
        self,
        anchors: list[tuple[Entity, float]],
    ) -> dict[str, EntityCandidateEvidence]:
        evidence_by_chunk: dict[str, EntityCandidateEvidence] = {}

        for anchor, anchor_score in anchors:
            if anchor.id not in self.graph or not math.isfinite(anchor_score) or anchor_score <= 0.0:
                continue

            provenance = source_chunks(self.graph.nodes[anchor.id])
            if not provenance:
                continue
            specificity = normalized_idf_specificity(
                fanout=len(provenance),
                corpus_size=len(self.chunks),
            )
            for chunk_id in provenance:
                evidence_by_chunk.setdefault(chunk_id, EntityCandidateEvidence()).add(
                    anchor_score,
                    specificity,
                )

        return evidence_by_chunk

    def _render_context(self, hits: list[Hit]) -> str:
        return "\n\n".join(self.chunks[hit.chunk_id] for hit in hits if hit.chunk_id in self.chunks)

    async def _json_call(self, prompt: Prompt, values: JsonObject, fallback: JsonObject) -> JsonObject:
        try:
            raw = await self.llm.infer(
                instructions=prompt["instructions"],
                input_text=prompt["input"].format(**values),
            )
            payload = json_repair.loads(raw)
            if isinstance(payload, dict):
                return cast(JsonObject, payload)
            logger.warning("LLM payload was not a JSON object")
        except Exception:
            logger.exception("LLM JSON call failed")
        return fallback


def render_sub_question(step: PlanStep, answers: dict[int, str]) -> str:
    sub_question = step["sub_question"]
    for dep_id in step["dependencies"]:
        sub_question = sub_question.replace(f"<dep:{dep_id}>", answers.get(dep_id, NOT_FOUND))
    return sub_question


def collect_step_memory(step: PlanStep, base_memory: tuple[str, ...], new_memory: dict[int, str]) -> tuple[str, ...]:
    return base_memory + tuple(new_memory[dep_id] for dep_id in step["dependencies"] if dep_id in new_memory)


def format_memory(memory: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in memory if item)


def source_chunks(data: Mapping[str, object]) -> list[str]:
    chunks = data.get("source_chunks", [])
    if isinstance(chunks, str):
        return [chunks]
    if isinstance(chunks, list | tuple | set):
        return [str(chunk_id) for chunk_id in chunks if chunk_id]
    return []


def normalize_scores(scores: ScoreMap) -> ScoreMap:
    finite = {key: value for key, value in scores.items() if math.isfinite(value)}
    if not finite:
        return {}
    low = min(finite.values())
    high = max(finite.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in finite}
    return {key: (value - low) / (high - low) for key, value in finite.items()}


def normalized_idf_specificity(fanout: int, corpus_size: int) -> float:
    idf = math.log1p((corpus_size - fanout + 0.5) / (fanout + 0.5))
    maximum_idf = math.log1p((corpus_size - 0.5) / 1.5)
    return idf / maximum_idf


def specificity_weighted_entity_scores(
    evidence_by_chunk: Mapping[str, EntityCandidateEvidence],
) -> ScoreMap:
    normalized = normalize_scores(
        {
            chunk_id: evidence.similarity_sum
            for chunk_id, evidence in evidence_by_chunk.items()
        }
    )
    return {
        chunk_id: score * evidence_by_chunk[chunk_id].weighted_specificity
        for chunk_id, score in normalized.items()
    }


def ratsat_fusion_score(
    relation_score: float,
    entity_score: float,
    text_score: float,
) -> float:
    structural_mean = (relation_score + entity_score) / 2.0
    return text_score + structural_mean / (1.0 + structural_mean)


def bounded_residual_relation_score(scores: list[float]) -> float:
    finite = [score for score in scores if math.isfinite(score)]
    positive = [score for score in finite if score > 0.0]
    if not positive:
        return 0.0
    best = max(positive)
    normalized_residual_support = (math.fsum(positive) - best) / len(finite)
    return best + (1.0 - best) * normalized_residual_support


def clean_answer(value: object) -> str:
    text = clean_text(NOT_FOUND if value is None else value)
    return text or NOT_FOUND


def clean_optional(value: object) -> str:
    return "" if value is None else clean_text(value)


def single_step_plan(question: str) -> list[PlanStep]:
    sub_question = clean_text(question) or "Answer the original question"
    return [{"id": 0, "sub_question": sub_question, "dependencies": []}]
