import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ReadOnly, TypedDict

from conrag.common import (
    TRACE_CONTEXT,
    map_bounded,
    progress_bar,
    read_json,
    write_json,
)
from conrag.prompts import ANSWER_EVALUATION_PROMPT

if TYPE_CHECKING:
    from conrag.clients import InferenceClient
    from conrag.config import Config

logger = logging.getLogger(__name__)

type JsonRecord = dict[str, object]


@dataclass(slots=True, frozen=True)
class EvaluationScore:
    string_accuracy: float
    answer_accuracy: float


class EvaluationReport(TypedDict):
    string_accuracy: ReadOnly[float]
    answer_accuracy: ReadOnly[float]


class Evaluator:
    def __init__(self, config: Config, llm: InferenceClient, results_path: Path) -> None:
        self.config = config
        self.llm = llm
        self.results_path = results_path
        self.records = [record for record in read_json(results_path, list) if isinstance(record, dict)]

    async def run(self) -> EvaluationReport:
        with progress_bar(len(self.records), "Evaluation", "sample") as bar:

            async def process(index: int, record: JsonRecord) -> EvaluationScore:
                with TRACE_CONTEXT.set(f"evaluation:{index}"):
                    try:
                        return await self.evaluate_record(record)
                    finally:
                        bar.update(1)

            scores = await map_bounded(self.records, self.config.max_workers, process)

        report = self.average(scores)

        await asyncio.to_thread(write_json, self.results_path, self.records)
        await asyncio.to_thread(write_json, self.results_path.parent / "evaluation_report.json", report)
        logger.info("Evaluation complete: %s", report)

        return report

    async def evaluate_record(self, record: JsonRecord) -> EvaluationScore:
        pred_answer = str(record.get("pred_answer", ""))
        gold_answer = str(record.get("gold_answer", ""))

        str_acc = 1.0 if self.normalize_answer(gold_answer) in self.normalize_answer(pred_answer) else 0.0
        ans_acc = await self.judge(pred_answer, gold_answer)

        score = EvaluationScore(str_acc, ans_acc)

        for field in ("em", "f1", "string_precision"):
            record.pop(field, None)
        record.update(
            {
                "string_accuracy": str_acc,
                "answer_accuracy": ans_acc,
            }
        )

        return score

    async def judge(self, pred_answer: str, gold_answer: str) -> float:
        if not pred_answer.strip():
            return 0.0
        try:
            response = await self.llm.infer(
                instructions=ANSWER_EVALUATION_PROMPT["instructions"],
                input_text=ANSWER_EVALUATION_PROMPT["input"].format(
                    pred_answer=pred_answer,
                    gold_answer=gold_answer,
                ),
            )
            return 1.0 if response.strip().lower() == "correct" else 0.0
        except Exception:
            logger.exception("LLM judge failed")
            return 0.0

    @staticmethod
    def normalize_answer(text: object) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = text.replace("-", " ")
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(text.split())
        return text

    @staticmethod
    def average(scores: list[EvaluationScore]) -> EvaluationReport:
        if not scores:
            return {
                "string_accuracy": 0.0,
                "answer_accuracy": 0.0,
            }

        avg_str_acc = sum(score.string_accuracy for score in scores) / len(scores)
        avg_ans_acc = sum(score.answer_accuracy for score in scores) / len(scores)

        return {
            "string_accuracy": avg_str_acc,
            "answer_accuracy": avg_ans_acc,
        }
