import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NotRequired, ReadOnly, TypedDict, cast

import json_repair
import networkx as nx

from conrag.common import (
    TRACE_CONTEXT,
    clean_text,
    map_bounded,
    progress_bar,
    read_pickle,
    write_pickle,
)
from conrag.prompts import EXTRACTION_PROMPT

if TYPE_CHECKING:
    from conrag.clients import InferenceClient
    from conrag.config import Config

logger = logging.getLogger(__name__)

type AttrMap = dict[str, list[str]]
type EdgeKey = tuple[str, str, str]


class NodeData(TypedDict):
    name: ReadOnly[str]
    type: ReadOnly[str]
    level: NotRequired[ReadOnly[int]]
    attributes: NotRequired[AttrMap]
    source_chunks: NotRequired[list[str]]


class EdgeData(TypedDict):
    relation: ReadOnly[str]
    weight: NotRequired[float]
    source_chunks: NotRequired[list[str]]


@dataclass(slots=True)
class Entity:
    id: str
    name: str
    type: str
    attributes: AttrMap = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Relation:
    source_id: str
    target_id: str
    key: EdgeKey
    text: str


@dataclass(slots=True)
class GraphPatch:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


class GraphBuilder:
    def __init__(self, config: Config, llm: InferenceClient) -> None:
        self.config = config
        self.llm = llm
        self.graph = nx.MultiDiGraph()
        self._schema_text = json.dumps(config.schema, ensure_ascii=False, indent=2)

    async def build(self, chunks: dict[str, str]) -> nx.MultiDiGraph:
        items = list(chunks.items())
        patches: asyncio.Queue[tuple[str, GraphPatch]] = asyncio.Queue(maxsize=self.config.max_workers)
        self.graph = nx.MultiDiGraph()
        logger.info("Building graph from %d chunks", len(items))

        with progress_bar(len(items), "Graph Construction", "chunk") as bar:

            async def extract(_index: int, item: tuple[str, str]) -> None:
                chunk_id, text = item
                with TRACE_CONTEXT.set(f"chunk:{chunk_id}"):
                    try:
                        await patches.put((chunk_id, await self._extract(text, chunk_id)))
                    finally:
                        bar.update(1)

            async def merge() -> None:
                for _ in items:
                    chunk_id, patch = await patches.get()
                    try:
                        self._apply_patch(patch, chunk_id)
                    finally:
                        patches.task_done()

            async with asyncio.TaskGroup() as group:
                group.create_task(merge(), name="graph-merge")
                group.create_task(
                    map_bounded(items, self.config.max_workers, extract),
                    name="graph-extract",
                )

        logger.info("Graph built (nodes=%d, edges=%d)", self.graph.number_of_nodes(), self.graph.number_of_edges())
        return self.graph

    def load(self) -> nx.MultiDiGraph:
        path = self.config.output_dir / "knowledge_graph.pkl"
        graph = read_pickle(path)
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError(f"Invalid graph type: {type(graph).__name__}")
        self.graph = graph
        logger.info("Loaded graph from %s (nodes=%d, edges=%d)", path, graph.number_of_nodes(), graph.number_of_edges())
        return graph

    def save(self) -> None:
        path = self.config.output_dir / "knowledge_graph.pkl"
        write_pickle(path, self.graph)
        logger.info("Saved graph to %s", path)

    async def _extract(self, text: str, chunk_id: str) -> GraphPatch:
        try:
            raw = await self.llm.infer(
                instructions=EXTRACTION_PROMPT["instructions"],
                input_text=EXTRACTION_PROMPT["input"].format(schema=self._schema_text, passage=text),
            )
            return parse_graph_payload(json_repair.loads(raw))
        except Exception:
            logger.exception("Graph extraction failed for chunk %s", chunk_id)
            return GraphPatch()

    def _apply_patch(self, patch: GraphPatch, chunk_id: str) -> None:
        for entity in patch.entities:
            current = self.graph.nodes.get(entity.id)
            if current is None:
                self.graph.add_node(
                    entity.id,
                    name=entity.name,
                    type=entity.type,
                    level=0,
                    attributes={key: list(values) for key, values in entity.attributes.items()},
                    source_chunks=[chunk_id],
                )
                continue
            merge_attributes(cast(AttrMap, current.setdefault("attributes", {})), entity.attributes)
            add_unique(cast(list[str], current.setdefault("source_chunks", [])), chunk_id)

        for relation in patch.relations:
            current = self.graph.get_edge_data(relation.source_id, relation.target_id, key=relation.key)
            if current is None:
                self.graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    key=relation.key,
                    relation=relation.text,
                    weight=1.0,
                    source_chunks=[chunk_id],
                )
                continue
            current["weight"] = float(current.get("weight", 1.0)) + 1.0
            add_unique(cast(list[str], current.setdefault("source_chunks", [])), chunk_id)


def parse_graph_payload(payload: object) -> GraphPatch:
    if not isinstance(payload, Mapping):
        return GraphPatch()

    entity_types = parse_entities(payload.get("entities", {}))
    attributes = parse_attributes(payload.get("attributes", {}), entity_types)
    triples = parse_triples(payload.get("triples", []))

    entities: dict[str, Entity] = {}
    relations: dict[EdgeKey, Relation] = {}

    for name, entity_type in entity_types.items():
        entity_id = node_id(name, entity_type)
        entities[entity_id] = Entity(id=entity_id, name=name, type=entity_type)

    for name, attrs in attributes.items():
        entity_type = entity_types.get(name, "unknown")
        entity_id = node_id(name, entity_type)
        entity = entities.setdefault(entity_id, Entity(id=entity_id, name=name, type=entity_type))
        merge_attributes(entity.attributes, attrs)

    for source, relation, target in triples:
        source_type = entity_types.get(source, "unknown")
        target_type = entity_types.get(target, "unknown")
        source_id = node_id(source, source_type)
        target_id = node_id(target, target_type)
        edge_key = (source_id, target_id, relation)
        entities.setdefault(source_id, Entity(id=source_id, name=source, type=source_type))
        entities.setdefault(target_id, Entity(id=target_id, name=target, type=target_type))
        relations.setdefault(edge_key, Relation(source_id=source_id, target_id=target_id, key=edge_key, text=relation))

    return GraphPatch(entities=list(entities.values()), relations=list(relations.values()))


def parse_entities(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    entities: dict[str, str] = {}
    for key, value in raw.items():
        name = clean_text(key)
        entity_type = clean_text(value)
        if name and entity_type:
            entities[name] = entity_type
    return entities


def parse_attributes(raw: object, entity_types: dict[str, str]) -> dict[str, AttrMap]:
    if not isinstance(raw, Mapping):
        return {}
    parsed: dict[str, AttrMap] = {}
    for raw_name, raw_values in raw.items():
        name = clean_text(raw_name)
        if not name:
            continue
        entity_types.setdefault(name, "unknown")
        values = raw_values if isinstance(raw_values, list | tuple | set) else [raw_values]
        attrs: AttrMap = {}
        for item in values:
            key, separator, value = str(item).partition(":")
            if not separator:
                continue
            clean_key = clean_text(key)
            clean_value = clean_text(value)
            if clean_key and clean_value:
                add_unique(attrs.setdefault(clean_key, []), clean_value)
        if attrs:
            parsed[name] = attrs
    return parsed


def parse_triples(raw: object) -> list[tuple[str, str, str]]:
    if not isinstance(raw, list | tuple):
        return []
    triples: list[tuple[str, str, str]] = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) < 3:
            continue
        source, relation, target = (clean_text(value) for value in item[:3])
        if source and relation and target:
            triples.append((source, relation, target))
    return triples


def node_text(data: NodeData) -> str:
    parts = [f"{data['name']}::{data['type']}"]
    if attrs := data.get("attributes"):
        attr_map = cast(AttrMap, attrs)
        parts.append("Attributes:\n" + "\n".join(f" - {key}::{'; '.join(values)}" for key, values in sorted(attr_map.items())))
    return "\n".join(parts)


def merge_attributes(target: AttrMap, source: AttrMap) -> None:
    for key, values in source.items():
        for value in values:
            add_unique(target.setdefault(key, []), value)


def add_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def node_id(name: str, entity_type: str) -> str:
    return f"{name}::{entity_type}"


def require_graph(graph: nx.MultiDiGraph | None) -> nx.MultiDiGraph:
    if graph is None:
        raise RuntimeError("Graph is not initialized")
    return graph
