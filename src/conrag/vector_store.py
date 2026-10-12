import logging
from typing import TYPE_CHECKING, Any, cast

import faiss
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from conrag.common import read_pickle, write_pickle
from conrag.graph import (
    AttrMap,
    EdgeData,
    Entity,
    NodeData,
    Relation,
    node_text,
    require_graph,
)

if TYPE_CHECKING:
    from conrag.clients import EmbeddingClient
    from conrag.config import Config

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, config: Config, embeddings: EmbeddingClient) -> None:
        self.config = config
        self.embeddings = embeddings
        self.graph: nx.MultiDiGraph | None = None
        self.chunks: dict[str, str] = {}
        self.node_index: faiss.Index | None = None
        self.relation_index: faiss.Index | None = None
        self.chunk_index: faiss.Index | None = None
        self.node_map: dict[int, Entity] = {}
        self.relation_map: dict[int, Relation] = {}
        self.chunk_map: dict[int, str] = {}

    def build(self, graph: nx.MultiDiGraph, chunks: dict[str, str]) -> None:
        if not chunks:
            raise ValueError("Cannot build vector store without chunks")
        self.graph = graph
        self.chunks = dict(chunks)

        chunk_vectors = self.embeddings.encode(list(self.chunks.values()))
        if chunk_vectors.ndim != 2 or chunk_vectors.shape[1] <= 0:
            raise ValueError(f"Invalid embedding shape: {chunk_vectors.shape}")
        dim = int(chunk_vectors.shape[1])

        self.chunk_index = make_index(chunk_vectors, dim)
        self.chunk_map = dict(enumerate(self.chunks))

        node_texts, nodes = self._node_documents()
        self.node_index = make_index(self._encode_or_empty(node_texts, dim), dim)
        self.node_map = dict(enumerate(nodes))

        relation_texts, relations = self._relation_documents()
        self.relation_index = make_index(self._encode_or_empty(relation_texts, dim), dim)
        self.relation_map = dict(enumerate(relations))

        logger.info("Vector store built (nodes=%d, relations=%d, chunks=%d)", len(nodes), len(relations), len(self.chunks))

    def load(self, graph: nx.MultiDiGraph) -> dict[str, str]:
        self.graph = graph
        self.node_index = faiss.read_index(str(self.config.output_dir / "nodes.faiss"))
        self.node_map = cast(dict[int, Entity], read_pickle(self.config.output_dir / "node_map.pkl"))
        self.relation_index = faiss.read_index(str(self.config.output_dir / "relations.faiss"))
        self.relation_map = cast(
            dict[int, Relation],
            read_pickle(self.config.output_dir / "relation_map.pkl"),
        )
        self.chunk_index = faiss.read_index(str(self.config.output_dir / "chunks.faiss"))
        self.chunk_map = cast(dict[int, str], read_pickle(self.config.output_dir / "chunk_map.pkl"))
        self.chunks = cast(dict[str, str], read_pickle(self.config.output_dir / "chunk_corpus.pkl"))
        logger.info("Loaded vector store from %s", self.config.output_dir)
        return self.chunks

    def save(self) -> None:
        if self.node_index is None or self.relation_index is None or self.chunk_index is None:
            raise RuntimeError("Vector store is not initialized")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.node_index, str(self.config.output_dir / "nodes.faiss"))
        faiss.write_index(self.relation_index, str(self.config.output_dir / "relations.faiss"))
        faiss.write_index(self.chunk_index, str(self.config.output_dir / "chunks.faiss"))
        write_pickle(self.config.output_dir / "node_map.pkl", self.node_map)
        write_pickle(self.config.output_dir / "relation_map.pkl", self.relation_map)
        write_pickle(self.config.output_dir / "chunk_map.pkl", self.chunk_map)
        write_pickle(self.config.output_dir / "chunk_corpus.pkl", self.chunks)

    def search_nodes(self, query: NDArray[np.float32], top_k: int) -> list[tuple[Entity, float]]:
        return [(self.node_map[idx], score) for idx, score in search(self.node_index, query, top_k) if idx in self.node_map]

    def search_relations(self, query: NDArray[np.float32], top_k: int) -> list[tuple[Relation, float]]:
        return [(self.relation_map[idx], score) for idx, score in search(self.relation_index, query, top_k) if idx in self.relation_map]

    def search_chunks(self, query: NDArray[np.float32], top_k: int) -> list[tuple[str, float]]:
        return [(self.chunk_map[idx], score) for idx, score in search(self.chunk_index, query, top_k) if idx in self.chunk_map]

    def _node_documents(self) -> tuple[list[str], list[Entity]]:
        graph = require_graph(self.graph)
        texts: list[str] = []
        nodes: list[Entity] = []
        for node_id, raw_data in graph.nodes(data=True):
            data = cast(NodeData, raw_data)
            entity = Entity(
                id=node_id,
                name=str(data["name"]),
                type=str(data["type"]),
                attributes=dict(cast(AttrMap, data.get("attributes", {}))),
            )
            texts.append(node_text(data))
            nodes.append(entity)
        return texts, nodes

    def _relation_documents(self) -> tuple[list[str], list[Relation]]:
        graph = require_graph(self.graph)
        texts: list[str] = []
        relations: list[Relation] = []
        for source_id, target_id, edge_key, raw_data in graph.edges(data=True, keys=True):
            data = cast(EdgeData, raw_data)
            source = graph.nodes[source_id]
            target = graph.nodes[target_id]
            relation = str(data["relation"])
            texts.append(f"{source['name']}::{source['type']} [{relation}] {target['name']}::{target['type']}")
            relations.append(Relation(source_id=source_id, target_id=target_id, key=edge_key, text=relation))
        return texts, relations

    def _encode_or_empty(self, texts: list[str], dim: int) -> NDArray[np.float32]:
        return self.embeddings.encode(texts) if texts else np.empty((0, dim), dtype=np.float32)


def make_index(vectors: NDArray[np.float32], dim: int) -> faiss.Index:
    index = faiss.IndexFlatIP(dim)
    if vectors.shape[0] > 0:
        cast(Any, index).add(vectors)
    return index


def search(index: faiss.Index | None, query: NDArray[np.float32], top_k: int) -> list[tuple[int, float]]:
    if index is None or index.ntotal == 0 or top_k <= 0:
        return []

    matrix = np.ascontiguousarray(query, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != index.d:
        raise ValueError(f"Query dim {matrix.shape[1]} does not match index dim {index.d}")

    scores, ids = cast(Any, index).search(matrix, top_k)
    return [
        (int(idx), float(score))
        for score, idx in zip(scores[0], ids[0], strict=True)
        if idx != -1
    ]
