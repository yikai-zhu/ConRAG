import json
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import ReadOnly, TypedDict, cast

from dotenv import load_dotenv

load_dotenv()

class Schema(TypedDict):
    nodes: ReadOnly[list[str]]
    relations: ReadOnly[list[str]]
    attributes: ReadOnly[list[str]]


@dataclass(kw_only=True, slots=True)
class Config:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    dataset: str = "hotpotqa"
    rebuild_knowledge_base: bool = False

    log_level: str = "INFO"
    log_file_name: str = "app.log"

    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = "API_KEY"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_seconds: int = 60
    llm_retry_count: int = 3
    max_output_tokens: int = 8192
    temperature: float = 0.0

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 64

    sequential_questions: bool = False
    max_workers: int = 64
    candidate_top_k: int = 16
    final_top_k: int = 3

    dataset_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    run_dir: Path = field(init=False)
    log_path: Path = field(init=False)
    schema: Schema = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = self.base_dir.expanduser().resolve()
        self.dataset_dir = self.base_dir / "datasets" / self.dataset
        self.output_dir = self.base_dir / "outputs" / self.dataset
        self.run_dir = (
            self.base_dir
            / "results"
            / self.dataset
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        )
        self.log_path = self.run_dir / "logs" / self.log_file_name

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        schema_resource = resources.files("conrag.prompts").joinpath("schema.json")
        self.schema = parse_schema(json.loads(schema_resource.read_text(encoding="utf-8")))


def parse_schema(raw: object) -> Schema:
    if not isinstance(raw, dict):
        raise TypeError("schema must be a JSON object")
    schema: dict[str, list[str]] = {}
    for key in ("nodes", "relations", "attributes"):
        values = raw.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
            raise TypeError(f"schema.{key} must be a non-empty list of strings")
        schema[key] = cast(list[str], values)
    return cast(Schema, schema)
