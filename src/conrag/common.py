import asyncio
import json
import logging
import logging.handlers
import pickle
import re
import sys
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from tqdm import tqdm

if TYPE_CHECKING:
    from conrag.config import Config

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(trace_context)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
PROGRESS_FORMAT = "{desc:<20} {percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
TRACE_CONTEXT = ContextVar("conrag_trace_context", default="-")

_DASH_TRANSLATION = str.maketrans({
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
})
_SPACE_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\s'\"“”‘’`.,;:!?()\[\]{}]+|[\s'\"“”‘’`.,;:!?()\[\]{}]+$")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(_DASH_TRANSLATION)
    text = _SPACE_RE.sub(" ", text).strip()
    return _EDGE_PUNCT_RE.sub("", text)


def read_json[T](path: Path, expected_type: type[T]) -> T:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, expected_type):
        raise TypeError(f"{path} must contain {expected_type.__name__}, got {type(payload).__name__}")
    return cast(T, payload)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def configure_logging(config: Config) -> None:
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)

    root.setLevel(level)
    root.addHandler(_console_handler(level))
    root.addHandler(_file_handler(config.log_path, level))
    for name in ("httpx","httpx2", "httpcore", "networkx", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


async def map_bounded[T, R](
    items: Sequence[T],
    max_workers: int,
    handler: Callable[[int, T], Awaitable[R]],
) -> list[R]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    queue: asyncio.Queue[tuple[int, T]] = asyncio.Queue(maxsize=max_workers)
    results: dict[int, R] = {}

    async def producer() -> None:
        for index, item in enumerate(items):
            await queue.put((index, item))
        queue.shutdown()

    async def worker() -> None:
        while True:
            try:
                index, payload = await queue.get()
            except asyncio.QueueShutDown:
                return
            results[index] = await handler(index, payload)

    async with asyncio.TaskGroup() as group:
        group.create_task(producer(), name="bounded-producer")
        for index in range(max_workers):
            group.create_task(worker(), name=f"bounded-worker-{index}")

    return [results[index] for index in range(len(items))]


class _TraceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_context = TRACE_CONTEXT.get()
        return True


def progress_bar(total: int, desc: str, unit: str) -> tqdm[Any]:
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=False,
        dynamic_ncols=True,
        bar_format=PROGRESS_FORMAT,
    )


def _console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.addFilter(_TraceContextFilter())
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    return handler


def _file_handler(path: Path, level: int) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename=path,
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.addFilter(_TraceContextFilter())
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    return handler
