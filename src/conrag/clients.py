import asyncio
import re
from collections import deque
from collections.abc import Sequence
from contextvars import Context
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from conrag.config import Config

_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_FENCE_RE = re.compile(r"```(?:\s*\w+)?\s*\n(?P<body>[\s\S]*?)\n\s*```", re.MULTILINE)


@dataclass(slots=True)
class _EmbeddingRequest:
    texts: tuple[str, ...]
    task: str
    future: asyncio.Future[NDArray[np.float32]]


class InferenceClient(Protocol):
    async def infer(self, *, instructions: str, input_text: str) -> str: ...


class EmbeddingProvider(Protocol):
    async def encode_async(
        self,
        texts: str,
        task: str = "document",
    ) -> NDArray[np.float32]: ...


class LLMClient:
    _client: AsyncOpenAI

    def __init__(self, config: Config) -> None:
        self._model = config.llm_model
        self._max_output_tokens = config.max_output_tokens
        self._temperature = config.temperature
        self._base_url = config.llm_base_url
        self._api_key = config.llm_api_key
        self._timeout_seconds = config.llm_timeout_seconds
        self._retry_count = config.llm_retry_count

    async def __aenter__(self) -> Self:
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            max_retries=self._retry_count,
            http_client=DefaultAsyncHttpxClient(
                timeout=self._timeout_seconds
            ),
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def infer(self, *, instructions: str, input_text: str) -> str:
        response = await self._client.responses.create(
            model=self._model,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            instructions=instructions.strip(),
            input=input_text.strip(),
        )
        return clean_llm_text(response.output_text)

    async def close(self) -> None:
        await self._client.close()


class EmbeddingClient:
    def __init__(self, config: Config) -> None:
        self._batch_size = config.embedding_batch_size
        self._model = SentenceTransformer(
            config.embedding_model,
            trust_remote_code=True,
            device=config.embedding_device,
        )
        self._requests: asyncio.Queue[_EmbeddingRequest] = asyncio.Queue()
        self._owner: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def encode_async(
        self, texts: str | Sequence[str], task: str = "document"
    ) -> NDArray[np.float32]:
        batch = (texts,) if isinstance(texts, str) else tuple(texts)
        if not batch:
            return np.empty((0, 0), dtype=np.float32)

        if self._owner is None:
            self._owner = asyncio.create_task(
                self._run_owner(),
                name="embedding-owner",
                context=Context(),
            )
        future = asyncio.get_running_loop().create_future()
        await self._requests.put(_EmbeddingRequest(batch, task, future))
        return await future

    async def close(self) -> None:
        self._requests.shutdown()
        if self._owner is not None:
            await self._owner

    async def _run_owner(self) -> None:
        pending: deque[_EmbeddingRequest] = deque()
        while True:
            if pending:
                first = pending.popleft()
            else:
                try:
                    first = await self._requests.get()
                except asyncio.QueueShutDown:
                    return

            requests = [first]
            text_count = len(first.texts)
            await asyncio.sleep(0)
            while text_count < self._batch_size:
                try:
                    request = self._requests.get_nowait()
                except (asyncio.QueueEmpty, asyncio.QueueShutDown):
                    break
                if request.task == first.task and text_count + len(request.texts) <= self._batch_size:
                    requests.append(request)
                    text_count += len(request.texts)
                else:
                    pending.append(request)

            texts = [text for request in requests for text in request.texts]
            try:
                vectors = await asyncio.to_thread(self.encode, texts, first.task)
            except Exception as exc:
                for request in requests:
                    if not request.future.done():
                        request.future.set_exception(exc)
            else:
                offset = 0
                for request in requests:
                    next_offset = offset + len(request.texts)
                    if not request.future.done():
                        request.future.set_result(vectors[offset:next_offset].copy())
                    offset = next_offset
            finally:
                for _request in requests:
                    self._requests.task_done()

    def encode(
        self, texts: str | Sequence[str], task: str = "document"
    ) -> NDArray[np.float32]:
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            return np.empty((0, 0), dtype=np.float32)

        encode = getattr(self._model, f"encode_{task}", self._model.encode)
        vectors = np.ascontiguousarray(
            encode(
                batch,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors


def clean_llm_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = _ZERO_WIDTH_RE.sub(
        "", value.replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    if match := _FENCE_RE.search(text):
        text = match.group("body").strip()
    elif text.startswith("```") and text.endswith("```") and len(text) >= 6:
        text = text[3:-3].strip()
    if text.lower().startswith("json\n"):
        text = text.split("\n", 1)[1].strip()
    return text
