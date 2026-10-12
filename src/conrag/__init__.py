from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conrag.config import Config
    from conrag.pipeline import ConRAG

__all__ = ("ConRAG", "Config")


def __getattr__(name: str) -> object:
    if name == "Config":
        from conrag.config import Config

        return Config
    if name == "ConRAG":
        from conrag.pipeline import ConRAG

        return ConRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
