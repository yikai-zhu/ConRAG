import logging

from simple_parsing import parse

from conrag import Config, ConRAG
from conrag.common import configure_logging

logger = logging.getLogger("conrag.main")


def main() -> int:
    config = parse(Config)
    configure_logging(config)
    logger.info("Starting ConRAG")
    ConRAG(config).run()
    logger.info("Finished ConRAG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
