import logging

from app.bootstrap.admin import bootstrap_admin
from app.core.config import get_settings
from app.db.session import get_session_factory

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    with get_session_factory()() as db:
        result = bootstrap_admin(db, settings)
    if result.created:
        logger.info("Created initial ADMIN user: %s", result.username)
    else:
        logger.info("ADMIN bootstrap skipped; user already exists: %s", result.username)


if __name__ == "__main__":
    main()
