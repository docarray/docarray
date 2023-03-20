import pytest
import logging


@pytest.fixture(autouse=True)
def set_logger_level():
    logger = logging.getLogger('docarray')
    logger.setLevel(logging.DEBUG)
