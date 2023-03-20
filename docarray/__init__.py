__version__ = '0.30.0a3'

from docarray.array import DocumentArray, DocumentArrayStacked
from docarray.base_document.document import BaseDocument

__all__ = ['BaseDocument', 'DocumentArray', 'DocumentArrayStacked']

import logging

logger = logging.getLogger('docarray')


handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)
