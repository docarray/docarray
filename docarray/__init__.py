__version__ = '0.30.0a3'

from docarray.array import DocumentArray, DocumentArrayStacked
from docarray.base_document.document import BaseDocument
import logging

__all__ = ['BaseDocument', 'DocumentArray', 'DocumentArrayStacked']

logger = logging.getLogger('docarray')

handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
