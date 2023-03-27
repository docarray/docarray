__version__ = '0.30.0a3'

import logging

from docarray.array import DocArray, DocArrayStacked
from docarray.base_doc.doc import BaseDoc

__all__ = ['BaseDoc', 'DocArray', 'DocArrayStacked']

logger = logging.getLogger('docarray')

handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
