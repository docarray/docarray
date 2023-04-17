__version__ = '0.30.1'

import logging

from docarray.array import DocList, DocVec
from docarray.base_doc.doc import BaseDoc

__all__ = ['BaseDoc', 'DocList', 'DocVec']

logger = logging.getLogger('docarray')

handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
