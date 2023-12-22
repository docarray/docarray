__version__ = '0.40.0'

import logging

from docarray.array import DocList, DocVec
from docarray.base_doc.doc import BaseDoc
from docarray.utils._internal.misc import _get_path_from_docarray_root_level

__all__ = ['BaseDoc', 'DocList', 'DocVec']

logger = logging.getLogger('docarray')

handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def __getattr__(name: str):
    if name in ['Document', 'DocumentArray']:
        raise ImportError(
            f'Cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\'.\n'
            f'The object named \'{name}\' does not exist anymore in this version of docarray.\n'
            f'If you still want to use \'{name}\' please downgrade to version <=0.21.0 '
            f'with: `pip install -U docarray==0.21.0`.'
        )
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )
