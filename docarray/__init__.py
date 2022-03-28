__version__ = '0.9.19'

import os

from .array import DocumentArray
from .document import Document

if 'DA_NO_RICH_HANDLER' not in os.environ:
    from rich.traceback import install

    install()
