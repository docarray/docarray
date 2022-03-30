__version__ = '0.10.3'

import os

from .document import Document
from .array import DocumentArray


if 'DA_NO_RICH_HANDLER' not in os.environ:
    from rich.traceback import install

    install()
