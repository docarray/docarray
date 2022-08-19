__version__ = '0.15.3'

import os

from docarray.document import Document
from docarray.array import DocumentArray
from docarray.dataclasses import dataclass, field

if 'DA_RICH_HANDLER' in os.environ:
    from rich.traceback import install

    install()
