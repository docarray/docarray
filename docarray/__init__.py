__version__ = '0.9.14'

from .document import Document
from .array import DocumentArray

from rich.traceback import install

install(show_locals=True)
