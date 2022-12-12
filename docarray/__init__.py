__version__ = '0.1.0'

from docarray.array import DocumentArray
from docarray.document.document import BaseDocument as Document
from docarray.predefined_document import Image, Mesh3D, PointCloud3D, Text

__all__ = ['Document', 'DocumentArray', 'Image', 'Text', 'Mesh3D', 'PointCloud3D']
