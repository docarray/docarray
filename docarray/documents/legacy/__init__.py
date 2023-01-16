from docarray import DocumentArray as BaseDocumentArray
from docarray.documents.legacy.legacy_document import Document

DocumentArray = BaseDocumentArray[Document]

__all__ = ['Document', 'DocumentArray']
