from typing import (
    TYPE_CHECKING,
    Iterable,
)

from .memory import DocumentArrayInMemory

if TYPE_CHECKING:
    from ..document import Document


class MatchArray(DocumentArrayInMemory):
    """
    :class:`MatchArray` inherits from :class:`DocumentArray`.
    It's a subset of Documents that represents the matches

    :param docs: Set of matches of the `reference_doc`
    :param reference_doc: Reference :class:`Document` for the sub-documents
    """

    def __init__(self, docs, reference_doc: 'Document'):
        self._ref_doc = reference_doc
        super().__init__(docs)
        if isinstance(docs, Iterable) and self._ref_doc is not None:
            for d in docs:
                d.adjacency = self._ref_doc.adjacency + 1

    def append(self, document: 'Document'):
        """Add a matched document to the current Document.

        :param document: Sub-document to be added
        """
        document.adjacency = self._ref_doc.adjacency + 1
        super().append(document)

    @property
    def reference_doc(self) -> 'Document':
        """Get the document that this :class:`MatchArray` referring to.
        :return: the document the match refers to
        """
        return self._ref_doc

    @property
    def granularity(self) -> int:
        """Get granularity of all document in this array.
        :return: the granularity of the documents of which these are match
        """
        return self._ref_doc.granularity

    @property
    def adjacency(self) -> int:
        """Get the adjacency of all document in this array.
        :return: the adjacency of the array of matches
        """
        return self._ref_doc.adjacency + 1
