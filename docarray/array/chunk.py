import itertools
from typing import (
    TYPE_CHECKING,
    Generator,
    Iterator,
    Sequence,
)

from .document import DocumentArray

if TYPE_CHECKING:
    from ..document import Document


class ChunkArray(DocumentArray):
    """
    :class:`ChunkArray` inherits from :class:`DocumentArray`.
    It's a subset of Documents.

    :param docs: Set of sub-documents (i.e chunks) of `reference_doc`
    :param reference_doc: Reference :class:`Document` for the sub-documents
    """

    def __init__(self, docs, reference_doc: 'Document'):
        """
        Set constructor method.

        :param doc_views: protobuf representation of the chunks
        :param reference_doc: parent document
        """
        self._ref_doc = reference_doc
        super().__init__(docs)
        if (
            isinstance(
                docs, (DocumentArray, Sequence, Generator, Iterator, itertools.chain)
            )
            and self._ref_doc is not None
        ):
            for d in docs:
                d.parent_id = self._ref_doc.id
                d.granularity = self._ref_doc.granularity + 1

    def append(self, document: 'Document'):
        """Add a sub-document (i.e chunk) to the current Document.

        :param document: Sub-document to be appended

        .. note::
            Comparing to :attr:`DocumentArray.append()`, this method adds more safeguard to
            make sure the added chunk is legit.
        """
        document.parent_id = self._ref_doc.id
        document.granularity = self._ref_doc.granularity + 1
        super().append(document)

    @property
    def reference_doc(self) -> 'Document':
        """
        Get the document that :class:`ChunkArray` belongs to.

        :return: reference doc
        """
        return self._ref_doc

    @property
    def granularity(self) -> int:
        """
        Get granularity of all document in this array.

        :return: granularity
        """
        return self._ref_doc.granularity + 1

    @property
    def adjacency(self) -> int:
        """
        Get adjacency of all document in this array.

        :return: adjacency
        """
        return self._ref_doc.adjacency
