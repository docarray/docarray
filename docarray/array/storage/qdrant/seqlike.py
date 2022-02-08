from typing import MutableSequence, Iterable, Iterator, Union
from docarray import Document

from qdrant_client import QdrantClient


class SequenceLikeMixin(MutableSequence[Document]):
    @property
    def client(self) -> QdrantClient:
        raise NotImplementedError()

    @property
    def collection_name(self) -> str:
        raise NotImplementedError()

    @property
    def config(self):
        raise NotImplementedError()

    def insert(self, index: int, value: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        """
        raise NotImplementedError()

    def __eq__(self, other):
        """Compare this object to the other, returns True if and only if other
        as the same type as self and other has the same meta information

        :param other: the other object to check for equality
        :return: ``True`` if other is equal to self
        """
        # two DAW are considered as the same if they have the same client meta data
        return (
            type(self) is type(other)
            and self.client.openapi_client.client.host
            == other.openapi_client.client.host
            and self.config == other.config
        )

    def __len__(self):
        return self.client.http.collections_api.get_collection(
            self.collection_name
        ).vectors_count

    def __iter__(self) -> Iterable['Document']:
        raise NotImplementedError()

    def __contains__(self, x: Union[str, 'Document']):
        raise NotImplementedError()

    def __bool__(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def __add__(self, other: Union['Document', Iterable['Document']]):
        raise NotImplementedError()

    def append(self, value: 'Document'):
        # optional, if you have better implementation than `insert`
        raise NotImplementedError()

    def extend(self, values: Iterable['Document']) -> None:
        # optional, if you have better implementation than `insert` one by one
        raise NotImplementedError()
