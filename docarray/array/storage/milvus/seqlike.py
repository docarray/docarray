from typing import Iterable, Iterator, Union, TYPE_CHECKING
from docarray.array.storage.base.seqlike import BaseSequenceLikeMixin

if TYPE_CHECKING:
    from docarray import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    def __eq__(self, other):
        ...

    def __contains__(self, x: Union[str, 'Document']):
        ...

    def __repr__(self):
        ...

    def __add__(self, other: Union['Document', Iterable['Document']]):
        ...

    def insert(self, index: int, value: 'Document'):
        # Optional. By default, this will add a new item and update offset2id
        # if you want to customize this, make sure to handle offset2id
        ...

    def _append(self, value: 'Document'):
        # Optional. Override this if you have a better implementation than inserting at the last position
        ...

    def _extend(self, values: Iterable['Document']) -> None:
        # Optional. Override this if you have better implementation than appending one by one
        ...

    def __len__(self):
        # Optional. By default, this will rely on offset2id to get the length
        ...

    def __iter__(self) -> Iterator['Document']:
        # Optional. By default, this will rely on offset2id to iterate
        ...
