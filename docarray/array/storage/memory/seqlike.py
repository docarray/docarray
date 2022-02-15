from typing import Iterator, Union, Iterable

from ..base.seqlike import BaseSequenceLikeMixin
from .... import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods"""

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and type(self._data) is type(other._data)
            and self._data == other._data
        )

    def __len__(self):
        return len(self._data)

    def __iter__(self) -> Iterator['Document']:
        yield from self._data.values()

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, str):
            return x in self._data
        elif isinstance(x, Document):
            return x.id in self._data
        else:
            return False

    def __repr__(self):
        return f'<DocumentArray (length={len(self)}) at {id(self)}>'

    def __add__(self, other: Union['Document', Iterable['Document']]):
        v = type(self)(self)
        v.extend(other)
        return v
