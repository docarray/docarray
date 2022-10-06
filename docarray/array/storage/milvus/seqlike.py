from typing import Iterable, Iterator, Union, TYPE_CHECKING
from docarray.array.storage.base.seqlike import BaseSequenceLikeMixin

if TYPE_CHECKING:
    from docarray import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    def __eq__(self, other):
        """Compare this object to the other, returns True if and only if other
        as the same type as self and other have the same Milvus Collections for data and offset2id

        :param other: the other object to check for equality
        :return: `True` if other is equal to self
        """
        # two DAW are considered as the same if they have the same client meta data
        return (
            type(self) is type(other)
            and self._collection.name == other._collection.name
            and self._offset2id_collection.name == other._offset2id_collection.name
            and self._config == other._config
        )

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, Document):
            x = x.id
        try:
            self._get_doc_by_id(x)
            return True
        except:  # TODO(johannes) make exception more specific
            return False

    def __repr__(self):
        return f'<DocumentArray[Milvus] (length={len(self)}) at {id(self)}>'

    def __add__(self, other: Union['Document', Iterable['Document']]):
        if isinstance(other, Document):
            self.append(other)
        else:
            self.extend(other)
        return self

    # def insert(self, index: int, value: 'Document'):
    #     # Optional. By default, this will add a new item and update offset2id
    #     # if you want to customize this, make sure to handle offset2id
    #     ...
    #
    # def _append(self, value: 'Document'):
    #     # Optional. Override this if you have a better implementation than inserting at the last position
    #     ...
    #
    # def _extend(self, values: Iterable['Document']) -> None:
    #     # Optional. Override this if you have better implementation than appending one by one
    #     ...
    #
    # def __len__(self):
    #     # Optional. By default, this will rely on offset2id to get the length
    #     ...
    #
    # def __iter__(self) -> Iterator['Document']:
    #     # Optional. By default, this will rely on offset2id to iterate
    #     ...
