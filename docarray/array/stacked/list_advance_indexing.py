from functools import wraps
from typing import Callable, MutableSequence, TypeVar

from docarray.array.array.sequence_indexing_mixin import IndexingSequenceMixin

T_item = TypeVar('T_item')


def _delegate_meth_to_data(meth_name: str) -> Callable:
    """
    create a function that mimic a function call to the data attribute of the
    ListAdvanceIndex

    :param meth_name: name of the method
    :return: a method that mimic the meth_name
    """
    func = getattr(list, meth_name)

    @wraps(func)
    def _delegate_meth(self, *args, **kwargs):
        return getattr(self._data, meth_name)(*args, **kwargs)

    return _delegate_meth


class ListAdvanceIndex(IndexingSequenceMixin[T_item]):
    """
    A list wrapper that implement custom indexing

    You can index into a ListAdvanceIndex like a numpy array or torch tensor:

    .. code-block:: python
        da[0]  # index by position
        da[0:5:2]  # index by slice
        da[[0, 2, 3]]  # index by list of indices
        da[True, False, True, True, ...]  # index by boolean mask

    You can delete items from a DocumentArray like a Python List

    .. code-block:: python
        del da[0]  # remove first element from DocumentArray
        del da[0:5]  # remove elements for 0 to 5 from DocumentArray

    """

    _data: MutableSequence[T_item]

    def __init__(self, data: MutableSequence[T_item]):
        self._data = data

    pop = _delegate_meth_to_data('pop')
    remove = _delegate_meth_to_data('remove')
    reverse = _delegate_meth_to_data('reverse')
    sort = _delegate_meth_to_data('sort')
    __len__ = _delegate_meth_to_data('__len__')
