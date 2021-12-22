from typing import Type, TYPE_CHECKING

from ... import Document

if TYPE_CHECKING:
    from ...types import T


class EmptyMixin:
    """Helper functions for building arrays with empty Document."""

    @classmethod
    def empty(cls: Type['T'], size: int = 0) -> 'T':
        """Create a :class:`DocumentArray`  object with :attr:`size` empty
        :class:`Document` objects.

        :param size: the number of empty Documents in this container
        :return: a :class:`DocumentArray` object
        """
        return cls(Document() for _ in range(size))
