import json
import os.path
from contextlib import nullcontext
from typing import Union, TextIO, TYPE_CHECKING, Type, List

if TYPE_CHECKING:
    from ....types import T


class JsonIOMixin:
    """Save/load a array into a JSON file."""

    def save_json(self, file: Union[str, TextIO]) -> None:
        """Save array elements into a JSON file.

        Comparing to :meth:`save_binary`, it is human-readable but slower to save/load and the file size larger.

        :param file: File or filename to which the data is saved.
        """
        if hasattr(file, 'write'):
            file_ctx = nullcontext(file)
        else:
            file_ctx = open(file, 'w')

        with file_ctx as fp:
            for d in self:
                json.dump(d.to_dict(), fp)
                fp.write('\n')

    @classmethod
    def load_json(cls: Type['T'], file: Union[str, TextIO]) -> 'T':
        """Load array elements from a JSON file.

        :param file: File or filename or a JSON string to which the data is saved.

        :return: a DocumentArrayLike object
        """

        from .... import Document

        constructor = Document.from_json
        if hasattr(file, 'read'):
            file_ctx = nullcontext(file)
        elif os.path.exists(file):
            file_ctx = open(file)
        else:
            file_ctx = nullcontext(json.loads(file))
            constructor = Document.from_dict

        with file_ctx as fp:
            return cls(constructor(v) for v in fp)

    @classmethod
    def from_json(cls: Type['T'], file: Union[str, TextIO]) -> 'T':
        return cls.load_json(file)

    @classmethod
    def from_list(cls: Type['T'], values: List) -> 'T':
        from .... import Document

        return cls(Document.from_dict(v) for v in values)

    def to_list(self, strict: bool = True) -> List:
        """Convert the object into a Python list.

        .. note::
            Array like object such as :class:`numpy.ndarray` will be converted to Python list.

        :return: a Python list
        """
        return [d.to_dict(strict=strict) for d in self]

    def to_json(self) -> str:
        """Convert the object into a JSON string. Can be loaded via :meth:`.load_json`.

        :return: a Python list
        """
        return json.dumps(self.to_list())
