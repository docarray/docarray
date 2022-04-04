import json
from contextlib import nullcontext
from typing import Union, TextIO, TYPE_CHECKING, Type, List

if TYPE_CHECKING:
    from ....typing import T


class JsonIOMixin:
    """Save/load a array into a JSON file."""

    def save_json(
        self,
        file: Union[str, TextIO],
        protocol: str = 'jsonschema',
        encoding: str = 'utf-8',
        **kwargs
    ) -> None:
        """Save array elements into a JSON file.

        Comparing to :meth:`save_binary`, it is human-readable but slower to save/load and the file size larger.

        :param file: File or filename to which the data is saved.
        :param protocol: `jsonschema` or `protobuf`
        :param encoding: encoding used to save data into a JSON file. By default, ``utf-8`` is used.
        """
        if hasattr(file, 'write'):
            file_ctx = nullcontext(file)
        else:
            file_ctx = open(file, 'w', encoding=encoding)

        with file_ctx as fp:
            fp.write(self.to_json(protocol=protocol, **kwargs))

    @classmethod
    def load_json(
        cls: Type['T'],
        file: Union[str, TextIO],
        protocol: str = 'jsonschema',
        encoding: str = 'utf-8',
        **kwargs
    ) -> 'T':
        """Load array elements from a JSON file.

        :param file: File or filename or a JSON string to which the data is saved.
        :param protocol: `jsonschema` or `protobuf`
        :param encoding: encoding used to load data from a JSON file. By default, ``utf-8`` is used.

        :return: a DocumentArrayLike object
        """
        if hasattr(file, 'read'):
            file_ctx = nullcontext(file)
        else:
            file_ctx = open(file, 'r', encoding=encoding)

        with file_ctx as fp:
            return cls.from_json(fp.read(), protocol=protocol, **kwargs)

    @classmethod
    def from_json(
        cls: Type['T'],
        file: Union[str, bytes, bytearray],
        protocol: str = 'jsonschema',
        **kwargs
    ) -> 'T':
        from .... import Document

        json_docs = json.loads(file)
        return cls(
            [Document.from_dict(v, protocol=protocol) for v in json_docs], **kwargs
        )

    @classmethod
    def from_list(
        cls: Type['T'], values: List, protocol: str = 'jsonschema', **kwargs
    ) -> 'T':
        from .... import Document

        return cls(Document.from_dict(v, protocol=protocol, **kwargs) for v in values)

    def to_list(self, protocol: str = 'jsonschema', **kwargs) -> List:
        """Convert the object into a Python list.

        :param protocol: `jsonschema` or `protobuf`
        :return: a Python list
        """
        return [d.to_dict(protocol=protocol, **kwargs) for d in self]

    def to_json(self, protocol: str = 'jsonschema', **kwargs) -> str:
        """Convert the object into a JSON string. Can be loaded via :meth:`.load_json`.

        :param protocol: `jsonschema` or `protobuf`
        :return: a Python list
        """
        return json.dumps(self.to_list(protocol=protocol, **kwargs))

    # to comply with Document interfaces but less semantically accurate
    to_dict = to_list
    from_dict = from_list
