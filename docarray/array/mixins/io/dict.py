from typing import TYPE_CHECKING, Type, List

if TYPE_CHECKING:
    from ....types import T


class DictIOMixin:
    """Save/load a DocumentArray into a dict of the form `{offset_0: doc_0, offset_1: doc_1, ...}`"""

    def to_dict(self, protocol: str = 'jsonschema', **kwargs) -> List:
        """Convert the object into a Python dict of the form `{offset_0: doc_0, offset_1: doc_1, ...}`

        :param protocol: `jsonschema` or `protobuf`
        :return: a Python list
        """
        return {k: d.to_dict(protocol=protocol, **kwargs) for k, d in enumerate(self)}

    @classmethod
    def from_dict(cls: Type['T'], input_dict: dict, *args, **kwargs) -> 'T':
        """Import a :class:`DocumentArray` from a :class:`dict` object of the form `{offset_0: doc_0, offset_1: doc_1, ...}`

        :param input_dict: a `dict` object.
        :return: a :class:`DocumentArray` object
        """
        from .... import Document, DocumentArray

        da = cls(*args, **kwargs).empty(len(input_dict))
        # da = DocumentArray(**kwargs).empty(len(input_dict))

        for offset, d in input_dict.items():
            da[offset] = Document(
                {k: v for k, v in d.items() if (not isinstance(v, float) or v == v)}
            )
        return da
