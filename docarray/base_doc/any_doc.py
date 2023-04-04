from typing import Type

from .doc import BaseDoc


class AnyDoc(BaseDoc):
    """
    AnyDoc is a Document that is not tied to any schema
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    @classmethod
    def _get_field_type(cls, field: str) -> Type['BaseDoc']:
        """
        Accessing the nested python Class define in the schema.
        Could be useful for reconstruction of Document in
        serialization/deserilization
        :param field: name of the field
        :return:
        """
        return AnyDoc
