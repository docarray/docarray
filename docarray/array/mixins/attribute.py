from typing import TYPE_CHECKING, List, Union

from typing_inspect import is_union_type

from docarray.array.abstract_array import AbstractDocumentArray
from docarray.document import BaseDocument

if TYPE_CHECKING:
    from docarray.typing import NdArray, TorchTensor


class GetAttributeArrayMixin(AbstractDocumentArray):
    """Helpers that provide attributes getter in bulk"""

    def _get_array_attribute(
        self,
        field: str,
    ) -> Union[List, AbstractDocumentArray, 'TorchTensor', 'NdArray']:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        field_type = self.__class__.document_type._get_nested_document_class(field)
        if is_union_type(field_type):
            # determine type based on the first element
            field_type = type(getattr(self[0], field))

        if (
            self.is_stacked()
            and field in self._column_fields()
            and self._columns is not None
        ):
            attributes = self._columns[field]
            if attributes is not None:
                return attributes
            else:
                raise ValueError(
                    f'The DocumentArray is in stacked mode, but no stacked data is '
                    f'present for {field}. This is inconsistent'
                )
        elif issubclass(field_type, BaseDocument):
            # calling __class_getitem__ ourselves is a hack otherwise mypy complain
            # most likely a bug in mypy though
            # bug reported here https://github.com/python/mypy/issues/14111
            return self.__class__.__class_getitem__(field_type)(
                (getattr(doc, field) for doc in self)
            )
        else:
            return [getattr(doc, field) for doc in self]

    def _set_array_attribute(
        self,
        field: str,
        values: Union[List, AbstractDocumentArray, 'TorchTensor', 'NdArray'],
    ):
        """Set all Documents in this DocumentArray using the passed values

        :param field: name of the fields to extract
        :values: the values to set at the DocumentArray level
        """
        if (
            self.is_stacked()
            and self._columns is not None
            and field in self._column_fields()
            and not isinstance(values, List)
        ):
            self._columns[field] = values
        else:
            for doc, value in zip(self, values):
                setattr(doc, field, value)
