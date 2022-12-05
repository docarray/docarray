from typing import TYPE_CHECKING, List, Union

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
                    f'The column is not set for the field {field} even though '
                    f'it is in stacked mode'
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
        """Set all document if this DocumentArray with the passed values

        :param field: name of the fields to extract
        :values: the values to set at the document array level
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
