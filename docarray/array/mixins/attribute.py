from typing import List, Union

from docarray.array.abstract_array import AbstractDocumentArray
from docarray.document import BaseDocument


class GetAttributeArrayMixin(AbstractDocumentArray):
    """Helpers that provide attributes getter in bulk"""

    def _get_documents_attribute(
        self, field: str
    ) -> Union[List, AbstractDocumentArray]:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """

        field_type = self.__class__.document_type._get_nested_document_class(field)

        if issubclass(field_type, BaseDocument):
            # calling __class_getitem__ ourselves is a hack otherwise mypy complain
            # most likely a bug in mypy though
            # bug reported here https://github.com/python/mypy/issues/14111
            return self.__class__.__class_getitem__(field_type)(
                (getattr(doc, field) for doc in self)
            )
        else:
            return [getattr(doc, field) for doc in self]
