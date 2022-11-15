from typing import List

from docarray.array.abstract_array import AbstractDocumentArray


class GetAttributeArrayMixin(AbstractDocumentArray):
    """Helpers that provide attributes getter in bulk"""

    def _get_documents_attribute(self, field: str) -> List:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """

        return [getattr(doc, field) for doc in self]
