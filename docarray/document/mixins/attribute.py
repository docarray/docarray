from typing import List, Union, Any

from ...helper import dunder_get


class GetAttributesMixin:
    """Provide helper functions for :class:`Document` to allow advanced set and get attributes """

    def _get_attributes(self, *fields: str) -> Union[Any, List[Any]]:
        """Bulk fetch Document fields and return a list of the values of these fields

        :param fields: the variable length values to extract from the document
        :return: a list with the attributes of this document ordered as the args
        """

        ret = []
        for k in fields:
            if '__' in k:
                value = dunder_get(self, k)
            else:
                value = getattr(self, k)

            ret.append(value)

        # unboxing if args is single
        if len(fields) == 1:
            ret = ret[0]

        return ret
