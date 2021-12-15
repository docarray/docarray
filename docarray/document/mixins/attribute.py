import warnings
from typing import List, Union, Any

from ...helper import typename, dunder_get


class GetAttributesMixin:
    """Provide helper functions for :class:`Document` to allow advanced set and get attributes """

    def get_attributes(self, *fields: str) -> Union[Any, List[Any]]:
        """Bulk fetch Document fields and return a list of the values of these fields

        .. note::
            Arguments will be extracted using `dunder_get`
            .. highlight:: python
            .. code-block:: python

                d = Document({'id': '123', 'hello': 'world', 'tags': {'id': 'external_id', 'good': 'bye'}})

                assert d.id == '123'  # true
                assert d.tags['hello'] == 'world' # true
                assert d.tags['good'] == 'bye' # true
                assert d.tags['id'] == 'external_id' # true

                res = d.get_attrs_values(*['id', 'tags__hello', 'tags__good', 'tags__id'])

                assert res == ['123', 'world', 'bye', 'external_id']

        :param fields: the variable length values to extract from the document
        :return: a list with the attributes of this document ordered as the args
        """

        ret = []
        for k in fields:
            try:
                if '__' in k:
                    value = dunder_get(self, k)
                else:
                    value = getattr(self, k)

                if value is None:
                    raise ValueError

                ret.append(value)
            except (AttributeError, ValueError):
                warnings.warn(
                    f'Could not get attribute `{typename(self)}.{k}`, returning `None`'
                )
                ret.append(None)

        # unboxing if args is single
        if len(fields) == 1:
            ret = ret[0]

        return ret
