import re
from dataclasses import fields
from docarray.document.data import DocumentData

with open('../docarray/document/mixins/property.py', 'w') as fp:
    fp.write(f'''# auto-generated from {__file__}
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..score import NamedScore
    from ... import DocumentArray
    from ...typing import ArrayType, StructValueType
    from datetime import datetime


class PropertyMixin:

    @property
    def non_empty_fields(self) -> Tuple[str]:
        """Get all non-emtpy fields of this :class:`Document`.

        Non-empty fields are the fields with not-`None` and not-default values.

        :return: field names in a tuple.
        """
        return self._data.non_empty_fields
    ''')
    for f in fields(DocumentData):
        ftype = str(f.type).replace('typing.Dict', 'Dict').replace('typing.List', 'List').replace('datetime.datetime', '\'datetime\'')
        ftype = re.sub(r'typing.Union\[(.*), NoneType]', r'Optional[\g<1>]', ftype)
        ftype = re.sub(r'ForwardRef\((\'.*\')\)', r'\g<1>', ftype)
        ftype = re.sub(r'<class \'(.*)\'>', r'\g<1>', ftype)

        fp.write(f'''
    @property
    def {f.name}(self) -> {ftype}:
        self._data._set_default_value_if_none('{f.name}')
        return self._data.{f.name}

    @{f.name}.setter
    def {f.name}(self, value: {ftype}):
        self._data.{f.name} = value
        ''')
