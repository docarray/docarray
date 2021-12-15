import re
from dataclasses import fields
from docarray.document.data import DocumentData

with open('../docarray/document/mixins/property.py', 'w') as fp:
    fp.write(f'''# auto-generated from {__file__}
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..score import NamedScore
    from ... import DocumentArray
    from ...typing import ArrayType, DocumentContentType, StructValueType
    from datetime import datetime


class PropertyMixin:    
    ''')
    for f in fields(DocumentData):
        ftype = str(f.type).replace('typing.Dict', 'Dict').replace('typing.List', 'List').replace('datetime.datetime', '\'datetime\'')
        ftype = re.sub(r'typing.Union\[(.*), NoneType]', r'Optional[\g<1>]', ftype)
        ftype = re.sub(r'ForwardRef\((\'.*\')\)', r'\g<1>', ftype)
        ftype = re.sub(r'<class \'(.*)\'>', r'\g<1>', ftype)

        fp.write(f'''
    @property
    def {f.name}(self) -> {ftype}:
        self._set_default_value_if_none('{f.name}')
        return self._doc_data.{f.name}

    @{f.name}.setter
    def {f.name}(self, value: {ftype}):
        self._doc_data.{f.name} = value
        ''')
