# auto-generated from /Users/hanxiao/Documents/docarray/scripts/gen_ns_property_mixin.py
from typing import Optional


class PropertyMixin:
    @property
    def value(self) -> Optional[float]:
        self._data._set_default_value_if_none('value')
        return self._data.value

    @value.setter
    def value(self, value: float):
        self._data.value = value

    @property
    def op_name(self) -> Optional[str]:
        self._data._set_default_value_if_none('op_name')
        return self._data.op_name

    @op_name.setter
    def op_name(self, value: str):
        self._data.op_name = value

    @property
    def description(self) -> Optional[str]:
        self._data._set_default_value_if_none('description')
        return self._data.description

    @description.setter
    def description(self, value: str):
        self._data.description = value

    @property
    def ref_id(self) -> Optional[str]:
        self._data._set_default_value_if_none('ref_id')
        return self._data.ref_id

    @ref_id.setter
    def ref_id(self, value: str):
        self._data.ref_id = value
