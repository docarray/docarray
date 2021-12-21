from dataclasses import dataclass, fields, field
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from . import NamedScore

default_values = dict(value=0.0, op_name='', description='', ref_id='')


@dataclass(unsafe_hash=True)
class NamedScoreData:
    _reference_ns: 'NamedScore' = field(hash=False, compare=False)
    value: Optional[float] = None
    op_name: Optional[str] = None
    description: Optional[str] = None
    ref_id: Optional[str] = None

    @property
    def _non_empty_fields(self) -> Tuple[str]:
        r = []
        for f in fields(self):
            f_name = f.name
            if not f_name.startswith('_'):
                v = getattr(self, f_name)
                if v is not None:
                    if f_name not in default_values:
                        r.append(f_name)
                    else:
                        dv = default_values[f_name]
                        if v != dv:
                            r.append(f_name)

        return tuple(r)

    def _set_default_value_if_none(self, key):
        if getattr(self, key) is None:
            setattr(self, key, default_values[key])
