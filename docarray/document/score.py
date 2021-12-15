from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NamedScore:
    value: Optional[str] = None
    op_name: Optional[str] = None
    description: Optional[str] = None
    operands: Optional[List['NamedScore']] = None
    ref_id: Optional[str] = None
