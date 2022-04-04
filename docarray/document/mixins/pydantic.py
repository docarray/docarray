import base64
from collections import defaultdict
from typing import TYPE_CHECKING, Type

import numpy as np

if TYPE_CHECKING:
    from pydantic import BaseModel
    from ...typing import T
    from ..pydantic_model import PydanticDocument


class PydanticMixin:
    """Provide helper functions to convert to/from a Pydantic model"""

    @classmethod
    def get_json_schema(cls, indent: int = 2) -> str:
        """Return a JSON Schema of Document class."""
        from ..pydantic_model import PydanticDocument as DP

        from pydantic import schema_json_of

        return schema_json_of(DP, title='Document Schema', indent=indent)

    def to_pydantic_model(self) -> 'PydanticDocument':
        """Convert a Document object into a Pydantic model."""
        from ..pydantic_model import PydanticDocument as DP

        _p_dict = {}
        for f in self.non_empty_fields:
            v = getattr(self, f)
            if f in ('matches', 'chunks'):
                _p_dict[f] = v.to_pydantic_model()
            elif f in ('scores', 'evaluations'):
                _p_dict[f] = {k: v.to_dict() for k, v in v.items()}
            elif f == 'blob':
                _p_dict[f] = base64.b64encode(v).decode('utf8')
            else:
                _p_dict[f] = v
        return DP(**_p_dict)

    @classmethod
    def from_pydantic_model(cls: Type['T'], model: 'BaseModel') -> 'T':
        """Build a Document object from a Pydantic model

        :param model: the pydantic data model object that represents a Document
        :return: a Document object
        """
        from ... import Document

        fields = {}
        _field_chunks, _field_matches = None, None
        if model.chunks:
            _field_chunks = [Document.from_pydantic_model(d) for d in model.chunks]
        if model.matches:
            _field_matches = [Document.from_pydantic_model(d) for d in model.matches]

        for (field, value) in model.dict(
            exclude_none=True, exclude={'chunks', 'matches'}
        ).items():
            f_name = field
            if f_name == 'scores' or f_name == 'evaluations':
                from docarray.score import NamedScore

                fields[f_name] = defaultdict(NamedScore)
                for k, v in value.items():
                    fields[f_name][k] = NamedScore(v)
            elif f_name == 'embedding' or f_name == 'tensor':
                fields[f_name] = np.array(value)
            elif f_name == 'blob':
                fields[f_name] = base64.b64decode(value)
            else:
                fields[f_name] = value

        d = Document(**fields)
        if _field_chunks:
            d.chunks = _field_chunks
        if _field_matches:
            d.matches = _field_matches
        return d
