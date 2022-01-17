from collections import defaultdict
from typing import TYPE_CHECKING, Type

import numpy as np

if TYPE_CHECKING:
    from pydantic import BaseModel
    from ...types import T
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
            else:
                _p_dict[f] = v
        return DP(**_p_dict)

    @classmethod
    def from_pydantic_model(cls: Type['T'], model: 'BaseModel') -> 'T':
        """Build a Document object from a Pydantic model

        :param model: the pydantic data model object that represents a Document
        :param ndarray_as_list: if set to True, `embedding` and `tensor` are auto-casted to ndarray.
        :return: a Document object
        """
        from ... import Document

        fields = {}
        if model.chunks:
            fields['chunks'] = [Document.from_pydantic_model(d) for d in model.chunks]
        if model.matches:
            fields['matches'] = [Document.from_pydantic_model(d) for d in model.matches]

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
            else:
                fields[f_name] = value
        return Document(**fields)
