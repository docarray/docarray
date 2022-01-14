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
    def json_schema(cls, indent: int = 2) -> str:
        """Return a JSON Schema of Document class."""
        from ..pydantic_model import PydanticDocument as DP

        return DP.schema_json(indent=indent)

    def to_pydantic_model(self) -> 'PydanticDocument':
        """Convert a Document object into a Pydantic model."""
        from ..pydantic_model import PydanticDocument as DP

        return DP(**{f: getattr(self, f) for f in self.non_empty_fields})

    @classmethod
    def from_pydantic_model(
        cls: Type['T'], model: 'BaseModel', ndarray_as_list: bool = False
    ) -> 'T':
        """Build a Document object from a Pydantic model

        :param model: the pydantic data model object that represents a Document
        :param ndarray_as_list: if set to True, `embedding` and `blob` are auto-casted to ndarray.
        :return: a Document object
        """
        from ... import Document

        fields = {}
        for (field, value) in model.dict(exclude_none=True).items():
            f_name = field
            if f_name == 'chunks' or f_name == 'matches':
                fields[f_name] = [Document.from_pydantic_model(d) for d in value]
            elif f_name == 'scores' or f_name == 'evaluations':
                fields[f_name] = defaultdict(value)
            elif f_name == 'embedding' or f_name == 'blob':
                if not ndarray_as_list:
                    fields[f_name] = np.array(value)
            else:
                fields[f_name] = value
        return Document(**fields)
