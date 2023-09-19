from abc import ABC, abstractmethod
from typing import Any, Dict

from docarray.base_doc.doc import BaseDoc
from docarray.utils._internal.pydantic import is_pydantic_v2

if is_pydantic_v2:
    from pydantic_core import CoreSchema, core_schema


class PredefinedDoc(BaseDoc, ABC):
    """
    Custom class for handling predefined documents and can override their init and validation input.
    """

    @abstractmethod
    @classmethod
    def _docarray_custom_val(cls, value: Any) -> Dict[str, Any]:
        ...

    if is_pydantic_v2:

        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler) -> CoreSchema.CoreSchema:
            if '__pydantic_core_schema__' in cls.__dict__:
                if not cls.__pydantic_generic_metadata__['origin']:
                    schema = cls.__pydantic_core_schema__

            schema = handler(source)
            return core_schema.general_wrap_validator_function(
                function=cls._docarray_default_value_validation, schema=schema
            )

        @classmethod
        def _docarray_default_value_validation(cls, value, model_validator, _):
            return model_validator(cls._docarray_custom_val(value))

    else:

        @classmethod
        def validate(cls, value: Any):
            return super().validate(cls._docarray_custom_val(value))
