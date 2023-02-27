from typing import Any, Type

from pydantic import create_model

from docarray import BaseDocument


def create_from_dict(model_name: str, **field_definitions: Any) -> Type[BaseDocument]:
    """
    Dynamically create a subclass of BaseDocument from a dict of field definitions.
    :param model_name: name of the created class
    :param field_definitions: fields of the class (or extra fields if a base is supplied)
        in the format `<name>=(<type>, <default default>)` or `<name>=<default value>`
    :return: the created class
    """
    dynamicclass = create_model(model_name, __base__=BaseDocument, **field_definitions)

    return dynamicclass
