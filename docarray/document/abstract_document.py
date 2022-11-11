from typing import Dict, Iterable

from pydantic.fields import ModelField


class AbstractDocument(Iterable):
    __fields__: Dict[str, ModelField]
