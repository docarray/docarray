import dataclasses
from collections import defaultdict
from typing import TYPE_CHECKING, Type, List

if TYPE_CHECKING:
    from ...typing import T
    from ..strawberry_type import StrawberryDocument


class StrawberryMixin:
    """Provide helper functions to convert to/from a Strawberry model"""

    def to_strawberry_type(self) -> 'StrawberryDocument':
        """Convert a Document object into a Strawberry type."""
        from ..strawberry_type import StrawberryDocument as SD
        from ..strawberry_type import _NameScoreItem, _NamedScore

        _p_dict = {}
        for f in self.non_empty_fields:
            v = getattr(self, f)
            if f in ('matches', 'chunks'):
                _p_dict[f] = v.to_strawberry_type()
            elif f in ('scores', 'evaluations'):
                _p_dict[f] = [
                    _NameScoreItem(k, _NamedScore(**v.to_dict())) for k, v in v.items()
                ]

            else:
                _p_dict[f] = v
        return SD(**_p_dict)

    @classmethod
    def from_strawberry_type(cls: Type['T'], model) -> 'T':
        """Build a Document object from a Strawberry model

        :param model: the Strawberry data model object that represents a Document
        :return: a Document object
        """
        from ... import Document

        fields = {}
        _field_chunks, _field_matches = None, None
        if model.chunks:
            _field_chunks = [Document.from_strawberry_type(d) for d in model.chunks]
        if model.matches:
            _field_matches = [Document.from_strawberry_type(d) for d in model.matches]

        for field in dataclasses.fields(model):
            f_name = field.name
            value = getattr(model, f_name)
            if value is None:
                continue
            if f_name == 'scores' or f_name == 'evaluations':
                from docarray.score import NamedScore
                from ..strawberry_type import _NameScoreItem

                value: List[_NameScoreItem]
                fields[f_name] = defaultdict(NamedScore)
                for v in value:
                    fields[f_name][v.name] = NamedScore(**dataclasses.asdict(v.score))
            else:
                fields[f_name] = value

        d = Document(**fields)
        if _field_chunks:
            d.chunks = _field_chunks
        if _field_matches:
            d.matches = _field_matches
        return d
