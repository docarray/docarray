from dataclasses import is_dataclass

import typing

from docarray.types import ImageDocument, BlobDocument, TextDocument


class MultiModalMixin:
    @classmethod
    def from_dataclass(cls, obj):
        if not is_dataclass:
            raise ValueError('Object is not a dataclass instance')

        from docarray import Document

        root = Document()
        tags = {}
        for position, (key, field) in enumerate(obj.__dataclass_fields__):
            attribute = getattr(obj, key)
            if is_dataclass(field.type):
                root.chunks.append(cls.from_dataclass(attribute))
            elif field.type == ImageDocument:
                root.chunks.append(Document(tensor=attribute, modality='image'))
            elif field.type == BlobDocument:
                root.chunks.append(Document(blob=attribute, modality='blob'))
            elif field.type == TextDocument:
                root.chunks.append(Document(text=attribute, modality='text'))
            elif isinstance(field.type, typing._GenericAlias):
                if field.type._name in ['List', 'Iterable']:
                    sub_type = field.type.__args__[0]
                    if sub_type in [str, int, float]:
                        tags[key] = attribute
                    else:
                        chunk = Document()
                        for element in attribute:
                            chunk.chunks.append(cls.from_dataclass(element))

            elif field.type in [str, int, float]:
                tags[key] = attribute
