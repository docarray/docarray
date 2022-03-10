from dataclasses import is_dataclass

import typing

from docarray.types import ImageDocument, BlobDocument, TextDocument


class MultiModalMixin:
    @classmethod
    def from_dataclass(cls, obj):
        if not is_dataclass(obj):
            raise ValueError('Object is not a dataclass instance')

        from docarray import Document

        root = Document()
        tags = {}
        for position, (key, field) in enumerate(obj.__dataclass_fields__.items()):
            attribute = getattr(obj, key)
            if field.type in [str, int, float]:
                tags[key] = attribute
            elif isinstance(field.type, typing._GenericAlias):
                if field.type._name in ['List', 'Iterable']:
                    sub_type = field.type.__args__[0]
                    if sub_type in [str, int, float]:
                        tags[key] = attribute
                    else:
                        chunk = Document()
                        for element in attribute:
                            chunk.chunks.append(cls._from_obj(element, sub_type))
                        root.chunks.append(chunk)
                else:
                    raise ValueError(f'Unsupported type annotation {field.type._name}')
            else:
                root.chunks.append(cls._from_obj(attribute, field.type))

        # TODO: may have to modify this?
        root.tags = tags

        return root

    @classmethod
    def _from_obj(cls, obj, obj_type):
        from docarray import Document

        if is_dataclass(obj_type):
            doc = cls.from_dataclass(obj)
        elif obj_type == ImageDocument:
            doc = Document(tensor=obj, modality='image')
        elif obj_type == BlobDocument:
            doc = Document(blob=obj, modality='blob')
        elif obj_type == TextDocument:
            doc = Document(text=obj, modality='text')
        else:
            raise ValueError(f'Unsupported type annotation')
        return doc
