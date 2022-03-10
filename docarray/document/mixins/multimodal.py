from dataclasses import is_dataclass

import typing
from enum import Enum

from docarray.types import ImageDocument, BlobDocument, TextDocument

if typing.TYPE_CHECKING:
    from docarray import Document


class AttributeType(Enum):
    DOCUMENT = 1
    PRIMITIVE = 2
    ITERABLE_PRIMITIVE = 3
    ITERABLE_DOCUMENT = 4
    NESTED = 5


class MultiModalMixin:
    @classmethod
    def from_dataclass(cls, obj):
        if not is_dataclass(obj):
            raise ValueError('Object is not a dataclass instance')

        from docarray import Document

        root = Document()
        tags = {}
        multi_modal_schema = {}
        for key, field in obj.__dataclass_fields__.items():
            attribute = getattr(obj, key)
            if field.type in [str, int, float]:
                tags[key] = attribute
                multi_modal_schema[key] = {
                    'attribute_type': AttributeType.PRIMITIVE,
                    'type': field.type.__name__,
                }

            elif isinstance(field.type, typing._GenericAlias):
                if field.type._name in ['List', 'Iterable']:
                    sub_type = field.type.__args__[0]
                    if sub_type in [str, int, float]:
                        tags[key] = attribute
                        multi_modal_schema[key] = {
                            'attribute_type': AttributeType.ITERABLE_PRIMITIVE,
                            'type': f'{field.type._name}[{sub_type.__name__}]',
                        }

                    else:
                        chunk = Document()
                        for element in attribute:
                            doc, _ = cls._from_obj(element, sub_type)
                            chunk.chunks.append(doc)
                        multi_modal_schema[key] = {
                            'attribute_type': AttributeType.ITERABLE_DOCUMENT,
                            'type': f'{field.type._name}[{sub_type.__name__}]',
                            'position': len(root.chunks),
                        }
                        root.chunks.append(chunk)
                else:
                    raise ValueError(f'Unsupported type annotation {field.type._name}')
            else:
                doc, attribute_type = cls._from_obj(attribute, field.type)
                multi_modal_schema[key] = {
                    'attribute_type': attribute_type,
                    'type': field.type.__name__,
                    'position': len(root.chunks),
                }
                root.chunks.append(doc)

        # TODO: may have to modify this?
        root.tags = tags
        root.meta_tags['multi_modal_schema'] = multi_modal_schema

        return root

    @classmethod
    def _from_obj(cls, obj, obj_type) -> typing.Tuple['Document', AttributeType]:
        from docarray import Document

        attribute_type = AttributeType.DOCUMENT

        if is_dataclass(obj_type):
            doc = cls.from_dataclass(obj)
            attribute_type = AttributeType.NESTED
        elif obj_type == ImageDocument:
            doc = Document(tensor=obj, modality='image')
        elif obj_type == BlobDocument:
            doc = Document(blob=obj, modality='blob')
        elif obj_type == TextDocument:
            doc = Document(text=obj, modality='text')
        else:
            raise ValueError(f'Unsupported type annotation')
        return doc, attribute_type
