import base64
import typing

from ...dataclasses.types import Field, is_multimodal, _is_field
from ...dataclasses.types import AttributeType

if typing.TYPE_CHECKING:
    from docarray import Document, DocumentArray


class MultiModalMixin:
    @property
    def is_multimodal(self) -> bool:
        """
        Return true if this Document can be represented by a class wrapped
        by :meth:`docarray.dataclasses.types.dataclass`.
        """
        return 'multi_modal_schema' in self._metadata

    @classmethod
    def _from_dataclass(cls, obj) -> 'Document':
        if not is_multimodal(obj):
            raise TypeError(
                f'Object {type(obj).__name__} is not a `docarray.dataclass` instance'
            )

        from docarray import Document

        root = Document()
        tags = {}
        multi_modal_schema = {}
        for key, field in obj.__dataclass_fields__.items():
            attribute = getattr(obj, key)
            if attribute is None:
                continue

            if field.type in [str, int, float, bool] and not _is_field(field):
                tags[key] = attribute
                multi_modal_schema[key] = {
                    'attribute_type': AttributeType.PRIMITIVE,
                    'type': field.type.__name__,
                }

            elif field.type == bytes and not _is_field(field):
                tags[key] = base64.b64encode(attribute).decode()
                multi_modal_schema[key] = {
                    'attribute_type': AttributeType.PRIMITIVE,
                    'type': field.type.__name__,
                }
            elif isinstance(field.type, typing._GenericAlias):
                if field.type._name in ['List', 'Iterable']:
                    sub_type = field.type.__args__[0]
                    if sub_type in [str, int, float, bool]:
                        tags[key] = attribute
                        multi_modal_schema[key] = {
                            'attribute_type': AttributeType.ITERABLE_PRIMITIVE,
                            'type': f'{field.type._name}[{sub_type.__name__}]',
                        }

                    else:
                        chunk = Document()
                        for element in attribute:
                            doc, attribute_type = cls._from_obj(
                                element, sub_type, field
                            )
                            if attribute_type == AttributeType.DOCUMENT:
                                attribute_type = AttributeType.ITERABLE_DOCUMENT
                            elif attribute_type == AttributeType.NESTED:
                                attribute_type = AttributeType.ITERABLE_NESTED
                            else:
                                raise TypeError(
                                    f'Unsupported type annotation inside Iterable: {sub_type}'
                                )
                            chunk.chunks.append(doc)
                        multi_modal_schema[key] = {
                            'attribute_type': attribute_type,
                            'type': f'{field.type._name}[{sub_type.__name__}]',
                            'position': len(root.chunks),
                        }
                        root.chunks.append(chunk)
                else:
                    raise TypeError(
                        f'Unsupported type annotation on field `{field.type._name}`'
                    )
            else:
                doc, attribute_type = cls._from_obj(attribute, field.type, field)
                multi_modal_schema[key] = {
                    'attribute_type': attribute_type,
                    'type': field.type.__name__,
                    'position': len(root.chunks),
                }
                root.chunks.append(doc)

        # TODO: may have to modify this?
        root.tags = tags
        root._metadata['multi_modal_schema'] = multi_modal_schema

        return root

    def get_multi_modal_attribute(self, attribute: str) -> 'DocumentArray':
        from docarray import DocumentArray

        if not self.is_multimodal:
            raise ValueError(
                'the Document does not correspond to a Multi Modal Document'
            )

        if attribute not in self._metadata['multi_modal_schema']:
            raise ValueError(
                f'the Document schema does not contain attribute {attribute}'
            )

        attribute_type = self._metadata['multi_modal_schema'][attribute][
            'attribute_type'
        ]
        position = self._metadata['multi_modal_schema'][attribute].get('position')

        if attribute_type in [AttributeType.DOCUMENT, AttributeType.NESTED]:
            return DocumentArray([self.chunks[position]])
        elif attribute_type in [
            AttributeType.ITERABLE_DOCUMENT,
            AttributeType.ITERABLE_NESTED,
        ]:
            return self.chunks[position].chunks
        else:
            raise ValueError(
                f'Invalid attribute {attribute}: must a Document attribute or nested dataclass'
            )

    @classmethod
    def _from_obj(cls, obj, obj_type, field) -> typing.Tuple['Document', AttributeType]:
        attribute_type = AttributeType.DOCUMENT

        if is_multimodal(obj_type):
            doc = cls(obj)
            attribute_type = AttributeType.NESTED
        elif _is_field(field):
            doc = field.setter(obj)
        else:
            raise ValueError(f'Unsupported type annotation')
        return doc, attribute_type
