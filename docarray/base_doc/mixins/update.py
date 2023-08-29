from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Type, TypeVar

from typing_inspect import get_origin

from docarray.utils._internal._typing import safe_issubclass

T = TypeVar('T', bound='UpdateMixin')

if TYPE_CHECKING:
    from pydantic.fields import ModelField


class UpdateMixin:
    _docarray_fields: Dict[str, 'ModelField']

    def _get_string_for_regex_filter(self):
        return str(self)

    @classmethod
    @abstractmethod
    def _get_field_annotation(cls, field: str) -> Type['UpdateMixin']:
        ...

    def update(self, other: T):
        """
        Updates self with the content of other. Changes are applied to self.
        Updating one Document with another consists in the following:

         - Setting data properties of the second Document to the first Document
         if they are not None
         - Concatenating lists and updating sets
         - Updating recursively Documents and DocLists
         - Updating Dictionaries of the left with the right

        It behaves as an update operation for Dictionaries, except that since
        it is applied to a static schema type, the presence of the field is
        given by the field not having a None value and that DocLists,
        lists and sets are concatenated. It is worth mentioning that Tuples
        are not merged together since they are meant to be immutable,
        so they behave as regular types and the value of `self` is updated
        with the value of `other`.


        ---

        ```python
        from typing import List, Optional

        from docarray import BaseDoc


        class MyDocument(BaseDoc):
            content: str
            title: Optional[str] = None
            tags_: List


        doc1 = MyDocument(
            content='Core content of the document', title='Title', tags_=['python', 'AI']
        )
        doc2 = MyDocument(content='Core content updated', tags_=['docarray'])

        doc1.update(doc2)
        assert doc1.content == 'Core content updated'
        assert doc1.title == 'Title'
        assert doc1.tags_ == ['python', 'AI', 'docarray']
        ```

        ---
        :param other: The Document with which to update the contents of this
        """
        if not _similar_schemas(self, other):
            raise Exception(
                f'Update operation can only be applied to '
                f'Documents of the same schema. '
                f'Trying to update Document of type '
                f'{type(self)} with Document of type '
                f'{type(other)}'
            )
        from collections import namedtuple

        from docarray import DocList
        from docarray.utils.reduce import reduce

        # Declaring namedtuple()
        _FieldGroups = namedtuple(
            '_FieldGroups',
            [
                'simple_non_empty_fields',
                'list_fields',
                'set_fields',
                'dict_fields',
                'nested_docarray_fields',
                'nested_docs_fields',
            ],
        )

        FORBIDDEN_FIELDS_TO_UPDATE = ['ID']

        def _group_fields(doc: 'UpdateMixin') -> _FieldGroups:
            simple_non_empty_fields: List[str] = []
            list_fields: List[str] = []
            set_fields: List[str] = []
            dict_fields: List[str] = []
            nested_docs_fields: List[str] = []
            nested_docarray_fields: List[str] = []

            for field_name, field in doc._docarray_fields().items():
                if field_name not in FORBIDDEN_FIELDS_TO_UPDATE:
                    field_type = doc._get_field_annotation(field_name)

                    if isinstance(field_type, type) and safe_issubclass(
                        field_type, DocList
                    ):
                        nested_docarray_fields.append(field_name)
                    else:
                        origin = get_origin(field_type)
                        if origin is list:
                            list_fields.append(field_name)
                        elif origin is set:
                            set_fields.append(field_name)
                        elif origin is dict:
                            dict_fields.append(field_name)
                        else:
                            v = getattr(doc, field_name)
                            if v is not None:
                                if isinstance(v, UpdateMixin):
                                    nested_docs_fields.append(field_name)
                                else:
                                    simple_non_empty_fields.append(field_name)
            return _FieldGroups(
                simple_non_empty_fields,
                list_fields,
                set_fields,
                dict_fields,
                nested_docarray_fields,
                nested_docs_fields,
            )

        doc1_fields = _group_fields(self)
        doc2_fields = _group_fields(other)

        for field in doc2_fields.simple_non_empty_fields:
            setattr(self, field, getattr(other, field))

        for field in set(
            doc1_fields.nested_docs_fields + doc2_fields.nested_docs_fields
        ):
            sub_doc_1: T = getattr(self, field)
            sub_doc_2: T = getattr(other, field)
            sub_doc_1.update(sub_doc_2)
            setattr(self, field, sub_doc_1)

        for field in set(doc1_fields.list_fields + doc2_fields.list_fields):
            array1 = getattr(self, field)
            array2 = getattr(other, field)
            if array1 is None and array2 is not None:
                setattr(self, field, array2)
            elif array1 is not None and array2 is not None:
                array1.extend(array2)
                setattr(self, field, array1)

        for field in set(doc1_fields.set_fields + doc2_fields.set_fields):
            array1 = getattr(self, field)
            array2 = getattr(other, field)
            if array1 is None and array2 is not None:
                setattr(self, field, array2)
            elif array1 is not None and array2 is not None:
                array1.update(array2)
                setattr(self, field, array1)

        for field in set(
            doc1_fields.nested_docarray_fields + doc2_fields.nested_docarray_fields
        ):
            array1 = getattr(self, field)
            array2 = getattr(other, field)
            if array1 is None and array2 is not None:
                setattr(self, field, array2)
            elif array1 is not None and array2 is not None:
                array1 = reduce(array1, array2)
                setattr(self, field, array1)

        for field in set(doc1_fields.dict_fields + doc2_fields.dict_fields):
            dict1 = getattr(self, field)
            dict2 = getattr(other, field)
            if dict1 is None and dict2 is not None:
                setattr(self, field, dict2)
            elif dict1 is not None and dict2 is not None:
                dict1.update(dict2)
                setattr(self, field, dict1)


def _similar_schemas(model1, model2):
    return model1.__annotations__ == model2.__annotations__
