from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar

import orjson
from pydantic import BaseModel, Field, parse_obj_as
from rich.console import Console
from typing_inspect import get_origin

from docarray.base_document.abstract_document import AbstractDocument
from docarray.base_document.base_node import BaseNode
from docarray.base_document.io.json import orjson_dumps, orjson_dumps_and_decode
from docarray.base_document.mixins import PlotMixin, ProtoMixin
from docarray.typing import ID

_console: Console = Console()

T = TypeVar('T', bound='BaseDocument')


class BaseDocument(BaseModel, PlotMixin, ProtoMixin, AbstractDocument, BaseNode):
    """
    The base class for Document
    """

    id: ID = Field(default_factory=lambda: parse_obj_as(ID, os.urandom(16).hex()))

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps_and_decode
        json_encoders = {dict: orjson_dumps}

        validate_assignment = True

    @classmethod
    def _get_field_type(cls, field: str) -> Type['BaseDocument']:
        """
        Accessing the nested python Class define in the schema. Could be useful for
        reconstruction of Document in serialization/deserilization
        :param field: name of the field
        :return:
        """
        return cls.__fields__[field].outer_type_

    def __str__(self):
        with _console.capture() as capture:
            _console.print(self)

        return capture.get().strip()

    def _get_string_for_regex_filter(self):
        return str(self)

    def update(self, other: 'BaseDocument'):
        """
        Updates self with the content of other. Changes are applied to self.
        Updating one Document with another consists in the following:
         - setting data properties of the second Document to the first Document
         if they are not None
         - Concatenating lists and updating sets
         - Updating recursively Documents and DocumentArrays
         - Updating Dictionaries of the left with the right

        It behaves as an update operation for Dictionaries, except that since
        it is applied to a static schema type, the presence of the field is
        given by the field not having a None value and that DocumentArrays,
        lists and sets are concatenated. It is worth mentioning that Tuples
        are not merged together since they are meant to be inmutable,
        so they behave as regular types and the value of `self` is updated
        with the value of `other`

            EXAMPLE USAGE

            .. code-block:: python

                from docarray import BaseDocument
                from docarray.documents import Text


                class MyDocument(BaseDocument):
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

        :param other: The Document with which to update the contents of this
        """
        if type(self) != type(other):
            raise Exception(
                f'Update operation can only be applied to '
                f'Documents of the same type. '
                f'Trying to update Document of type '
                f'{type(self)} with Document of type '
                f'{type(other)}'
            )
        from collections import namedtuple

        from docarray import DocumentArray
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

        def _group_fields(doc: 'BaseDocument') -> _FieldGroups:
            simple_non_empty_fields: List[str] = []
            list_fields: List[str] = []
            set_fields: List[str] = []
            dict_fields: List[str] = []
            nested_docs_fields: List[str] = []
            nested_docarray_fields: List[str] = []

            for field_name, field in doc.__fields__.items():
                if field_name not in FORBIDDEN_FIELDS_TO_UPDATE:
                    field_type = doc._get_field_type(field_name)

                    if isinstance(field_type, type) and issubclass(
                        field_type, DocumentArray
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
                            if v:
                                if isinstance(v, BaseDocument):
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
            sub_doc_1: BaseDocument = getattr(self, field)
            sub_doc_2: BaseDocument = getattr(other, field)
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

    @classmethod
    def smart_parse_obj(
        cls: Type[T],
        obj: Mapping[str, Any],
        cast_map: Optional[Mapping[str, str]] = None,
    ) -> T:
        """
        Parse the given object, validate the data and return a new initialized object of
         type `cls`. The difference from
         :func:`~docarray.document.BaseDocument.parse_obj` is that it will smartly
         resolve conflicts and undefined keys.

        To do this smart cast we follow the simple following algorithm:

        To instantiate an object from this class, `smart_parse_obj()` selects fields from the
        input dictionary that are valid candidates to respect the Document
        schema.

        To do so docarray do:

        1 - Start by selecting all the fields from `obj` that match the Document
        schema field in name and in type. (The value from obj[field_name] should be
        an instance of the field type)

        if all the fields are collected  we are over, otherwise:

        2 - For each of the remaining field of the schema, this function iterate over
        the remaining value in obj and select the value that match the field type.
        This is deterministic since Dict are ordered by default.

        lets take an example

        .. code-block:: python
            from docarray import BaseDocument
            from docarray.typing import AnyUrl, NdArray


            class A(BaseDocument):
                url_0: AnyUrl
                url: AnyUrl

                tensor: NdArray
                tensor_2: NdArray


            class B(BaseDocument):
                url: AnyUrl
                array: NdArray


            a = A(url='file.png', tensor=np.zeros(3))

            b = B.smart_parse_obj(a.dict())
            print(b.url == a.url)  # True
            print((b.array == a.tensor).all())  # True

        in this example the `url` field of both schema matches in type, therefore
        the field `url` is selected over `url_0`. So far there is still a missing
        NdArray object for the `array`field. Since they are no `array` field in A picked
        the first field that is NdArray, in this cas `tensor`

        You can as well use the `cast_map`argument to explicit precise the mapping for
        certain field. For instance with the Document defines above:

        .. code-block:: python
            a = A(url='file.png', url_0='hello', tensor=np.zeros(3), tensor_2=np.zeros(4))
            b = B.smart_parse_obj(a.dict(), {'url': 'url_0'})
            print(b.url == a.url_0)  # True
            print((b.array == a.tensor).all())  # True

        :param obj: a dict of field and type that will be parsed into the Document
        :param cast_map: define an explicit casting from the schema
        of this Document to your object input.
        :return: a Document initialized from the dict data
        """
        fields: Dict[str, Any] = {}

        if cast_map:
            for field_schema, field_obj in cast_map.items():
                fields[field_schema] = obj[field_obj]

        if len(fields) == len(cls.__fields__):
            # if we already collected all the fields we can skip the rest
            return cls(**fields)

        remaining_field_name = [
            field_name
            for field_name in cls.__fields__.keys()
            if field_name not in fields.keys()
        ]

        for field_name in remaining_field_name & obj.keys():
            field_type = cls._get_field_type(field_name)  # TODO handle Union
            if issubclass(field_type, type) and isinstance(
                obj[field_name], cls._get_field_type(field_name)
            ):
                fields[field_name] = obj[field_name]

        if len(fields) == len(cls.__fields__):
            # if we already collected all the fields we can skip the rest
            return cls(**fields)

        # fields contains for now the field that match in name and in type
        remaining_field_name = [
            field_name
            for field_name in cls.__fields__.keys()
            if field_name not in fields.keys()
        ]

        remaining_field_obj = {
            field_name: field
            for field_name, field in obj.items()
            if field_name not in fields.keys()
        }

        for _, val in remaining_field_obj.items():
            for field_name in remaining_field_name:
                if isinstance(val, cls._get_field_type(field_name)):
                    fields[field_name] = val
                    break

            if len(fields) == len(cls.__fields__):
                return cls(**fields)

        raise ValueError('Cannot smartly cast the obj')
