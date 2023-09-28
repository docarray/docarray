from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from docarray.base_doc.doc import BaseDocWithoutId
from docarray import BaseDoc, DocList
from docarray.typing import AnyTensor
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils._internal.pydantic import is_pydantic_v2

RESERVED_KEYS = [
    'type',
    'anyOf',
    '$ref',
    'additionalProperties',
    'allOf',
    'items',
    'definitions',
    'properties',
    'default',
]


def create_pure_python_type_model(model: BaseModel) -> BaseDoc:
    """
    Take a Pydantic model and cast DocList fields into List fields.

    This may be necessary due to limitations in Pydantic:

    https://github.com/docarray/docarray/issues/1521
    https://github.com/pydantic/pydantic/issues/1457

    ---

    ```python
    from docarray import BaseDoc


    class MyDoc(BaseDoc):
        tensor: Optional[AnyTensor]
        url: ImageUrl
        title: str
        texts: DocList[TextDoc]


    MyDocCorrected = create_new_model_cast_doclist_to_list(CustomDoc)
    ```

    ---
    :param model: The input model
    :return: A new subclass of BaseDoc, where every DocList type in the schema is replaced by List.
    """
    fields: Dict[str, Any] = {}
    import copy

    fields_copy = copy.deepcopy(model.__fields__)
    annotations_copy = copy.deepcopy(model.__annotations__)
    for field_name, field in annotations_copy.items():
        if field_name not in fields_copy:
            continue

        if is_pydantic_v2:
            field_info = fields_copy[field_name]
        else:
            field_info = fields_copy[field_name].field_info
        try:
            if safe_issubclass(field, DocList):
                t: Any = field.doc_type
                fields[field_name] = (List[t], field_info)
            else:
                fields[field_name] = (field, field_info)
        except TypeError:
            fields[field_name] = (field, field_info)

    return create_model(model.__name__, __base__=model, __doc__=model.__doc__, **fields)


def _get_field_annotation_from_schema(
    field_schema: Dict[str, Any],
    field_name: str,
    root_schema: Dict[str, Any],
    cached_models: Dict[str, Any],
    is_tensor: bool = False,
    num_recursions: int = 0,
    definitions: Optional[Dict] = None,
) -> type:
    """
    Private method used to extract the corresponding field type from the schema.
    :param field_schema: The schema from which to extract the type
    :param field_name: The name of the field to be created
    :param root_schema: The schema of the root object, important to get references
    :param cached_models: Parameter used when this method is called recursively to reuse partial nested classes.
    :param is_tensor: Boolean used to tell between tensor and list
    :param num_recursions: Number of recursions to properly handle nested types (Dict, List, etc ..)
    :param definitions: Parameter used when this method is called recursively to reuse root definitions of other schemas.
    :return: A type created from the schema
    """
    if not definitions:
        definitions = {}
    field_type = field_schema.get('type', None)
    tensor_shape = field_schema.get('tensor/array shape', None)
    ret: Any
    if 'anyOf' in field_schema:
        any_of_types = []
        for any_of_schema in field_schema['anyOf']:
            if '$ref' in any_of_schema:
                obj_ref = any_of_schema.get('$ref')
                ref_name = obj_ref.split('/')[-1]
                any_of_types.append(
                    create_base_doc_from_schema(
                        root_schema['definitions'][ref_name],
                        ref_name,
                        cached_models=cached_models,
                        definitions=definitions,
                    )
                )
            else:
                any_of_types.append(
                    _get_field_annotation_from_schema(
                        any_of_schema,
                        field_name,
                        root_schema=root_schema,
                        cached_models=cached_models,
                        is_tensor=tensor_shape is not None,
                        num_recursions=0,
                        definitions=definitions,
                    )
                )  # No Union of Lists
        ret = Union[tuple(any_of_types)]
        for rec in range(num_recursions):
            ret = List[ret]
    elif field_type == 'string':
        ret = str
        for rec in range(num_recursions):
            ret = List[ret]
    elif field_type == 'integer':
        ret = int
        for rec in range(num_recursions):
            ret = List[ret]
    elif field_type == 'number':
        if num_recursions <= 1:
            # This is a hack because AnyTensor is more generic than a simple List and it comes as simple List
            if is_tensor:
                ret = AnyTensor
            else:
                ret = List[float]
        else:
            ret = float
            for rec in range(num_recursions):
                ret = List[ret]
    elif field_type == 'boolean':
        ret = bool
        for rec in range(num_recursions):
            ret = List[ret]
    elif field_type == 'object' or field_type is None:
        doc_type: Any
        if 'additionalProperties' in field_schema:  # handle Dictionaries
            additional_props = field_schema['additionalProperties']
            if additional_props.get('type') == 'object':
                doc_type = create_base_doc_from_schema(
                    additional_props, field_name, cached_models=cached_models
                )
                ret = Dict[str, doc_type]
            else:
                ret = Dict[str, Any]
        else:
            obj_ref = field_schema.get('$ref') or field_schema.get('allOf', [{}])[
                0
            ].get('$ref', None)
            if num_recursions == 0:  # single object reference
                if obj_ref:
                    ref_name = obj_ref.split('/')[-1]
                    ret = create_base_doc_from_schema(
                        definitions[ref_name],
                        ref_name,
                        cached_models=cached_models,
                        definitions=definitions,
                    )
                else:
                    ret = Any
            else:  # object reference in definitions
                if obj_ref:
                    ref_name = obj_ref.split('/')[-1]
                    doc_type = create_base_doc_from_schema(
                        definitions[ref_name],
                        ref_name,
                        cached_models=cached_models,
                        definitions=definitions,
                    )
                    ret = DocList[doc_type]
                else:
                    doc_type = create_base_doc_from_schema(
                        field_schema, field_name, cached_models=cached_models
                    )
                    ret = DocList[doc_type]
    elif field_type == 'array':
        ret = _get_field_annotation_from_schema(
            field_schema=field_schema.get('items', {}),
            field_name=field_name,
            root_schema=root_schema,
            cached_models=cached_models,
            is_tensor=tensor_shape is not None,
            num_recursions=num_recursions + 1,
            definitions=definitions,
        )
    elif field_type == 'null':
        ret = None
    else:
        if num_recursions > 0:
            raise ValueError(
                f"Unknown array item type: {field_type} for field_name {field_name}"
            )
        else:
            raise ValueError(
                f"Unknown field type: {field_type} for field_name {field_name}"
            )
    return ret


def create_base_doc_from_schema(
    schema: Dict[str, Any],
    base_doc_name: str,
    cached_models: Optional[Dict] = None,
    definitions: Optional[Dict] = None,
) -> Type:
    """
    Dynamically create a `BaseDoc` subclass from a `schema` of another `BaseDoc`.

    This method is intended to dynamically create a `BaseDoc` compatible with the schema
    of another BaseDoc. This is useful when that other `BaseDoc` is not available in the current scope. For instance, you may have stored the schema
    as a JSON, or sent it to another service, etc.

    Due to this Pydantic limitation (https://github.com/docarray/docarray/issues/1521, https://github.com/pydantic/pydantic/issues/1457), we need to make sure that the
    input schema uses `List` and not `DocList`. Therefore this is recommended to be used in combination with `create_new_model_cast_doclist_to_list`
    to make sure that `DocLists` in schema are converted to `List`.

    ---

    ```python
    from docarray import BaseDoc


    class MyDoc(BaseDoc):
        tensor: Optional[AnyTensor]
        url: ImageUrl
        title: str
        texts: DocList[TextDoc]


    MyDocCorrected = create_pure_python_type_model(CustomDoc)
    new_my_doc_cls = create_base_doc_from_schema(CustomDocCopy.schema(), 'MyDoc')
    ```

    ---
    :param schema: The schema of the original `BaseDoc` where DocLists are passed as regular Lists of Documents.
    :param base_doc_name: The name of the new pydantic model created.
    :param cached_models: Parameter used when this method is called recursively to reuse partial nested classes.
    :param definitions: Parameter used when this method is called recursively to reuse root definitions of other schemas.
    :return: A BaseDoc class dynamically created following the `schema`.
    """
    if not definitions:
        definitions = (
            schema.get('definitions', {}) if not is_pydantic_v2 else schema.get('$defs')
        )

    cached_models = cached_models if cached_models is not None else {}
    fields: Dict[str, Any] = {}
    if base_doc_name in cached_models:
        return cached_models[base_doc_name]
    has_id = False
    for field_name, field_schema in schema.get('properties', {}).items():
        if field_name == 'id':
            has_id = True
        field_type = _get_field_annotation_from_schema(
            field_schema=field_schema,
            field_name=field_name,
            root_schema=schema,
            cached_models=cached_models,
            is_tensor=False,
            num_recursions=0,
            definitions=definitions,
        )
        if not is_pydantic_v2:
            field_schema['default'] = field_schema.get('default', None)
            fields[field_name] = (
                field_type,
                FieldInfo(**field_schema),
            )
        else:
            field_kwargs = {}
            field_json_schema_extra = {}
            for k, v in field_schema.items():
                if k in FieldInfo.__slots__:
                    field_kwargs[k] = v
                else:
                    field_json_schema_extra[k] = v
            fields[field_name] = (
                field_type,
                FieldInfo(
                    json_schema_extra=field_json_schema_extra,
                    **field_kwargs,
                ),
            )

    base_model = BaseDoc if has_id else BaseDocWithoutId
    model = create_model(base_doc_name, __base__=base_model, **fields)
    if not is_pydantic_v2:
        model.__config__.title = schema.get('title', model.__config__.title)
    else:
        set_title = schema.get('title', model.model_config.get('title', None))
        if set_title:
            model.model_config['title'] = set_title

    for k in RESERVED_KEYS:
        if k in schema:
            schema.pop(k)
    if not is_pydantic_v2:
        model.__config__.schema_extra = schema
    else:
        model.model_config['json_schema_extra'] = schema
    cached_models[base_doc_name] = model
    return model
