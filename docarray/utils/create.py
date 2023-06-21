from docarray import DocList, BaseDoc
from docarray.typing import AnyTensor
from pydantic import create_model
from typing import Dict, List, Any, Union, Optional, Type


def create_new_model_cast_doclist_to_list(model: Any) -> BaseDoc:
    fields: Dict[str, Any] = {}
    for field_name, field in model.__annotations__.items():
        try:
            if issubclass(field, DocList):
                t: Any = field.doc_type
                fields[field_name] = (List[t], {})
            else:
                fields[field_name] = (field, {})
        except TypeError:
            fields[field_name] = (field, {})
    return create_model(
        model.__name__, __base__=model, __validators__=model.__validators__, **fields
    )


def _get_field_from_type(
    field_schema,
    field_name,
    root_schema,
    cached_models,
    is_tensor=False,
    num_recursions=0,
):
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
                    )
                )
            else:
                any_of_types.append(
                    _get_field_from_type(
                        any_of_schema,
                        field_name,
                        root_schema=root_schema,
                        cached_models=cached_models,
                        is_tensor=tensor_shape is not None,
                        num_recursions=0,
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
                        root_schema['definitions'][ref_name],
                        ref_name,
                        cached_models=cached_models,
                    )
                else:
                    ret = Any
            else:  # object reference in definitions
                if obj_ref:
                    ref_name = obj_ref.split('/')[-1]
                    doc_type = create_base_doc_from_schema(
                        root_schema['definitions'][ref_name],
                        ref_name,
                        cached_models=cached_models,
                    )
                    ret = DocList[doc_type]
                else:
                    doc_type = create_base_doc_from_schema(
                        field_schema, field_name, cached_models=cached_models
                    )
                    ret = DocList[doc_type]
    elif field_type == 'array':
        ret = _get_field_from_type(
            field_schema=field_schema.get('items', {}),
            field_name=field_name,
            root_schema=root_schema,
            cached_models=cached_models,
            is_tensor=tensor_shape is not None,
            num_recursions=num_recursions + 1,
        )
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
    schema: Dict[str, Any], model_name: str, cached_models: Optional[Dict] = None
) -> Type:
    cached_models = cached_models if cached_models is not None else {}
    fields: Dict[str, Any] = {}
    if model_name in cached_models:
        return cached_models[model_name]
    for field_name, field_schema in schema.get('properties', {}).items():
        field_type = _get_field_from_type(
            field_schema=field_schema,
            field_name=field_name,
            root_schema=schema,
            cached_models=cached_models,
            is_tensor=False,
            num_recursions=0,
        )
        fields[field_name] = (field_type, field_schema.get('description'))

    model = create_model(model_name, __base__=BaseDoc, **fields)
    cached_models[model_name] = model
    return model
