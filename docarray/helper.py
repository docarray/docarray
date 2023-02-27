from typing import TYPE_CHECKING, Any, Dict, List, Type

if TYPE_CHECKING:
    from docarray import BaseDocument


def _is_access_path_valid(doc_type: Type['BaseDocument'], access_path: str) -> bool:
    """
    Check if a given access path ("__"-separated) is a valid path for a given Document class.
    """

    field_type = _get_field_type_by_access_path(doc_type, access_path)
    return field_type is not None


def _all_access_paths_valid(
    doc_type: Type['BaseDocument'], access_paths: List[str]
) -> List[bool]:
    """
    Check if all access paths ("__"-separated) are valid for a given Document class.
    """
    return [_is_access_path_valid(doc_type, path) for path in access_paths]


def _access_path_to_dict(access_path: str, value) -> Dict[str, Any]:
    """
    Convert an access path ("__"-separated) and its value to a (potentially) nested dict.

    EXAMPLE USAGE
    .. code-block:: python
        assert access_path_to_dict('image__url', 'img.png') == {'image': {'url': 'img.png'}}
    """
    fields = access_path.split('__')
    for field in reversed(fields):
        result = {field: value}
        value = result
    return result


def _access_path_dict_to_nested_dict(access_path2val: Dict[str, Any]) -> Dict[Any, Any]:
    """
    Convert a dict, where the keys are access paths ("__"-separated) to a nested dictionary.

    EXAMPLE USAGE

    .. code-block:: python

        access_path2val = {'image__url': 'some.png'}
        assert access_path_dict_to_nested_dict(access_path2val) == {
            'image': {'url': 'some.png'}
        }

    :param access_path2val: dict with access_paths as keys
    :return: nested dict where the access path keys are split into separate field names and nested keys
    """
    nested_dict: Dict[Any, Any] = {}
    for access_path, value in access_path2val.items():
        field2val = _access_path_to_dict(
            access_path=access_path,
            value=value if value not in ['', 'None'] else None,
        )
        _update_nested_dicts(to_update=nested_dict, update_with=field2val)
    return nested_dict


def _dict_to_access_paths(d: dict) -> Dict[str, Any]:
    """
    Convert a (nested) dict to a Dict[access_path, value].
    Access paths are defined as a path of field(s) separated by "__".

    EXAMPLE USAGE
    .. code-block:: python
        assert dict_to_access_paths({'image': {'url': 'img.png'}}) == {'image__url', 'img.png'}
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = _dict_to_access_paths(v)
            for nested_k, nested_v in v.items():
                new_key = '__'.join([k, nested_k])
                result[new_key] = nested_v
        else:
            result[k] = v
    return result


def _update_nested_dicts(
    to_update: Dict[Any, Any], update_with: Dict[Any, Any]
) -> None:
    """
    Update a dict with another one, while considering shared nested keys.

    EXAMPLE USAGE:

    .. code-block:: python

        d1 = {'image': {'tensor': None}, 'title': 'hello'}
        d2 = {'image': {'url': 'some.png'}}

        update_nested_dicts(d1, d2)
        assert d1 == {'image': {'tensor': None, 'url': 'some.png'}, 'title': 'hello'}

    :param to_update: dict that should be updated
    :param update_with: dict to update with
    :return: merged dict
    """
    for k, v in update_with.items():
        if k not in to_update.keys():
            to_update[k] = v
        else:
            _update_nested_dicts(to_update[k], update_with[k])


def _get_field_type_by_access_path(
    doc_type: Type['BaseDocument'], access_path: str
) -> Any:
    """
    Get field type by "__"-separated access path.
    :param doc_type: type of document
    :param access_path: "__"-separated access path
    :return: field type of accessed attribute. If access path is invalid, return None.
    """
    from docarray import BaseDocument, DocumentArray

    field, _, remaining = access_path.partition('__')
    field_valid = field in doc_type.__fields__.keys()

    if field_valid:
        if len(remaining) == 0:
            return doc_type._get_field_type(field)
        else:
            d = doc_type._get_field_type(field)
            if issubclass(d, DocumentArray):
                return _get_field_type_by_access_path(d.document_type, remaining)
            elif issubclass(d, BaseDocument):
                return _get_field_type_by_access_path(d, remaining)
    else:
        return None
