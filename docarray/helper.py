from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from docarray import BaseDocument


def is_access_path_valid(doc: Type['BaseDocument'], access_path: str) -> bool:
    """
    Check if a given access path ("__"-separated) is a valid path for a given Document class.
    """
    from docarray import BaseDocument

    field, _, remaining = access_path.partition('__')
    if len(remaining) == 0:
        return access_path in doc.__fields__.keys()
    else:
        valid_field = field in doc.__fields__.keys()
        if not valid_field:
            return False
        else:
            d = doc._get_field_type(field)
            if not issubclass(d, BaseDocument):
                return False
            else:
                return is_access_path_valid(d, remaining)


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
