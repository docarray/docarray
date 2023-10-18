import glob
import itertools
import os
import re
from types import LambdaType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    Union,
)

import numpy as np

from docarray.utils._internal._typing import safe_issubclass
from docarray.utils._internal.misc import (
    is_jax_available,
    is_tf_available,
    is_torch_available,
)

if is_torch_available():
    import torch

if is_jax_available():
    import jax

if is_tf_available():
    import tensorflow as tf

if TYPE_CHECKING:
    from docarray import BaseDoc


def _is_access_path_valid(doc_type: Type['BaseDoc'], access_path: str) -> bool:
    """
    Check if a given access path ("__"-separated) is a valid path for a given Document class.
    """

    field_type = _get_field_annotation_by_access_path(doc_type, access_path)
    return field_type is not None


def _all_access_paths_valid(
    doc_type: Type['BaseDoc'], access_paths: List[str]
) -> List[bool]:
    """
    Check if all access paths ("__"-separated) are valid for a given Document class.
    """
    return [_is_access_path_valid(doc_type, path) for path in access_paths]


def _access_path_to_dict(access_path: str, value) -> Dict[str, Any]:
    """
    Convert an access path ("__"-separated) and its value to a (potentially) nested dict.

    ```python
    assert access_path_to_dict('image__url', 'img.png') == {'image': {'url': 'img.png'}}
    ```
    """
    fields = access_path.split('__')
    for field in reversed(fields):
        result = {field: value}
        value = result
    return result


def _is_none_like(val: Any) -> bool:
    """
    :param val: any value
    :return: true iff `val` equals to `None`, `'None'` or `''`
    """
    # Convoluted implementation, but fixes https://github.com/docarray/docarray/issues/1821

    # tensor-like types can have unexpected (= broadcast) `==`/`in` semantics,
    # so treat separately
    is_np_arr = isinstance(val, np.ndarray)
    if is_np_arr:
        return False

    is_torch_tens = is_torch_available() and isinstance(val, torch.Tensor)
    if is_torch_tens:
        return False

    is_tf_tens = is_tf_available() and isinstance(val, tf.Tensor)
    if is_tf_tens:
        return False

    is_jax_arr = is_jax_available() and isinstance(val, jax.numpy.ndarray)
    if is_jax_arr:
        return False

    # "normal" case
    return val in ['', 'None', None]


def _access_path_dict_to_nested_dict(access_path2val: Dict[str, Any]) -> Dict[Any, Any]:
    """
    Convert a dict, where the keys are access paths ("__"-separated) to a nested dictionary.

    ---

    ```python
    access_path2val = {'image__url': 'some.png'}
    assert access_path_dict_to_nested_dict(access_path2val) == {
        'image': {'url': 'some.png'}
    }
    ```

    ---

    :param access_path2val: dict with access_paths as keys
    :return: nested dict where the access path keys are split into separate field names and nested keys
    """
    nested_dict: Dict[Any, Any] = {}
    for access_path, value in access_path2val.items():
        field2val = _access_path_to_dict(
            access_path=access_path,
            value=None if _is_none_like(value) else value,
        )
        _update_nested_dicts(to_update=nested_dict, update_with=field2val)
    return nested_dict


def _dict_to_access_paths(d: dict) -> Dict[str, Any]:
    """
    Convert a (nested) dict to a Dict[access_path, value].
    Access paths are defined as a path of field(s) separated by "__".

    ```python
    assert dict_to_access_paths({'image': {'url': 'img.png'}}) == {'image__url', 'img.png'}
    ```

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

    ```python
    d1 = {'image': {'tensor': None}, 'title': 'hello'}
    d2 = {'image': {'url': 'some.png'}}

    update_nested_dicts(d1, d2)
    assert d1 == {'image': {'tensor': None, 'url': 'some.png'}, 'title': 'hello'}
    ```

    :param to_update: dict that should be updated
    :param update_with: dict to update with
    :return: merged dict
    """
    for k, v in update_with.items():
        if k not in to_update.keys():
            to_update[k] = v
        else:
            _update_nested_dicts(to_update[k], update_with[k])


def _get_field_annotation_by_access_path(
    doc_type: Type['BaseDoc'], access_path: str
) -> Optional[Type]:
    """
    Get field type by "__"-separated access path.

    :param doc_type: type of document
    :param access_path: "__"-separated access path
    :return: field type of accessed attribute. If access path is invalid, return None.
    """
    from docarray import BaseDoc, DocList

    field, _, remaining = access_path.partition('__')
    field_valid = field in doc_type._docarray_fields().keys()

    if field_valid:
        if len(remaining) == 0:
            return doc_type._get_field_annotation(field)
        else:
            d = doc_type._get_field_annotation(field)
            if safe_issubclass(d, DocList):
                return _get_field_annotation_by_access_path(d.doc_type, remaining)
            elif safe_issubclass(d, BaseDoc):
                return _get_field_annotation_by_access_path(d, remaining)
            else:
                return None
    else:
        return None


def _is_lambda_or_partial_or_local_function(func: Callable[[Any], Any]) -> bool:
    """
    Return True if `func` is lambda, local or partial function, else False.
    """
    return (
        (isinstance(func, LambdaType) and func.__name__ == '<lambda>')
        or not hasattr(func, '__qualname__')
        or ('<locals>' in func.__qualname__)
    )


def get_paths(
    patterns: Union[str, List[str]],
    recursive: bool = True,
    size: Optional[int] = None,
    exclude_regex: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Yield file paths described by `patterns`.

    ---

    ```python
    from typing import Optional
    from docarray import BaseDoc, DocList
    from docarray.helper import get_paths
    from docarray.typing import TextUrl, ImageUrl


    class Banner(BaseDoc):
        text_url: TextUrl
        image_url: Optional[ImageUrl]


    # you can call it in the constructor
    docs = DocList[Banner]([Banner(text_url=url) for url in get_paths(patterns='*.txt')])

    # and call it after construction to set the urls
    docs.image_url = list(get_paths(patterns='*.jpg', exclude_regex='test'))

    for doc in docs:
        assert doc.image_url.endswith('.txt')
        assert doc.text_url.endswith('.jpg')
    ```

    ---

    :param patterns: The pattern may contain simple shell-style wildcards,
        e.g. '\*.py', '[\*.zip, \*.gz]'
    :param recursive: If recursive is true, the pattern '**' will match any
        files and zero or more directories and subdirectories
    :param size: the maximum number of the files
    :param exclude_regex: if set, then filenames that match to this pattern
        are not included.
    :yield: file paths

    """

    if isinstance(patterns, str):
        patterns = [patterns]

    regex_to_exclude = None
    if exclude_regex:
        try:
            regex_to_exclude = re.compile(exclude_regex)
        except re.error:
            raise ValueError(f'`{exclude_regex}` is not a valid regex.')

    def _iter_file_extensions(ps):
        return itertools.chain.from_iterable(
            glob.iglob(os.path.expanduser(p), recursive=recursive) for p in ps
        )

    num_docs = 0
    for file_path in _iter_file_extensions(patterns):
        if regex_to_exclude and regex_to_exclude.match(file_path):
            continue

        yield file_path

        num_docs += 1
        if size is not None and num_docs >= size:
            break


def _shallow_copy_doc(doc):
    return doc.__class__._shallow_copy(doc)
