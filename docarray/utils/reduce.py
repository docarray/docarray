__all__ = ['reduce', 'reduce_all']

from typing import Dict, List, Optional

from docarray import DocArray


def reduce(
    left: DocArray, right: DocArray, left_id_map: Optional[Dict] = None
) -> 'DocArray':
    """
    Reduces left and right DocArray into one DocArray in-place.
    Changes are applied to the left DocArray.
    Reducing 2 DocArrays consists in adding Documents in the second DocArray
    to the first DocArray if they do not exist.
    If a Document exists in both DocArrays (identified by ID),
    the data properties are merged with priority to the left Document.

    Nested DocArrays are also reduced in the same way.
    :param left: First DocArray to be reduced. Changes will be applied to it
    in-place
    :param right: Second DocArray to be reduced
    :param left_id_map: Optional parameter to be passed in repeated calls
    for optimizations, keeping a map of the Document ID to its offset
    in the DocArray
    :return: Reduced DocArray
    """
    left_id_map = left_id_map or {doc.id: i for i, doc in enumerate(left)}

    for doc in right:
        if doc.id in left_id_map:
            left[left_id_map[doc.id]].update(doc)
        else:
            left.append(doc)

    return left


def reduce_all(docarrays: List[DocArray]) -> DocArray:
    """
    Reduces a list of DocArrays into one DocArray.
    Changes are applied to the first DocArray in-place.

    The resulting DocArray contains Documents of all DocArrays.
    If a Document exists (identified by their ID) in many DocArrays,
    data properties are merged with priority to the left-most
    DocArrays (that is, if a data attribute is set in a Document
    belonging to many DocArrays, the attribute value of the left-most
     DocArray is kept).
    Nested DocArrays belonging to many DocArrays
     are also reduced in the same way.
    .. note::
        - Nested DocArrays order does not follow any specific rule.
        You might want to re-sort them in a later step.
        - The final result depends on the order of DocArrays
        when applying reduction.

    :param docarrays: List of DocArrays to be reduced
    :return: the resulting DocArray
    """
    if len(docarrays) <= 1:
        raise Exception(
            'In order to reduce DocArrays' ' we should have more than one DocArray'
        )
    left = docarrays[0]
    others = docarrays[1:]
    left_id_map = {doc.id: i for i, doc in enumerate(left)}
    for da in others:
        reduce(left, da, left_id_map)
    return left
