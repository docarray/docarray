from docarray import DocumentArray
from typing import List, Optional, Dict


def reduce(
    left: DocumentArray, right: DocumentArray, left_id_map: Optional[Dict] = None
) -> 'DocumentArray':
    """
    Reduces left and right DocumentArray into one DocumentArray in-place.
    Changes are applied to the left DocumentArray.
    Reducing 2 DocumentArrays consists in adding Documents in the second DocumentArray
    to the first DocumentArray if they do not exist.
    If a Document exists in both DocumentArrays (identified by ID),
    the data properties are merged with priority to the left Document.

    Nested DocumentArrays are also reduced in the same way.
    :param left: First DocumentArray to be reduced. Changes will be applied to it
    in-place
    :param right: Second DocumentArray to be reduced
    :param left_id_map: Optional parameter to be passed in repeated calls
    for optimizations, keeping a map of the Document ID to its offset
    in the DocumentArray
    :return: Reduced DocumentArray
    """
    left_id_map = left_id_map or {doc.id: i for i, doc in enumerate(left)}

    for doc in right:
        if doc.id in left_id_map:
            left[left_id_map[doc.id]].update(doc)
        else:
            left.doc_append(doc)

    return left


def reduce_all(docarrays: List[DocumentArray]) -> DocumentArray:
    """
    Reduces a list of DocumentArrays into one DocumentArray.
    Changes are applied to the first DocumentArray in-place.

    The resulting DocumentArray contains Documents of all DocumentArrays.
    If a Document exists (identified by their ID) in many DocumentArrays,
    data properties are merged with priority to the left-most
    DocumentArrays (that is, if a data attribute is set in a Document
    belonging to many DocumentArrays, the attribute value of the left-most
     DocumentArray is kept).
    Nested DocumentArrays belonging to many DocumentArrays
     are also reduced in the same way.
    .. note::
        - Nested DocumentArrays order does not follow any specific rule.
        You might want to re-sort them in a later step.
        - The final result depends on the order of DocumentArrays
        when applying reduction.

    :param docarrays: List of DocumentArrays to be reduced
    :return: the resulting DocumentArray
    """
    if len(docarrays) <= 1:
        raise Exception(
            'In order to reduce DocumentArrays'
            ' we should have more than one DocumentArray'
        )
    left = docarrays[0]
    others = docarrays[1:]
    left_id_map = {doc.id: i for i, doc in enumerate(left)}
    for da in others:
        reduce(left, da, left_id_map)
    return left
