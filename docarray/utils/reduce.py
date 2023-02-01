from docarray import DocumentArray
from typing import List, Optional, Dict, TYPE_CHECKING, Tuple, _GenericAlias
from typing_inspect import is_union_type


if TYPE_CHECKING:  # pragma: no cover
    from docarray.base_document import BaseDocument


def _non_empty_fields(doc: 'BaseDocument') -> Tuple[str]:
    r: List[str] = []
    for field_name in doc.__fields__.keys():
        v = getattr(doc, field_name)
        if v:
            r.append(field_name)
    return tuple(r)


def _array_fields(doc: 'BaseDocument') -> Tuple[str]:
    ret: List[str] = []
    for field_name, field in doc.__fields__.items():
        field_type = field.outer_type_
        print(f'HEY {field_type} => {type(field_type)}')
        print(f' {isinstance(field_type, _GenericAlias)}')
        if isinstance(field_type, _GenericAlias):
            print(field_type.__origin__)
        if isinstance(field_type, DocumentArray) or (isinstance(field_type, _GenericAlias) and field_type.__origin__ is list):
            ret.append(field_name)
        else:
            print(f' hhey 2')
    return tuple(ret)


"""
A mixin that provides reducing logic for :class:`DocumentArray`
Reducing 2 or more DocumentArrays consists in merging all Documents into the same DocumentArray.
If a Document belongs to 2 or more DocumentArrays, it is added once and data attributes are merged with priority to
the Document belonging to the left-most DocumentArray. Matches and chunks are also reduced in the same way.
Reduction is applied to all levels of DocumentArrays, that is, from root Documents to all their chunk and match
children.
"""


def reduce_docs(doc1: 'BaseDocument', doc2: 'BaseDocument', array_fields: Optional[List[str]] = None):
    """
    Reduces doc1 and doc2 into one Document in-place. Changes are applied to doc1.
    Reducing 2 Documents consists in setting data properties of the second Document to the first Document if they
    are empty (that is, priority to the left-most Document) and reducing the matches and the chunks of both
    documents.
    Non-data properties are ignored.
    Reduction of matches and chunks relies on :class:`DocumentArray`.:method:`reduce`.
    :param doc1: first Document
    :param doc2: second Document
    :param array_fields:
    """
    doc1_fields = set(_non_empty_fields(doc1))
    doc2_fields = set(_non_empty_fields(doc2))

    # update only fields that are set in doc2 and not set in doc1
    fields = doc2_fields - doc1_fields

    for field in fields:
        setattr(doc1, field, getattr(doc2, field))

    array_fields = array_fields or _array_fields(doc1)
    for field in array_fields:
        array1 = getattr(doc1, field)
        array2 = getattr(doc2, field)
        if array1 is None and array2 is not None:
            setattr(doc1, field, array2)
        elif array1 is not None and array2 is not None:
            array1.extend(array2)
            setattr(doc1, field, array1)  # I am not sure if this is optimal, how can I do (doc1.field.extend())

    return doc1


def reduce(left: DocumentArray, other: DocumentArray, left_id_map: Optional[Dict] = None,
           array_fields: Optional[List[str]] = None) -> 'DocumentArray':
    """
    Reduces other and the current DocumentArray into one DocumentArray in-place. Changes are applied to the current
    DocumentArray.
    Reducing 2 DocumentArrays consists in adding Documents in the second DocumentArray to the first DocumentArray
    if they do not exist. If a Document exists in both DocumentArrays, the data properties are merged with priority
    to the first Document (that is, to the current DocumentArray's Document). The matches and chunks are also
    reduced in the same way.
    :param left: DocumentArray
    :param other: DocumentArray
    :param left_id_map:
        :param array_fields:

    :return: DocumentArray
    """
    left_id_map = left_id_map or {doc.id: i for i, doc in enumerate(left)}
    array_fields = array_fields or left[0].array_fields

    for doc in other:
        if doc.id in left_id_map:
            reduce_docs(left[left_id_map[doc.id]], doc, array_fields)
        else:
            left.append(doc)

    return left


def reduce_all(left: DocumentArray, others: List[DocumentArray]) -> DocumentArray:
    """
    Reduces a list of DocumentArrays and this DocumentArray into one DocumentArray. Changes are applied to this
    DocumentArray in-place.

    Reduction consists in reducing this DocumentArray with every DocumentArray in `others` sequentially using
    :class:`DocumentArray`.:method:`reduce`.
    The resulting DocumentArray contains Documents of all DocumentArrays.
    If a Document exists in many DocumentArrays, data properties are merged with priority to the left-most
    DocumentArrays (that is, if a data attribute is set in a Document belonging to many DocumentArrays, the
    attribute value of the left-most DocumentArray is kept).
    Matches and chunks of a Document belonging to many DocumentArrays are also reduced in the same way.
    Other non-data properties are ignored.

    .. note::
        - Matches are not kept in a sorted order when they are reduced. You might want to re-sort them in a later
            step.
        - The final result depends on the order of DocumentArrays when applying reduction.

    :param left:
    :param others: List of DocumentArrays to be reduced
    :return: the resulting DocumentArray
    """
    assert len(left) > 0, 'In order to reduce DocumentArrays we should have a non empty DocumentArray'
    left_id_map = {doc.id: i for i, doc in enumerate(left)}
    array_fields = left[0].array_fields
    for da in others:
        reduce(left, da, left_id_map, array_fields)
    return left
