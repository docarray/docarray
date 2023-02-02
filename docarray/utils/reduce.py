from docarray import DocumentArray
from typing import List, Optional, Dict, Tuple, _GenericAlias
from docarray.base_document import BaseDocument


def _types_analysis(doc: 'BaseDocument') -> Tuple[List[str]]:
    simple_non_empty_fields: List[str] = []
    list_fields: List[str] = []
    set_fields: List[str] = []
    nested_docs_fields: List[str] = []
    nested_docarray_fields: List[str] = []

    for field_name, field in doc.__fields__.items():
        field_type = field.outer_type_
        if not isinstance(field_type, _GenericAlias) and issubclass(field_type, DocumentArray):
            nested_docarray_fields.append(field_name)
        elif isinstance(field_type, _GenericAlias) and field_type.__origin__ is list:
            list_fields.append(field_name)
        elif isinstance(field_type, _GenericAlias) and field_type.__origin__ is set:
            set_fields.append(field_name)
        v = getattr(doc, field_name)
        if v:
            if isinstance(v, BaseDocument):
                nested_docs_fields.append(field_name)
            else:
                simple_non_empty_fields.append(field_name)
    return tuple([simple_non_empty_fields, list_fields, set_fields, nested_docarray_fields, nested_docs_fields])


"""
A mixin that provides reducing logic for :class:`DocumentArray`
Reducing 2 or more DocumentArrays consists in merging all Documents into the same DocumentArray.
If a Document belongs to 2 or more DocumentArrays, it is added once and data attributes are merged with priority to
the Document belonging to the left-most DocumentArray. Matches and chunks are also reduced in the same way.
Reduction is applied to all levels of DocumentArrays, that is, from root Documents to all their chunk and match
children.
"""


def reduce_docs(doc1: 'BaseDocument', doc2: 'BaseDocument') -> 'BaseDocument':
    """
    Reduces doc1 and doc2 into one Document in-place. Changes are applied to doc1.
    Reducing 2 Documents consists in setting data properties of the second Document to the first Document if they
    are empty (that is, priority to the left-most Document) and reducing the matches and the chunks of both
    documents.
    Non-data properties are ignored.
    Reduction of matches and chunks relies on :class:`DocumentArray`.:method:`reduce`.
    :param doc1: first Document
    :param doc2: second Document
    """
    doc1_simple_non_empty_fields, doc1_list_fields, doc1_set_fields, doc1_nested_docarray_fields, doc1_nested_docs_fields = _types_analysis(
        doc1)
    doc2_simple_non_empty_fields, doc2_list_fields, doc2_set_fields, doc2_nested_docarray_fields, doc2_nested_docs_fields = _types_analysis(
        doc2)

    # update only fields that are set in doc2 and not set in doc1
    update_simple_fields = set(doc2_simple_non_empty_fields) - set(doc1_simple_non_empty_fields)

    for field in update_simple_fields:
        setattr(doc1, field, getattr(doc2, field))

    for field in set(doc1_nested_docs_fields + doc2_nested_docs_fields):
        setattr(doc1, field, reduce_docs(getattr(doc1, field), getattr(doc2, field)))

    for field in doc1_list_fields:
        array1 = getattr(doc1, field)
        array2 = getattr(doc2, field)
        if array1 is None and array2 is not None:
            setattr(doc1, field, array2)
        elif array1 is not None and array2 is not None:
            array1.extend(array2)
            setattr(doc1, field, array1)

    for field in doc1_set_fields:
        array1 = getattr(doc1, field)
        array2 = getattr(doc2, field)
        if array1 is None and array2 is not None:
            setattr(doc1, field, array2)
        elif array1 is not None and array2 is not None:
            array1.update(array2)
            setattr(doc1, field, array1)

    for field in doc1_nested_docarray_fields:
        array1 = getattr(doc1, field)
        array2 = getattr(doc2, field)
        if array1 is None and array2 is not None:
            setattr(doc1, field, array2)
        elif array1 is not None and array2 is not None:
            array1 = reduce(array1, array2)
            setattr(doc1, field, array1)

    return doc1


def reduce(left: DocumentArray, other: DocumentArray, left_id_map: Optional[Dict] = None) -> 'DocumentArray':
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
    :return: DocumentArray
    """
    left_id_map = left_id_map or {doc.id: i for i, doc in enumerate(left)}

    for doc in other:
        if doc.id in left_id_map:
            reduce_docs(left[left_id_map[doc.id]], doc)
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
    for da in others:
        reduce(left, da, left_id_map)
    return left
