__all__ = ['reduce', 'reduce_all']

from typing import Dict, List, Optional

from docarray import DocList


def reduce(
    left: DocList, right: DocList, left_id_map: Optional[Dict] = None
) -> 'DocList':
    """
    Reduces left and right DocList into one DocList in-place.
    Changes are applied to the left DocList.
    Reducing 2 DocLists consists in adding Documents in the second DocList
    to the first DocList if they do not exist.
    If a Document exists in both DocLists (identified by ID),
    the data properties are merged with priority to the left Document.

    Nested DocLists are also reduced in the same way.
    :param left: First DocList to be reduced. Changes will be applied to it
    in-place
    :param right: Second DocList to be reduced
    :param left_id_map: Optional parameter to be passed in repeated calls
    for optimizations, keeping a map of the Document ID to its offset
    in the DocList
    :return: Reduced DocList
    """
    left_id_map = left_id_map or {doc.id: i for i, doc in enumerate(left)}

    for doc in right:
        if doc.id in left_id_map:
            left[left_id_map[doc.id]].update(doc)
        else:
            casted = left.doc_type(**doc.__dict__)
            left.append(casted)

    return left


def reduce_all(docs: List[DocList]) -> DocList:
    """
    Reduces a list of DocLists into one DocList.
    Changes are applied to the first DocList in-place.

    The resulting DocList contains Documents of all DocLists.
    If a Document exists (identified by their ID) in many DocLists,
    data properties are merged with priority to the left-most
    DocLists (that is, if a data attribute is set in a Document
    belonging to many DocLists, the attribute value of the left-most
     DocList is kept).
    Nested DocLists belonging to many DocLists
     are also reduced in the same way.

    !!! note

        - Nested DocLists order does not follow any specific rule.
        You might want to re-sort them in a later step.
        - The final result depends on the order of DocLists
        when applying reduction.

    :param docs: List of DocLists to be reduced
    :return: the resulting DocList
    """
    if len(docs) <= 1:
        raise Exception(
            'In order to reduce DocLists' ' we should have more than one DocList'
        )
    left = docs[0]
    others = docs[1:]
    left_id_map = {doc.id: i for i, doc in enumerate(left)}
    for other_docs in others:
        reduce(left, other_docs, left_id_map)
    return left
