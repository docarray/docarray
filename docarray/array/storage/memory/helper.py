from typing import Sequence, Tuple, List

from docarray import Document


def _get_docs_ids(
    docs: Sequence['Document'], copy: bool = False
) -> Tuple[List['Document'], List[str]]:
    """ Returns a tuple of docs and ids while consuming the generator only once"""
    _docs, ids = [], []
    if copy:
        for doc in docs:
            _docs.append(Document(doc, copy=True))
            ids.append(doc.id)
    else:
        for doc in docs:
            _docs.append(Document(doc))
            ids.append(doc.id)
    return _docs, ids
