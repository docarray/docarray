import copy
from typing import Sequence, Tuple, List

import pandas as pd

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
            _docs.append(doc)
            ids.append(doc.id)
    return _docs, ids


class DocumentSeries(pd.Series):
    def __deepcopy__(self, memo):
        cp = self.copy(deep=True)
        for doc in cp:
            cp[doc.id] = copy.deepcopy(doc)
        return cp
