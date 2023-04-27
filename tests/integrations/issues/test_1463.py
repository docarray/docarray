from docarray import DocList, BaseDoc
from docarray.typing import NdArrayEmbedding
from docarray.utils.find import find

import numpy as np


def test_fix_max_recursion():
    class Doc(BaseDoc):
        text: str
        embedding: NdArrayEmbedding

    query = np.random.rand(128)
    index = DocList[Doc](
        [Doc(text=_, embedding=np.random.rand(128)) for _ in range(100)]
    )
    top_matches, _ = find(
        index=index, query=query, search_field='embedding', metric='cosine_sim'
    )
    df = top_matches.to_dataframe()
    assert 'id' in df.columns
    assert 'embedding' in df.columns
    assert 'text' in df.columns
