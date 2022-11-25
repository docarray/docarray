from docarray.array.find import find
import numpy as np

from docarray.typing import NdArray
from docarray import Document, DocumentArray


def test_find_method():
    query_embedding = np.zeros((100, 1))

    class CustomDoc(Document):
        text: str
        embedding: NdArray

    da = DocumentArray(
        [CustomDoc(text='hello', embedding=np.zeros((100, 1))) for _ in range(20)]
    )
    new_da = DocumentArray[CustomDoc].from_protobuf(da.to_protobuf())
    matched_da = find(query_embedding=query_embedding, index=new_da)
    len_of_da = len(matched_da)

    assert len_of_da == 20
