import numpy as np
import pytest

from docarray import BaseDocument, DocumentArray
from docarray.typing import NdArray


@pytest.mark.proto
def test_proto_stacked_mode_torch(batch):
    batch.from_protobuf(batch.to_protobuf())


@pytest.mark.proto
def test_proto_stacked_mode_numpy():
    class MyDoc(BaseDocument):
        tensor: NdArray[3, 224, 224]

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    da = da.stack()

    da.from_protobuf(da.to_protobuf())
