import numpy as np
import pytest

from docarray.proto import DocProto, NodeProto
from docarray.typing import NdArray


@pytest.mark.proto
def test_ndarray():
    original_ndarray = np.zeros((3, 224, 224))

    custom_ndarray = NdArray._docarray_from_native(original_ndarray)

    tensor = NdArray.from_protobuf(custom_ndarray.to_protobuf())

    assert (tensor == original_ndarray).all()


@pytest.mark.proto
def test_document_proto_set():
    data = {}

    nested_item1 = NodeProto(text='hello')

    ndarray = NdArray._docarray_from_native(np.zeros((3, 224, 224)))
    nd_proto = ndarray.to_protobuf()

    nested_item2 = NodeProto(ndarray=nd_proto)

    data['a'] = nested_item1
    data['b'] = nested_item2

    DocProto(data=data)
