import numpy as np

from docarray.proto import DocumentProto, NdArrayProto, NodeProto
from docarray.typing import Tensor


def test_nested_item_proto():
    NodeProto(text='hello')
    NodeProto(nested=DocumentProto())


def test_nested_optional_item_proto():
    NodeProto()


def test_ndarray():
    nd_proto = NdArrayProto()
    original_tensor = np.zeros((3, 224, 224))
    Tensor._flush_tensor_to_proto(nd_proto, value=original_tensor)
    nested_item = NodeProto(tensor=nd_proto)
    tensor = Tensor.from_protobuf(nested_item.tensor)

    assert (tensor == original_tensor).all()


def test_document_proto_set():

    data = {}

    nested_item1 = NodeProto(text='hello')

    nd_proto = NdArrayProto()
    original_tensor = np.zeros((3, 224, 224))
    Tensor._flush_tensor_to_proto(nd_proto, value=original_tensor)

    nested_item2 = NodeProto(tensor=nd_proto)

    data['a'] = nested_item1
    data['b'] = nested_item2

    DocumentProto(data=data)
