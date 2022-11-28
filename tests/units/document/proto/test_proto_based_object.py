import numpy as np

from docarray.proto import DocumentProto, NdArrayProto, NodeProto
from docarray.typing import NdArray


def test_nested_item_proto():
    NodeProto(text='hello')
    NodeProto(nested=DocumentProto())


def test_nested_optional_item_proto():
    NodeProto()


def test_ndarray():
    nd_proto = NdArrayProto()
    original_ndarray = np.zeros((3, 224, 224))
    NdArray._flush_tensor_to_proto(nd_proto, value=original_ndarray)
    nested_item = NodeProto(ndarray=nd_proto)
    tensor = NdArray.from_protobuf(nested_item.ndarray)

    assert (tensor == original_ndarray).all()


def test_document_proto_set():

    data = {}

    nested_item1 = NodeProto(text='hello')

    nd_proto = NdArrayProto()
    original_ndarray = np.zeros((3, 224, 224))
    NdArray._flush_tensor_to_proto(nd_proto, value=original_ndarray)

    nested_item2 = NodeProto(ndarray=nd_proto)

    data['a'] = nested_item1
    data['b'] = nested_item2

    DocumentProto(data=data)
