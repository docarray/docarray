import numpy as np

from docarray.proto import DocumentProto, NodeProto
from docarray.typing import NdArray


def test_nested_item_proto():
    NodeProto(text='hello')
    NodeProto(nested=DocumentProto())


def test_nested_optional_item_proto():
    NodeProto()


def test_ndarray():

    original_ndarray = np.zeros((3, 224, 224))

    custom_ndarray = NdArray._docarray_from_native(original_ndarray)

    tensor = NdArray.from_protobuf(custom_ndarray.to_protobuf())

    assert (tensor == original_ndarray).all()


def test_document_proto_set():

    data = {}

    nested_item1 = NodeProto(text='hello')

    ndarray = NdArray._docarray_from_native(np.zeros((3, 224, 224)))
    nd_proto = ndarray.to_protobuf()

    nested_item2 = NodeProto(ndarray=nd_proto)

    data['a'] = nested_item1
    data['b'] = nested_item2

    DocumentProto(data=data)
