from typing import Optional

import numpy as np
import torch

from docarray import DocumentArray
from docarray.document_base import BaseDocument
from docarray.typing import NdArray, TorchTensor


def test_proto_simple():
    class CustomDoc(BaseDocument):
        text: str

    doc = CustomDoc(text='hello')

    CustomDoc.from_protobuf(doc.to_protobuf())


def test_proto_ndarray():
    class CustomDoc(BaseDocument):
        tensor: NdArray

    tensor = np.zeros((3, 224, 224))
    doc = CustomDoc(tensor=tensor)

    new_doc = CustomDoc.from_protobuf(doc.to_protobuf())

    assert (new_doc.tensor == tensor).all()


def test_proto_with_nested_doc():
    class CustomInnerDoc(BaseDocument):
        tensor: NdArray

    class CustomDoc(BaseDocument):
        text: str
        inner: CustomInnerDoc

    doc = CustomDoc(text='hello', inner=CustomInnerDoc(tensor=np.zeros((3, 224, 224))))

    CustomDoc.from_protobuf(doc.to_protobuf())


def test_proto_with_chunks_doc():
    class CustomInnerDoc(BaseDocument):
        tensor: NdArray

    class CustomDoc(BaseDocument):
        text: str
        chunks: DocumentArray[CustomInnerDoc]

    doc = CustomDoc(
        text='hello',
        chunks=DocumentArray[CustomInnerDoc](
            [CustomInnerDoc(tensor=np.zeros((3, 224, 224))) for _ in range(5)],
        ),
    )

    new_doc = CustomDoc.from_protobuf(doc.to_protobuf())

    for chunk1, chunk2 in zip(doc.chunks, new_doc.chunks):

        assert (chunk1.tensor == chunk2.tensor).all()


def test_proto_with_nested_doc_pytorch():
    class CustomInnerDoc(BaseDocument):
        tensor: TorchTensor

    class CustomDoc(BaseDocument):
        text: str
        inner: CustomInnerDoc

    doc = CustomDoc(
        text='hello', inner=CustomInnerDoc(tensor=torch.zeros((3, 224, 224)))
    )

    CustomDoc.from_protobuf(doc.to_protobuf())


def test_proto_with_chunks_doc_pytorch():
    class CustomInnerDoc(BaseDocument):
        tensor: TorchTensor

    class CustomDoc(BaseDocument):
        text: str
        chunks: DocumentArray[CustomInnerDoc]

    doc = CustomDoc(
        text='hello',
        chunks=DocumentArray[CustomInnerDoc](
            [CustomInnerDoc(tensor=torch.zeros((3, 224, 224))) for _ in range(5)],
        ),
    )

    new_doc = CustomDoc.from_protobuf(doc.to_protobuf())

    for chunk1, chunk2 in zip(doc.chunks, new_doc.chunks):

        assert (chunk1.tensor == chunk2.tensor).all()


def test_optional_field_in_doc():
    class CustomDoc(BaseDocument):
        text: Optional[str]

    CustomDoc.from_protobuf(CustomDoc().to_protobuf())


def test_optional_field_nested_in_doc():
    class InnerDoc(BaseDocument):
        title: str

    class CustomDoc(BaseDocument):
        text: Optional[InnerDoc]

    CustomDoc.from_protobuf(CustomDoc().to_protobuf())
