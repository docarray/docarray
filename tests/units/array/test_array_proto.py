import numpy as np

from docarray import BaseDocument, DocumentArray, Image, Text
from docarray.array.array_stacked import DocumentArrayStacked
from docarray.typing import NdArray


def test_simple_proto():
    class CustomDoc(BaseDocument):
        text: str
        tensor: NdArray

    da = DocumentArray(
        [CustomDoc(text='hello', tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    new_da = DocumentArray[CustomDoc].from_protobuf(da.to_protobuf())

    for doc1, doc2 in zip(da, new_da):
        assert doc1.text == doc2.text
        assert (doc1.tensor == doc2.tensor).all()


def test_nested_proto():
    class CustomDocument(BaseDocument):
        text: Text
        image: Image

    da = DocumentArray[CustomDocument](
        [
            CustomDocument(
                text=Text(text='hello'), image=Image(tensor=np.zeros((3, 224, 224)))
            )
            for _ in range(10)
        ]
    )

    DocumentArray[CustomDocument].from_protobuf(da.to_protobuf())


def test_nested_proto_any_doc():
    class CustomDocument(BaseDocument):
        text: Text
        image: Image

    da = DocumentArray[CustomDocument](
        [
            CustomDocument(
                text=Text(text='hello'), image=Image(tensor=np.zeros((3, 224, 224)))
            )
            for _ in range(10)
        ]
    )

    DocumentArray.from_protobuf(da.to_protobuf())


def test_stacked_proto():
    class CustomDocument(BaseDocument):
        image: NdArray

    da = DocumentArray[CustomDocument](
        [CustomDocument(image=np.zeros((3, 224, 224))) for _ in range(10)]
    ).stack()

    da2 = DocumentArrayStacked.from_protobuf(da.to_protobuf())

    assert isinstance(da2, DocumentArrayStacked)
