from typing import Optional

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from docarray import BaseDocument, DocumentArray
from docarray.documents import AudioDoc, ImageDoc, TextDoc
from docarray.documents.helper import (
    create_doc,
    create_doc_from_typeddict,
    create_doc_from_dict,
)
from docarray.typing import AudioNdArray


def test_multi_modal_doc():
    class MyMultiModalDoc(BaseDocument):
        image: ImageDoc
        text: TextDoc

    doc = MyMultiModalDoc(
        image=ImageDoc(tensor=np.zeros((3, 224, 224))), text=TextDoc(text='hello')
    )

    assert isinstance(doc.image, BaseDocument)
    assert isinstance(doc.image, ImageDoc)
    assert isinstance(doc.text, TextDoc)

    assert doc.text.text == 'hello'
    assert (doc.image.tensor == np.zeros((3, 224, 224))).all()


def test_nested_chunks_document():
    class ChunksDocument(BaseDocument):
        text: str
        images: DocumentArray[ImageDoc]

    doc = ChunksDocument(
        text='hello',
        images=DocumentArray[ImageDoc]([ImageDoc() for _ in range(10)]),
    )

    assert isinstance(doc.images, DocumentArray)


def test_create_doc():
    with pytest.raises(ValueError):
        _ = create_doc(
            'MyMultiModalDoc',
            __base__=BaseModel,
            image=(ImageDoc, ...),
            text=(TextDoc, ...),
        )

    MyMultiModalDoc = create_doc(
        'MyMultiModalDoc', image=(ImageDoc, ...), text=(TextDoc, ...)
    )

    assert issubclass(MyMultiModalDoc, BaseDocument)

    doc = MyMultiModalDoc(
        image=ImageDoc(tensor=np.zeros((3, 224, 224))), text=TextDoc(text='hello')
    )

    assert isinstance(doc.image, BaseDocument)
    assert isinstance(doc.image, ImageDoc)
    assert isinstance(doc.text, TextDoc)

    assert doc.text.text == 'hello'
    assert (doc.image.tensor == np.zeros((3, 224, 224))).all()

    MyAudio = create_doc(
        'MyAudio',
        __base__=AudioDoc,
        title=(str, ...),
        tensor=(Optional[AudioNdArray], ...),
    )

    assert issubclass(MyAudio, BaseDocument)
    assert issubclass(MyAudio, AudioDoc)


def test_create_doc_from_typeddict():
    class MyMultiModalDoc(TypedDict):
        image: ImageDoc
        text: TextDoc

    with pytest.raises(ValueError):
        _ = create_doc_from_typeddict(MyMultiModalDoc, __base__=BaseModel)

    Doc = create_doc_from_typeddict(MyMultiModalDoc)

    assert issubclass(Doc, BaseDocument)

    class MyAudio(TypedDict):
        title: str
        tensor: Optional[AudioNdArray]

    Doc = create_doc_from_typeddict(MyAudio, __base__=AudioDoc)

    assert issubclass(Doc, BaseDocument)
    assert issubclass(Doc, AudioDoc)


def test_create_doc_from_dict():
    data_dict = {
        'image': ImageDoc(tensor=np.random.rand(3, 224, 224)),
        'text': TextDoc(text='hello'),
        'id': 123,
    }

    MyDoc = create_doc_from_dict(model_name='MyDoc', data_dict=data_dict)

    assert issubclass(MyDoc, BaseDocument)

    doc = MyDoc(
        image=ImageDoc(tensor=np.random.rand(3, 224, 224)),
        text=TextDoc(text='hey'),
        id=111,
    )

    assert isinstance(doc, BaseDocument)
    assert isinstance(doc.text, TextDoc)
    assert isinstance(doc.image, ImageDoc)
    assert isinstance(doc.id, int)

    # Create a doc with an incorrect type
    with pytest.raises(ValidationError):
        doc = MyDoc(
            image=ImageDoc(tensor=np.random.rand(3, 224, 224)),
            text=['some', 'text'],  # should be TextDoc
            id=111,
        )

    # Handle empty data_dict
    with pytest.raises(ValueError):
        MyDoc = create_doc_from_dict(model_name='MyDoc', data_dict={})

    # Data with a None value
    data_dict = {'text': 'some text', 'other': None}
    MyDoc = create_doc_from_dict(model_name='MyDoc', data_dict=data_dict)

    assert issubclass(MyDoc, BaseDocument)

    doc1 = MyDoc(text='txt', other=10)
    doc2 = MyDoc(text='txt', other='also text')

    assert isinstance(doc1, BaseDocument) and isinstance(doc2, BaseDocument)
