from typing import Optional

import numpy as np
import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from docarray import BaseDocument, DocumentArray
from docarray.documents import AudioDoc, ImageDoc, TextDoc
from docarray.documents.helper import create_doc, create_from_typeddict
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


def test_create_from_typeddict():
    class MyMultiModalDoc(TypedDict):
        image: ImageDoc
        text: TextDoc

    with pytest.raises(ValueError):
        _ = create_from_typeddict(MyMultiModalDoc, __base__=BaseModel)

    Doc = create_from_typeddict(MyMultiModalDoc)

    assert issubclass(Doc, BaseDocument)

    class MyAudio(TypedDict):
        title: str
        tensor: Optional[AudioNdArray]

    Doc = create_from_typeddict(MyAudio, __base__=AudioDoc)

    assert issubclass(Doc, BaseDocument)
    assert issubclass(Doc, AudioDoc)
