from typing import Optional

import numpy as np
import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from docarray import BaseDocument, DocumentArray
from docarray.documents import ImageDoc, TextDoc


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
            'MyMultiModalDoc', __base__=BaseModel, image=(Image, ...), text=(Text, ...)
        )

    MyMultiModalDoc = create_doc(
        'MyMultiModalDoc', image=(Image, ...), text=(Text, ...)
    )

    assert issubclass(MyMultiModalDoc, BaseDocument)

    doc = MyMultiModalDoc(
        image=Image(tensor=np.zeros((3, 224, 224))), text=Text(text='hello')
    )

    assert isinstance(doc.image, BaseDocument)
    assert isinstance(doc.image, Image)
    assert isinstance(doc.text, Text)

    assert doc.text.text == 'hello'
    assert (doc.image.tensor == np.zeros((3, 224, 224))).all()

    MyAudio = create_doc(
        'MyAudio',
        __base__=Audio,
        title=(str, ...),
        tensor=(Optional[AudioNdArray], ...),
    )

    assert issubclass(MyAudio, BaseDocument)
    assert issubclass(MyAudio, Audio)


def test_create_from_typeddict():
    class MyMultiModalDoc(TypedDict):
        image: Image
        text: Text

    with pytest.raises(ValueError):
        _ = create_from_typeddict(MyMultiModalDoc, __base__=BaseModel)

    Doc = create_from_typeddict(MyMultiModalDoc)

    assert issubclass(Doc, BaseDocument)

    class MyAudio(TypedDict):
        title: str
        tensor: Optional[AudioNdArray]

    Doc = create_from_typeddict(MyAudio, __base__=Audio)

    assert issubclass(Doc, BaseDocument)
    assert issubclass(Doc, Audio)
