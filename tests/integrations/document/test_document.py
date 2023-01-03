import numpy as np

from docarray import BaseDocument, DocumentArray, Image, Text


def test_multi_modal_doc():
    class MyMultiModalDoc(BaseDocument):
        image: Image
        text: Text

    doc = MyMultiModalDoc(
        image=Image(tensor=np.zeros((3, 224, 224))), text=Text(text='hello')
    )

    assert isinstance(doc.image, BaseDocument)
    assert isinstance(doc.image, Image)
    assert isinstance(doc.text, Text)

    assert doc.text.text == 'hello'
    assert (doc.image.tensor == np.zeros((3, 224, 224))).all()


def test_nested_chunks_document():
    class ChunksDocument(BaseDocument):
        text: str
        images: DocumentArray[Image]

    doc = ChunksDocument(
        text='hello',
        images=DocumentArray[Image]([Image() for _ in range(10)]),
    )

    assert isinstance(doc.images, DocumentArray)
