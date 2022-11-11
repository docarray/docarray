import numpy as np

from docarray import DocumentArray, Document, Image, Text


def test_multi_modal_doc_proto():
    class MyMultiModalDoc(Document):
        image: Image
        text: Text

    class MySUperDoc(Document):
        doc: MyMultiModalDoc
        description: str

    doc = MyMultiModalDoc(
        image=Image(tensor=np.zeros((3, 224, 224))), text=Text(text='hello')
    )

    MyMultiModalDoc.from_protobuf(doc.to_protobuf())
