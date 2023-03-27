import numpy as np

from docarray.base_document import AnyDoc, BaseDoc
from docarray.typing import NdArray


def test_any_doc():
    class InnerDocument(BaseDoc):
        text: str
        tensor: NdArray

    class CustomDoc(BaseDoc):
        inner: InnerDocument
        text: str

    doc = CustomDoc(
        text='bye', inner=InnerDocument(text='hello', tensor=np.zeros((3, 224, 224)))
    )

    any_doc = AnyDoc(**doc.__dict__)

    assert any_doc.text == doc.text
    assert any_doc.inner.text == doc.inner.text
    assert (any_doc.inner.tensor == doc.inner.tensor).all()
