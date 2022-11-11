import numpy as np

from docarray.document import AnyDocument, BaseDocument
from docarray.typing import Tensor


def test_any_doc():
    class InnerDocument(BaseDocument):
        text: str
        tensor: Tensor

    class CustomDoc(BaseDocument):
        inner: InnerDocument
        text: str

    doc = CustomDoc(
        text='bye', inner=InnerDocument(text='hello', tensor=np.zeros((3, 224, 224)))
    )

    any_doc = AnyDocument(**doc.__dict__)

    assert any_doc.text == doc.text
    assert any_doc.inner.text == doc.inner.text
    assert (any_doc.inner.tensor == doc.inner.tensor).all()
