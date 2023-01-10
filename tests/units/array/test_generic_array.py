from docarray import BaseDocument, DocumentArray
from docarray.document_base import AnyDocument


def test_generic_init():
    class Text(BaseDocument):
        text: str

    da = DocumentArray[Text]([])
    da.document_type == Text

    assert isinstance(da, DocumentArray)


def test_normal_access_init():
    da = DocumentArray([])
    da.document_type == AnyDocument

    assert isinstance(da, DocumentArray)
