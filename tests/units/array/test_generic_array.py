from docarray import BaseDoc, DocumentArray
from docarray.base_document import AnyDoc


def test_generic_init():
    class Text(BaseDoc):
        text: str

    da = DocumentArray[Text]([])
    da.document_type == Text

    assert isinstance(da, DocumentArray)


def test_normal_access_init():
    da = DocumentArray([])
    da.document_type == AnyDoc

    assert isinstance(da, DocumentArray)
