from docarray import BaseDoc, DocArray
from docarray.base_doc import AnyDoc


def test_generic_init():
    class Text(BaseDoc):
        text: str

    da = DocArray[Text]([])
    da._document_type == Text

    assert isinstance(da, DocArray)


def test_normal_access_init():
    da = DocArray([])
    da._document_type == AnyDoc

    assert isinstance(da, DocArray)
