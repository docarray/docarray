from docarray import BaseDoc, DocList
from docarray.base_doc import AnyDoc


def test_generic_init():
    class Text(BaseDoc):
        text: str

    da = DocList[Text]([])
    da.document_type == Text

    assert isinstance(da, DocList)


def test_normal_access_init():
    da = DocList([])
    da.document_type == AnyDoc

    assert isinstance(da, DocList)
