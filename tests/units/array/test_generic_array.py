from docarray import BaseDoc, DocList
from docarray.base_doc import AnyDoc


def test_generic_init():
    class Text(BaseDoc):
        text: str

    da = DocList[Text]([])
    da.doc_type == Text

    assert isinstance(da, DocList)


def test_normal_access_init():
    da = DocList([])
    da.doc_type == AnyDoc

    assert isinstance(da, DocList)
