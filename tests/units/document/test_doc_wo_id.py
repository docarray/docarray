from docarray import DocList
from docarray.base_doc.doc import BaseDocWithoutId


def test_doc_list():
    class A(BaseDocWithoutId):
        text: str

    cls_doc_list = DocList[A]

    assert isinstance(cls_doc_list, type)
