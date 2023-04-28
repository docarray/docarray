from typing import List, Optional

from docarray.base_doc.doc import BaseDoc


def test_base_document_init():
    doc = BaseDoc()

    assert doc.id is not None


def test_update():
    class MyDocument(BaseDoc):
        content: str
        title: Optional[str] = None
        tags_: List

    doc1 = MyDocument(
        content='Core content of the document', title='Title', tags_=['python', 'AI']
    )
    doc2 = MyDocument(content='Core content updated', tags_=['docarray'])

    doc1.update(doc2)
    assert doc1.content == 'Core content updated'
    assert doc1.title == 'Title'
    assert doc1.tags_ == ['python', 'AI', 'docarray']


def test_equal_nested_docs():
    import numpy as np

    from docarray import BaseDoc, DocList
    from docarray.typing import NdArray

    class SimpleDoc(BaseDoc):
        simple_tens: NdArray[10]

    class NestedDoc(BaseDoc):
        docs: DocList[SimpleDoc]

    nested_docs = NestedDoc(
        docs=DocList[SimpleDoc]([SimpleDoc(simple_tens=np.ones(10)) for j in range(2)]),
    )

    assert nested_docs == nested_docs
