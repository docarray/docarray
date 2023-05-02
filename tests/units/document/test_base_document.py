from typing import List, Optional

import numpy as np
import pytest

from docarray import BaseDoc, DocList
from docarray.typing import NdArray


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


@pytest.fixture
def nested_docs():
    class SimpleDoc(BaseDoc):
        simple_tens: NdArray[10]

    class NestedDoc(BaseDoc):
        docs: DocList[SimpleDoc]
        hello: str = 'world'

    nested_docs = NestedDoc(
        docs=DocList[SimpleDoc]([SimpleDoc(simple_tens=np.ones(10)) for j in range(2)]),
    )

    return nested_docs


def test_nested_to_dict(nested_docs):
    d = nested_docs.dict()
    assert (d['docs'][0]['simple_tens'] == np.ones(10)).all()


def test_nested_to_dict_exclude_1(nested_docs):
    d = nested_docs.dict(exclude={'docs'})
    assert 'docs' not in d.keys()


def test_nested_to_dict_exclude_2(nested_docs):
    d = nested_docs.dict(exclude={'hello'})
    assert 'hello' not in d.keys()


def test_nested_to_dict_exclude_3(nested_docs):  # doto change
    d = nested_docs.dict(exclude={'hello': True})
    assert 'docs' not in d.keys()
    assert 'hello' not in d.keys()
