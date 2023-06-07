from typing import List, Optional

import numpy as np
import pytest

from docarray import DocList
from docarray.base_doc.doc import BaseDoc
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


def test_init_class():
    class MyDoc(BaseDoc):
        content: str

    doc = MyDoc(content='hello world')
    assert doc.content == 'hello world'
    assert doc.id is not None


def test_nested_class():
    class InnerDoc(BaseDoc):
        content: str

    class OuterDoc(BaseDoc):
        inner: InnerDoc

    doc = OuterDoc(inner=InnerDoc(content='hello world'))
    assert doc.inner.content == 'hello world'
    assert doc.id is not None


def test_equal_nested_docs():
    import numpy as np

    from docarray import BaseDoc, DocList
    from docarray.typing import NdArray

    class SimpleDoc(BaseDoc):
        simple_tens: NdArray[10]

    class NestedDoc(BaseDoc):
        docs: DocList[SimpleDoc]

    nested_docs = [SimpleDoc(simple_tens=np.ones(10)) for j in range(2)]

    nested_docs = NestedDoc(
        docs=DocList[SimpleDoc](nested_docs),
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


def test_nested_to_model_dump(nested_docs):
    d = nested_docs.model_dump()
    assert (d['docs'][0]['simple_tens'] == np.ones(10)).all()


def test_nested_to_model_dump_exclude(nested_docs):
    d = nested_docs.model_dump(exclude={'docs'})
    assert 'docs' not in d.keys()


def test_nested_to_model_dump_exclude_set(nested_docs):
    d = nested_docs.model_dump(exclude={'hello'})
    assert 'hello' not in d.keys()


def test_nested_to_model_dump_exclude_model_dump(nested_docs):
    d = nested_docs.model_dump(exclude={'hello': True})
    assert 'hello' not in d.keys()


def test_nested_to_json(nested_docs):
    d = nested_docs.json()
    nested_docs.__class__.parse_raw(d)


@pytest.fixture
def nested_none_docs():
    class SimpleDoc(BaseDoc):
        simple_tens: NdArray[10]

    class NestedDoc(BaseDoc):
        docs: Optional[DocList[SimpleDoc]]
        hello: str = 'world'

    nested_docs = NestedDoc()

    return nested_docs


def test_nested_none_to_model_dump(nested_none_docs):
    d = nested_none_docs.model_dump()
    assert d == {'docs': None, 'hello': 'world', 'id': nested_none_docs.id}


def test_nested_none_to_json(nested_none_docs):
    d = nested_none_docs.json()
    d = nested_none_docs.__class__.parse_raw(d)
    assert d.model_dump() == {'docs': None, 'hello': 'world', 'id': nested_none_docs.id}
