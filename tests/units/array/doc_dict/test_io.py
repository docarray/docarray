import pytest

from docarray import BaseDoc, DocDict


class NestedDoc(BaseDoc):
    text: str


class MyDoc(BaseDoc):
    doc: NestedDoc


@pytest.fixture
def docs():
    return DocDict[MyDoc](
        x=MyDoc(id='x', doc=NestedDoc(text='a')),
        y=MyDoc(id='y', doc=NestedDoc(text='b')),
    )


def test_to_protobuf(docs):
    return docs.to_protobuf()


def test_from_protobuf(docs):
    docs2 = DocDict[MyDoc].from_protobuf(docs.to_protobuf())

    assert docs == docs2
    assert list(docs) == ['x', 'y']


def test_to_json(docs):
    return docs.to_json()


def test_from_json(docs):
    docs2 = DocDict[MyDoc].from_json(docs.to_json())

    assert docs == docs2
    assert list(docs) == ['x', 'y']
