import numpy as np
import pytest

from docarray import Document, DocumentArray


@pytest.mark.parametrize('init_args', [None, dict(id=123), Document()])
@pytest.mark.parametrize('copy', [True, False])
def test_construct_doc(init_args, copy):
    Document(init_args, copy)


def test_doc_hash_identical():
    d1 = Document(text='hello')
    d2 = Document(text='hello')
    assert hash(d1) != hash(d2)
    assert d1 != d2
    d1.id = d2.id
    assert hash(d1) == hash(d2)
    assert d1 == d2


def test_doc_hash_complicate_content():
    d1 = Document(text='hello', embedding=np.array([1, 2, 3]), id=1)
    d2 = Document(text='hello', embedding=np.array([1, 2, 3]), id=1)
    assert d1 == d2
    assert hash(d1) == hash(d2)


def test_pop_field():
    d1 = Document(text='hello', embedding=np.array([1, 2, 3]), id=1)
    assert d1.non_empty_fields == ('id', 'text', 'embedding')
    d1.pop('text')
    assert d1.non_empty_fields == ('id', 'embedding')
    d1.pop('id', 'embedding')
    assert d1.non_empty_fields == tuple()

    d1.pop('foobar')
    with pytest.raises(AttributeError):
        assert d1.foobar


def test_clear_fields():
    d1 = Document(text='hello', embedding=np.array([1, 2, 3]), id=1)
    d1.clear()
    assert d1.non_empty_fields == tuple()


def test_to_protobuf():
    with pytest.raises(TypeError):
        Document(text='hello', embedding=np.array([1, 2, 3]), id=1).to_protobuf()

    with pytest.raises(AttributeError):
        Document(tags=1).to_protobuf()

    assert Document(text='hello', embedding=np.array([1, 2, 3])).to_protobuf().text == 'hello'
    assert Document(tags={'hello': 'world'}).to_protobuf().tags
    assert len(Document(chunks=[Document(), Document()]).to_protobuf().chunks) == 2


def test_exclusive_content():
    d = Document(text='hello')
    assert d.content_type == 'text'
    d.buffer = b'123'
    assert d.buffer
    assert not d.text
    assert not d.blob
    assert d.content_type == 'buffer'
    d.blob = [1, 2, 3]
    assert d.blob
    assert not d.buffer
    assert not d.text
    assert d.content_type == 'blob'
    d.text = 'hello'
    assert d.text
    assert not d.buffer
    assert not d.blob
    assert d.content_type == 'text'


def test_content_setter():
    d = Document()
    assert not d.content_type
    d.content = 'hello'
    assert d.content_type == 'text'
    d.content = None
    assert not d.content_type


def test_chunks_matches_setter():
    d = Document(chunks=[Document()])
    assert len(d.chunks) == 1
    assert isinstance(d.chunks, DocumentArray)
