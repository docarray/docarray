import numpy as np
import pytest

from docarray import Document, DocumentArray
from docarray.array.chunk import ChunkArray
from docarray.array.match import MatchArray


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
    d = Document(chunks=[Document()], matches=[Document(), Document()])
    assert len(d.chunks) == 1
    assert len(d.matches) == 2
    assert isinstance(d.chunks, DocumentArray)
    assert isinstance(d.chunks, ChunkArray)
    assert isinstance(d.matches, DocumentArray)
    assert isinstance(d.matches, MatchArray)


def test_empty_doc_chunks_matches():
    assert isinstance(Document().chunks, DocumentArray)
    assert isinstance(Document().matches, DocumentArray)
    assert isinstance(Document().matches, MatchArray)
    assert isinstance(Document().chunks, ChunkArray)

    d = Document()
    d.chunks.append(Document())
    assert isinstance(d.chunks, ChunkArray)

    d.chunks = [Document(), Document()]
    assert isinstance(d.chunks, ChunkArray)


def test_chunk_match_increase_granularity():
    d = Document()
    d.chunks.append(Document())
    assert d.chunks[0].granularity == 1
    assert id(d.chunks.reference_doc) == id(d)
    d.matches.append(Document())
    assert d.matches[0].adjacency == 1
    assert id(d.matches.reference_doc) == id(d)

    d = d.chunks[0]
    d.chunks.append(Document())
    assert d.chunks[0].granularity == 2
    assert id(d.chunks.reference_doc) == id(d)

    d.matches.append(Document())
    assert d.matches[0].adjacency == 1
    assert id(d.matches.reference_doc) == id(d)


def test_offset():
    d1 = Document(offset=1.0)
    d2 = Document()
    d2.offset = 1.0
    assert d1.offset == d2.offset == 1.0


def test_exclusive_content_2():
    d = Document(text='hello', buffer=b'sda')
    assert len(d.non_empty_fields) == 2
    d.content = b'sda'
    assert d.content == b'sda'
    assert 'buffer' in d.non_empty_fields
    d = Document(content='hello')
    assert d.content_type == 'text'
    d = Document(content=b'hello')
    assert d.content_type == 'buffer'
    d = Document(content=[1,2,3])
    assert d.content_type == 'blob'

