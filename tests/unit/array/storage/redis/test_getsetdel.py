from abc import ABC

import numpy as np
import pytest
from docarray import Document, DocumentArray
from docarray.array.storage.base.helper import Offset2ID
from docarray.array.storage.memory import SequenceLikeMixin
from docarray.array.storage.redis.getsetdel import GetSetDelMixin
from docarray.array.storage.redis.backend import BackendMixin, RedisConfig


class StorageMixins(BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...


class DocumentArrayDummy(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def _load_offset2ids(self):
        pass

    def _save_offset2ids(self):
        pass


@pytest.fixture(scope='function')
def tag_indices():
    tag_indices = ['tag_1', 'tag_2']
    return tag_indices


@pytest.fixture(scope='function')
def columns():
    columns = [
        ('col_str', 'str'),
        ('col_bytes', 'bytes'),
        ('col_int', 'int'),
        ('col_float', 'float'),
        ('col_long', 'long'),
        ('col_double', 'double'),
    ]
    return columns


@pytest.fixture(scope='function')
def da_redis(tag_indices, columns):
    cfg = RedisConfig(n_dim=3, flush=True, tag_indices=tag_indices, columns=columns)
    da_redis = DocumentArrayDummy(storage='redis', config=cfg)
    return da_redis


@pytest.mark.parametrize(
    'embedding', [[1, 2, 3], [1.0, 2.0, 3.0], [1, 2, 3, 4, 5], None]
)
@pytest.mark.parametrize('text', ['test_text', None])
@pytest.mark.parametrize(
    'tag',
    [
        {'tag_1': 'tag1'},
        {'tag_1': 'tag1', 'tag_2': 'tag2'},
        {'tag_1': 'tag1', 'tag_2': 'tag2', 'tag_3': 'tag3'},
        None,
    ],
)
@pytest.mark.parametrize(
    'col',
    [
        {'col_str': 'hello', 'col_bytes': b'world'},
        {'col_int': 1, 'col_float': 1.0},
        {'col_long': 123, 'col_double': 1.1},
        None,
    ],
)
def test_document_to_embedding(
    embedding, text, tag, col, da_redis, columns, tag_indices, start_storage
):
    tags = {}
    if tag is not None:
        tags.update(tag)
    if col is not None:
        tags.update(col)
    doc = Document(embedding=embedding, text=text, tags=tags)
    payload = da_redis._document_to_redis(doc)

    if embedding is None:
        assert np.allclose(
            np.frombuffer(payload['embedding'], dtype=np.float32), np.zeros((3))
        )
    else:
        assert np.allclose(
            np.frombuffer(payload['embedding'], dtype=np.float32), np.array(embedding)
        )

    if text is None:
        with pytest.raises(KeyError):
            payload['text']
    else:
        assert payload['text'] == text

    for col, _ in columns:
        if col in tags:
            assert payload[col] == tags[col]
        else:
            with pytest.raises(KeyError):
                payload[col]

    for tag in tag_indices:
        if tag in tags:
            assert payload[tag] == tags[tag]
        else:
            with pytest.raises(KeyError):
                payload[tag]

    for key in tags:
        if (key not in tag_indices) and (key not in (col[0] for col in columns)):
            assert key not in payload


@pytest.mark.parametrize(
    'doc',
    [
        Document(id='0'),
        Document(id='1', text='hello world'),
        Document(id='2', embedding=[1, 2, 3], tags={'tag_1': 'tag1', 'tag_2': 'tag2'}),
        Document(
            text='hello world',
            embedding=[1, 2, 3],
            tags={'tag_1': 'tag1', 'tag_2': 'tag2'},
            chunks=[Document(text='token1'), Document(text='token2')],
        ),
    ],
)
def test_setgetdel_doc_by_id(doc, da_redis, start_storage):
    da_redis._set_doc_by_id(doc.id, doc)
    doc_get = da_redis._get_doc_by_id(doc.id)
    assert doc == doc_get

    da_redis._del_doc_by_id(doc.id)
    with pytest.raises(KeyError):
        da_redis._get_doc_by_id(doc.id)


def test_clear_storage(da_redis, start_storage):
    for i in range(3):
        doc = Document(id=str(i))
    da_redis._set_doc_by_id(str(i), doc)

    da_redis._clear_storage()

    for i in range(3):
        with pytest.raises(KeyError):
            da_redis._get_doc_by_id(i)


def test_offset2ids(da_redis, start_storage):
    ids = [str(i) for i in range(3)]
    for id in ids:
        doc = Document(id=id)
        da_redis._set_doc_by_id(id, doc)
    da_redis._offset2ids = Offset2ID(ids)
    da_redis._save_offset2ids()
    da_redis._load_offset2ids()
    assert da_redis._offset2ids.ids == ids
