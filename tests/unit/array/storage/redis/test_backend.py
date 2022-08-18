from abc import ABC

import numpy as np
import pytest
from docarray import DocumentArray
from docarray.array.storage.base.helper import Offset2ID
from docarray.array.storage.memory import GetSetDelMixin, SequenceLikeMixin
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


type_convert = {
    'int': b'NUMERIC',
    'float': b'NUMERIC',
    'double': b'NUMERIC',
    'long': b'NUMERIC',
    'str': b'TEXT',
    'bytes': b'TEXT',
}


@pytest.fixture(scope='function')
def da_redis():
    cfg = RedisConfig(n_dim=128, flush=True)
    da_redis = DocumentArrayDummy(storage='redis', config=cfg)
    return da_redis


@pytest.mark.parametrize('distance', ['L2', 'IP', 'COSINE'])
@pytest.mark.parametrize(
    'method,initial_cap,ef_construction,block_size',
    [
        ('HNSW', None, None, None),
        ('HNSW', 10, 250, None),
        ('HNSW', 10, 250, 1000000),
        ('FLAT', 10, 250, 1000000),
    ],
)
@pytest.mark.parametrize('tag_indices', [['attr3'], ['attr3', 'attr4']])
@pytest.mark.parametrize(
    'columns',
    [
        [('attr1', 'str'), ('attr2', 'bytes')],
        [('attr1', 'int'), ('attr2', 'float')],
        [('attr1', 'double'), ('attr2', 'long')],
    ],
)
@pytest.mark.parametrize('index_text', [True, False])
@pytest.mark.parametrize(
    'redis_config',
    [
        {'decode_responses': True},
        {'decode_responses': False},
        {'retry_on_timeout': True},
        {'decode_responses': True, 'retry_on_timeout': True},
        {},
    ],
)
def test_init_storage(
    distance,
    tag_indices,
    columns,
    method,
    initial_cap,
    ef_construction,
    block_size,
    index_text,
    redis_config,
    start_storage,
):
    cfg = RedisConfig(
        n_dim=128,
        distance=distance,
        flush=True,
        tag_indices=tag_indices,
        columns=columns,
        method=method,
        initial_cap=initial_cap,
        ef_construction=ef_construction,
        block_size=block_size,
        index_text=index_text,
        redis_config=redis_config,
    )
    redis_da = DocumentArrayDummy(storage='redis', config=cfg)

    assert redis_da._client.info()['tcp_port'] == redis_da._config.port
    assert redis_da._client.ft().info()['attributes'][0][1] == b'embedding'
    assert redis_da._client.ft().info()['attributes'][0][5] == b'VECTOR'

    for i in range(len(tag_indices)):
        assert redis_da._client.ft().info()['attributes'][i + 1][1] == bytes(
            redis_da._config.tag_indices[i], 'utf-8'
        )
        assert redis_da._client.ft().info()['attributes'][i + 1][5] == b'TEXT'

    for i in range(len(columns)):
        assert redis_da._client.ft().info()['attributes'][i + len(tag_indices) + 1][
            1
        ] == bytes(redis_da._config.columns[i][0], 'utf-8')
        assert (
            redis_da._client.ft().info()['attributes'][i + len(tag_indices) + 1][5]
            == type_convert[redis_da._config.columns[i][1]]
        )

    if index_text:
        assert redis_da._client.ft().info()['attributes'][-1][1] == b'text'
        assert redis_da._client.ft().info()['attributes'][-1][5] == b'TEXT'


def test_init_storage_update_schema(start_storage):
    index = 'aaa'
    cfg = RedisConfig(n_dim=128, tag_indices=['attr1'], index_name=index, flush=True)
    redis_da = DocumentArrayDummy(storage='redis', config=cfg)
    assert redis_da._client.ft(index).info()['attributes'][1][1] == b'attr1'

    cfg = RedisConfig(n_dim=128, tag_indices=['attr2'], update_schema=False)
    redis_da = DocumentArrayDummy(storage='redis', config=cfg)
    assert redis_da._client.ft(index).info()['attributes'][1][1] == b'attr1'

    index2 = 'bbb'
    cfg = RedisConfig(
        n_dim=128, tag_indices=['attr2'], index_name=index2, update_schema=True
    )
    redis_da = DocumentArrayDummy(storage='redis', config=cfg)
    assert redis_da._client.ft(index).info()['attributes'][1][1] == b'attr1'
    assert redis_da._client.ft(index2).info()['attributes'][1][1] == b'attr2'


def test_init_storage_empty_config(start_storage):
    with pytest.raises(ValueError):
        redis_da = DocumentArrayDummy(storage='redis')
