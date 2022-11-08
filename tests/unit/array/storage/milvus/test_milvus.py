import pytest
from docarray import Document
from docarray.array.milvus import DocumentArrayMilvus, MilvusConfig
from pymilvus import loading_progress
import numpy as np


def _is_fully_loaded(da):
    collections = da._collection, da._offset2id_collection
    fully_loaded = True
    for coll in collections:
        coll_loaded = (
            loading_progress(coll.name, using=da._connection_alias)['loading_progress']
            == '100%'
        )
        fully_loaded = fully_loaded and coll_loaded
    return fully_loaded


def _is_fully_released(da):
    collections = da._collection, da._offset2id_collection
    fully_released = True
    for coll in collections:
        coll_released = (
            loading_progress(coll.name, using=da._connection_alias)['loading_progress']
            == '0%'
        )
        fully_released = fully_released and coll_released
    return fully_released


def test_memory_release(start_storage):
    da = DocumentArrayMilvus(
        config={
            'n_dim': 10,
        },
    )
    da.extend([Document(embedding=np.random.random([10])) for _ in range(10)])
    da.find(Document(embedding=np.random.random([10])))
    assert _is_fully_released(da)


def test_memory_cntxt_mngr(start_storage):
    da = DocumentArrayMilvus(
        config={
            'n_dim': 10,
        },
    )

    # `with da` context manager
    assert _is_fully_released(da)
    with da:
        assert _is_fully_loaded(da)
        pass
    assert _is_fully_released(da)

    # `da.loaded_collection` context manager
    with da.loaded_collection(), da.loaded_collection(da._offset2id_collection):
        assert _is_fully_loaded(da)
        pass
    assert _is_fully_released(da)

    # both combined
    with da:
        assert _is_fully_loaded(da)
        with da.loaded_collection(), da.loaded_collection(da._offset2id_collection):
            assert _is_fully_loaded(da)
            pass
        assert _is_fully_loaded(da)
    assert _is_fully_released(da)


@pytest.fixture()
def mock_response():
    class MockHit:
        @property
        def entity(self):
            return {'serialized': Document().to_base64()}

    return [[MockHit()]]


@pytest.mark.parametrize(
    'method,meth_input',
    [
        ('append', [Document(embedding=np.random.random([10]))]),
        ('extend', [[Document(embedding=np.random.random([10]))]]),
        ('find', [Document(embedding=np.random.random([10]))]),
        ('insert', [0, Document(embedding=np.random.random([10]))]),
    ],
)
def test_consistency_level(start_storage, mocker, method, meth_input, mock_response):
    init_consistency = 'Session'
    da = DocumentArrayMilvus(
        config={
            'n_dim': 10,
            'consistency_level': init_consistency,
        },
    )

    # patch Milvus collection
    patch_methods = ['insert', 'search', 'delete', 'query']
    for m in patch_methods:
        setattr(da._collection, m, mocker.Mock(return_value=mock_response))

    # test consistency level set in config
    getattr(da, method)(*meth_input)
    for m in patch_methods:
        mock_meth = getattr(da._collection, m)
        for args, kwargs in mock_meth.call_args_list:
            if 'consistency_level' in kwargs:
                assert kwargs['consistency_level'] == init_consistency

    # reset the mocks
    for m in patch_methods:
        setattr(da._collection, m, mocker.Mock(return_value=mock_response))

    # test dynamic consistency level
    new_consistency = 'Strong'
    getattr(da, method)(*meth_input, consistency_level=new_consistency)
    for m in patch_methods:
        mock_meth = getattr(da._collection, m)
        for args, kwargs in mock_meth.call_args_list:
            if 'consistency_level' in kwargs:
                assert kwargs['consistency_level'] == new_consistency


@pytest.mark.parametrize(
    'method,meth_input',
    [
        ('append', [Document(embedding=np.random.random([10]))]),
        ('extend', [[Document(embedding=np.random.random([10]))]]),
        ('insert', [0, Document(embedding=np.random.random([10]))]),
    ],
)
def test_batching(start_storage, mocker, method, meth_input, mock_response):
    init_batch_size = 5
    da = DocumentArrayMilvus(
        config={
            'n_dim': 10,
            'batch_size': init_batch_size,
        },
    )

    # patch Milvus collection
    patch_methods = ['insert', 'search', 'delete', 'query']
    for m in patch_methods:
        setattr(da._collection, m, mocker.Mock(return_value=mock_response))

    # test batch_size set in config
    getattr(da, method)(*meth_input)
    for m in patch_methods:
        mock_meth = getattr(da._collection, m)
        for args, kwargs in mock_meth.call_args_list:
            if 'batch_size' in kwargs:
                assert kwargs['batch_size'] == init_batch_size

    # reset the mocks
    for m in patch_methods:
        setattr(da._collection, m, mocker.Mock(return_value=mock_response))

    # test dynamic consistency level
    new_batch_size = 100
    getattr(da, method)(*meth_input, batch_size=new_batch_size)
    for m in patch_methods:
        mock_meth = getattr(da._collection, m)
        for args, kwargs in mock_meth.call_args_list:
            if 'batch_size' in kwargs:
                assert kwargs['batch_size'] == new_batch_size
