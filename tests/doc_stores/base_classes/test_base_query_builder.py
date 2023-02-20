import pytest

from docarray.storage.abstract_doc_store import BaseDocumentStore, composable


class DummyQueryBuilder(BaseDocumentStore.QueryBuilder):
    def build(self):
        return self._queries


def _identity(*x, **y):
    return x, y


class DummyDocStore(BaseDocumentStore):
    _query_builder_cls = DummyQueryBuilder
    _db_config_cls = None
    _runtime_config_cls = None

    def python_type_to_db_type(self, x):
        return str

    index = _identity
    num_docs = _identity
    __delitem__ = _identity
    __getitem__ = _identity
    execute_query = _identity

    @composable
    def find(self, *args, **kwargs):
        return _identity(*args, **kwargs)

    @composable
    def filter(self, *args, **kwargs):
        return _identity(*args, **kwargs)

    @composable
    def text_search(self, *args, **kwargs):
        return _identity(*args, **kwargs)

    find_batched = _identity
    filter_batched = _identity
    text_search_batched = _identity


def test_collect_calls():
    qb = DummyQueryBuilder()
    qb.find(find='find', param='param')
    qb.filter(filter='filter', param='param')
    qb.text_search(text='text', param='param')

    collected_queries = qb._queries
    assert collected_queries[0] == ('find', {'find': 'find', 'param': 'param'})
    assert collected_queries[1] == ('filter', {'filter': 'filter', 'param': 'param'})
    assert collected_queries[2] == ('text_search', {'text': 'text', 'param': 'param'})


def test_fluent_interface():
    qb = DummyQueryBuilder()
    qb.find(find='find', param='param').filter(
        filter='filter', param='param'
    ).text_search(text='text', param='param')

    collected_queries = qb._queries
    assert collected_queries[0] == ('find', {'find': 'find', 'param': 'param'})
    assert collected_queries[1] == ('filter', {'filter': 'filter', 'param': 'param'})
    assert collected_queries[2] == ('text_search', {'text': 'text', 'param': 'param'})


def test_not_composable_raises():
    qb = DummyQueryBuilder()
    with pytest.raises(NotImplementedError):
        qb.filter_batched(filter_batched='filter_batched', param='param')
    with pytest.raises(NotImplementedError):
        qb.text_search_batched(text_batched='text_batched', param='param')
    with pytest.raises(NotImplementedError):
        qb.find_batched(find_batched='find_batched', param='param')
