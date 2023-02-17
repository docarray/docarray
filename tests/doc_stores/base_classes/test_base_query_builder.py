from docarray.storage.abstract_doc_store import BaseQueryBuilder


class DummyQueryBuilder(BaseQueryBuilder):
    def build(self):
        return self._queries


def test_collect_calls():
    qb = DummyQueryBuilder()
    qb.find(find='find', param='param')
    qb.filter(filter='filter', param='param')
    qb.text_search(text='text', param='param')
    qb.filter_batched(filter_batched='filter_batched', param='param')
    qb.text_search_batched(text_batched='text_batched', param='param')
    qb.find_batched(find_batched='find_batched', param='param')

    collected_queries = qb._queries
    assert collected_queries[0] == ('find', {'find': 'find', 'param': 'param'})
    assert collected_queries[1] == ('filter', {'filter': 'filter', 'param': 'param'})
    assert collected_queries[2] == ('text_search', {'text': 'text', 'param': 'param'})
    assert collected_queries[3] == (
        'filter_batched',
        {'filter_batched': 'filter_batched', 'param': 'param'},
    )
    assert collected_queries[4] == (
        'text_search_batched',
        {'text_batched': 'text_batched', 'param': 'param'},
    )
    assert collected_queries[5] == (
        'find_batched',
        {'find_batched': 'find_batched', 'param': 'param'},
    )
