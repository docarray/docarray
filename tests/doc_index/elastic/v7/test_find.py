import numpy as np
from pydantic import Field

from docarray import BaseDocument
from docarray.doc_index.backends.elastic_doc_index import ElasticDocumentIndex
from docarray.typing import NdArray
from tests.doc_index.elastic.fixture import start_storage_v7  # noqa: F401
from tests.doc_index.elastic.fixture import FlatDoc, SimpleDoc


def test_find_simple_schema():
    class SimpleSchema(BaseDocument):
        tens: NdArray[10]

    store = ElasticDocumentIndex[SimpleSchema]()

    index_docs = [SimpleDoc(tens=np.random.rand(10)) for _ in range(10)]
    store.index(index_docs)

    query = index_docs[-1]
    docs, scores = store.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    print([doc.id for doc in docs])
    print(scores)

    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens, index_docs[-1].tens)


def test_find_flat_schema():
    class FlatSchema(BaseDocument):
        tens_one: NdArray = Field(dims=10)
        tens_two: NdArray = Field(dims=50)

    store = ElasticDocumentIndex[FlatSchema]()

    index_docs = [
        FlatDoc(tens_one=np.random.rand(10), tens_two=np.random.rand(50))
        for _ in range(10)
    ]
    store.index(index_docs)

    query = index_docs[-1]

    # find on tens_one
    docs, scores = store.find(query, search_field='tens_one', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens_one, index_docs[-1].tens_one)
    assert np.allclose(docs[0].tens_two, index_docs[-1].tens_two)

    # find on tens_two
    docs, scores = store.find(query, search_field='tens_two', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens_one, index_docs[-1].tens_one)
    assert np.allclose(docs[0].tens_two, index_docs[-1].tens_two)


def test_find_nested_schema():
    class SimpleDoc(BaseDocument):
        tens: NdArray[10]

    class NestedDoc(BaseDocument):
        d: SimpleDoc
        tens: NdArray[10]

    class DeepNestedDoc(BaseDocument):
        d: NestedDoc
        tens: NdArray = Field(dims=10)

    store = ElasticDocumentIndex[DeepNestedDoc]()

    index_docs = [
        DeepNestedDoc(
            d=NestedDoc(d=SimpleDoc(tens=np.random.rand(10)), tens=np.random.rand(10)),
            tens=np.random.rand(10),
        )
        for _ in range(10)
    ]
    store.index(index_docs)

    query = index_docs[-1]

    # find on root level
    docs, scores = store.find(query, search_field='tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens, index_docs[-1].tens)

    # find on first nesting level
    docs, scores = store.find(query, search_field='d__tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].d.tens, index_docs[-1].d.tens)

    # find on second nesting level
    docs, scores = store.find(query, search_field='d__d__tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].d.d.tens, index_docs[-1].d.d.tens)


def test_find_batched():
    store = ElasticDocumentIndex[SimpleDoc]()

    index_docs = [SimpleDoc(tens=np.random.rand(10)) for _ in range(10)]
    store.index(index_docs)

    queries = index_docs[-2:]
    docs_batched, scores_batched = store.find_batched(
        queries, search_field='tens', limit=5
    )

    for docs, scores, query in zip(docs_batched, scores_batched, queries):
        assert len(docs) == 5
        assert len(scores) == 5
        assert docs[0].id == query.id
        assert np.allclose(docs[0].tens, query.tens)


def test_filter():
    import itertools

    class MyDoc(BaseDocument):
        A: bool
        B: int
        C: float

    store = ElasticDocumentIndex[MyDoc]()

    A_list = [True, False]
    B_list = [1, 2]
    C_list = [1.5, 2.5]

    # cross product of all possible combinations
    combinations = itertools.product(A_list, B_list, C_list)
    index_docs = [MyDoc(A=A, B=B, C=C) for A, B, C in combinations]
    store.index(index_docs)

    filter_query = {'term': {'A': True}}
    docs = store.filter(filter_query)
    assert len(docs) > 0
    for doc in docs:
        assert doc.A

    filter_query = {'term': {'B': 1}}
    docs = store.filter(filter_query)
    assert len(docs) > 0
    for doc in docs:
        assert doc.B == 1

    filter_query = {'term': {'C': 1.5}}
    docs = store.filter(filter_query)
    assert len(docs) > 0
    for doc in docs:
        assert doc.C == 1.5


def test_text_search():
    class MyDoc(BaseDocument):
        text: str

    store = ElasticDocumentIndex[MyDoc]()
    index_docs = [
        MyDoc(text='hello world'),
        MyDoc(text='never gonna give you up'),
        MyDoc(text='we are the world'),
    ]
    store.index(index_docs)

    query = 'world'
    docs, scores = store.text_search(query, search_field='text')

    assert len(docs) == 2
    assert len(scores) == 2
    assert docs[0].text.index(query) >= 0
    assert docs[1].text.index(query) >= 0

    queries = ['world', 'never']
    docs, scores = store.text_search_batched(queries, search_field='text')
    for query, da, score in zip(queries, docs, scores):
        assert len(da) > 0
        assert len(score) > 0
        for doc in da:
            assert doc.text.index(query) >= 0


def test_query_builder():
    class MyDoc(BaseDocument):
        tens: NdArray[10] = Field(dims=10)
        num: int
        text: str

    store = ElasticDocumentIndex[MyDoc]()
    index_docs = [
        MyDoc(
            id=f'{i}', tens=np.random.rand(10), num=int(i / 2), text=f'text {int(i/2)}'
        )
        for i in range(10)
    ]
    store.index(index_docs)

    # build_query
    q = store.build_query()
    assert isinstance(q, store.QueryBuilder)

    # filter
    q = store.build_query().filter('term', num=0).build()
    docs, _ = store.execute_query(q)
    assert [doc['id'] for doc in docs] == ['0', '1']

    # exclude
    q = store.build_query(extra={"size": 3}).exclude('term', num=0).build()
    docs, _ = store.execute_query(q)
    assert [doc['id'] for doc in docs] == ['2', '3', '4']

    # script_fields
    q = (
        store.build_query(extra={"size": 3})
        .script_fields(bigger_num="doc['num'].value + 10")
        .build()
    )
    docs, _ = store.execute_query(q)
    assert [doc['bigger_num'][0] for doc in docs] == [10, 10, 11]

    # query (text search)
    q = store.build_query().query('match', text='0').build()
    docs, _ = store.execute_query(q)
    assert [doc['id'] for doc in docs] == ['0', '1']

    # query (find)
    from elasticsearch_dsl import Q

    d = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'tens') + 1.0",
                "params": {'query_vector': index_docs[-1].tens},
            },
        }
    }
    q = store.build_query().query(Q(d)).sort("_score").build()
    docs, scores = store.execute_query(q)
    assert docs[0]['id'] == index_docs[-1].id
    assert np.allclose(docs[0]['tens'], index_docs[-1].tens)

    # combination
    q = (
        store.build_query()
        .query(Q(d))
        .exclude('term', num=5)
        .filter(Q("term", num=0) | Q("term", num=1))
        .build()
    )
    docs, _ = store.execute_query(q)
    assert sorted([doc['id'] for doc in docs]) == ['0', '1', '2', '3']
