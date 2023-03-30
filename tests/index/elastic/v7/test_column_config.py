import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import ElasticV7DocIndex
from tests.index.elastic.fixture import start_storage_v7  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.index]


def test_column_config():
    class MyDoc(BaseDoc):
        text: str
        color: str = Field(col_type='keyword')

    store = ElasticV7DocIndex[MyDoc]()
    index_docs = [
        MyDoc(id='0', text='hello world', color='red'),
        MyDoc(id='1', text='never gonna give you up', color='blue'),
        MyDoc(id='2', text='we are the world', color='green'),
    ]
    store.index(index_docs)

    query = 'world'
    docs, _ = store.text_search(query, search_field='text')
    assert [doc.id for doc in docs] == ['0', '2']

    filter_query = {'terms': {'color': ['red', 'blue']}}
    docs = store.filter(filter_query)
    assert [doc.id for doc in docs] == ['0', '1']


def test_field_object():
    class MyDoc(BaseDoc):
        manager: dict = Field(
            properties={
                'age': {'type': 'integer'},
                'name': {
                    'properties': {
                        'first': {'type': 'keyword'},
                        'last': {'type': 'keyword'},
                    }
                },
            }
        )

    store = ElasticV7DocIndex[MyDoc]()
    doc = [
        MyDoc(manager={'age': 25, 'name': {'first': 'Rachel', 'last': 'Green'}}),
        MyDoc(manager={'age': 30, 'name': {'first': 'Monica', 'last': 'Geller'}}),
        MyDoc(manager={'age': 35, 'name': {'first': 'Phoebe', 'last': 'Buffay'}}),
    ]
    store.index(doc)
    id_ = doc[0].id
    assert store[id_].id == id_
    assert store[id_].manager == doc[0].manager

    filter_query = {'range': {'manager.age': {'gte': 30}}}
    docs = store.filter(filter_query)
    assert [doc.id for doc in docs] == [doc[1].id, doc[2].id]


def test_field_geo_point():
    class MyDoc(BaseDoc):
        location: dict = Field(col_type='geo_point')

    store = ElasticV7DocIndex[MyDoc]()
    doc = [
        MyDoc(location={'lat': 40.12, 'lon': -72.34}),
        MyDoc(location={'lat': 41.12, 'lon': -73.34}),
        MyDoc(location={'lat': 42.12, 'lon': -74.34}),
    ]
    store.index(doc)

    query = {
        'query': {
            'geo_bounding_box': {
                'location': {
                    'top_left': {'lat': 42, 'lon': -74},
                    'bottom_right': {'lat': 40, 'lon': -72},
                }
            }
        },
    }

    docs, _ = store.execute_query(query)
    assert [doc['id'] for doc in docs] == [doc[0].id, doc[1].id]


def test_field_range():
    class MyDoc(BaseDoc):
        expected_attendees: dict = Field(col_type='integer_range')
        time_frame: dict = Field(col_type='date_range', format='yyyy-MM-dd')

    store = ElasticV7DocIndex[MyDoc]()
    doc = [
        MyDoc(
            expected_attendees={'gte': 10, 'lt': 20},
            time_frame={'gte': '2023-01-01', 'lt': '2023-02-01'},
        ),
        MyDoc(
            expected_attendees={'gte': 20, 'lt': 30},
            time_frame={'gte': '2023-02-01', 'lt': '2023-03-01'},
        ),
        MyDoc(
            expected_attendees={'gte': 30, 'lt': 40},
            time_frame={'gte': '2023-03-01', 'lt': '2023-04-01'},
        ),
    ]
    store.index(doc)

    query = {
        'query': {
            'bool': {
                'should': [
                    {'term': {'expected_attendees': {'value': 15}}},
                    {
                        'range': {
                            'time_frame': {
                                'gte': '2023-02-05',
                                'lt': '2023-02-10',
                                'relation': 'contains',
                            }
                        }
                    },
                ]
            }
        },
    }
    docs, _ = store.execute_query(query)
    assert [doc['id'] for doc in docs] == [doc[0].id, doc[1].id]
