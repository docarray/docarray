import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import OpenSearchDocIndex
from tests.index.opensearch.fixture import (  # noqa: F401
    auth,
    start_storage,
    tmp_index_name,
)

pytestmark = [pytest.mark.slow, pytest.mark.index, pytest.mark.opensearchv2]


def test_column_config(tmp_index_name, auth):  # noqa: F811
    class MyDoc(BaseDoc):
        text: str
        color: str = Field(col_type='keyword')

    index = OpenSearchDocIndex[MyDoc](index_name=tmp_index_name, auth=auth)
    index_docs = [
        MyDoc(id='0', text='hello world', color='red'),
        MyDoc(id='1', text='never gonna give you up', color='blue'),
        MyDoc(id='2', text='we are the world', color='green'),
    ]
    index.index(index_docs)

    query = 'world'
    docs, _ = index.text_search(query, search_field='text')
    assert [doc.id for doc in docs] == ['0', '2']

    filter_query = {'terms': {'color': ['red', 'blue']}}
    docs = index.filter(filter_query)
    assert [doc.id for doc in docs] == ['0', '1']


def test_field_object(tmp_index_name, auth):  # noqa: F811
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

    index = OpenSearchDocIndex[MyDoc](index_name=tmp_index_name, auth=auth)
    doc = [
        MyDoc(manager={'age': 25, 'name': {'first': 'Rachel', 'last': 'Green'}}),
        MyDoc(manager={'age': 30, 'name': {'first': 'Monica', 'last': 'Geller'}}),
        MyDoc(manager={'age': 35, 'name': {'first': 'Phoebe', 'last': 'Buffay'}}),
    ]
    index.index(doc)
    id_ = doc[0].id
    assert index[id_].id == id_
    assert index[id_].manager == doc[0].manager

    filter_query = {'range': {'manager.age': {'gte': 30}}}
    docs = index.filter(filter_query)
    assert [doc.id for doc in docs] == [doc[1].id, doc[2].id]


def test_field_geo_point(tmp_index_name, auth):  # noqa: F811
    class MyDoc(BaseDoc):
        location: dict = Field(col_type='geo_point')

    index = OpenSearchDocIndex[MyDoc](index_name=tmp_index_name, auth=auth)
    doc = [
        MyDoc(location={'lat': 40.12, 'lon': -72.34}),
        MyDoc(location={'lat': 41.12, 'lon': -73.34}),
        MyDoc(location={'lat': 42.12, 'lon': -74.34}),
    ]
    index.index(doc)

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

    docs, _ = index.execute_query(query)
    assert [doc['id'] for doc in docs] == [doc[0].id, doc[1].id]


def test_field_range(tmp_index_name, auth):  # noqa: F811
    class MyDoc(BaseDoc):
        expected_attendees: dict = Field(col_type='integer_range')
        time_frame: dict = Field(col_type='date_range', format='yyyy-MM-dd')

    index = OpenSearchDocIndex[MyDoc](index_name=tmp_index_name, auth=auth)
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
    index.index(doc)

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
    docs, _ = index.execute_query(query)
    assert [doc['id'] for doc in docs] == [doc[0].id, doc[1].id]


def test_index_name():
    class TextDoc(BaseDoc):
        text: str = Field()

    class StringDoc(BaseDoc):
        text: str = Field(col_type='text')

    index = OpenSearchDocIndex[TextDoc]()
    assert index.index_name == TextDoc.__name__.lower()

    index = OpenSearchDocIndex[StringDoc]()
    assert index.index_name == StringDoc.__name__.lower()
