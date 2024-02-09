// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
                                                 // "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

    index = ElasticV7DocIndex[MyDoc]()
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

    index = ElasticV7DocIndex[MyDoc]()
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


def test_field_geo_point():
    class MyDoc(BaseDoc):
        location: dict = Field(col_type='geo_point')

    index = ElasticV7DocIndex[MyDoc]()
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


def test_field_range():
    class MyDoc(BaseDoc):
        expected_attendees: dict = Field(col_type='integer_range')
        time_frame: dict = Field(col_type='date_range', format='yyyy-MM-dd')

    index = ElasticV7DocIndex[MyDoc]()
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

    index = ElasticV7DocIndex[TextDoc]()
    assert index.index_name == TextDoc.__name__.lower()

    index = ElasticV7DocIndex[StringDoc]()
    assert index.index_name == StringDoc.__name__.lower()
