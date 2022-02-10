from typing import List

import numpy as np
import pytest
import strawberry

from docarray import DocumentArray
from docarray.document.strawberry_type import StrawberryDocument
from tests import random_docs


@pytest.fixture
def simple_match_da():
    da = DocumentArray.empty(10)
    da.embeddings = np.random.random([10, 15])
    db = DocumentArray.empty(10)
    db.embeddings = np.random.random([10, 15])
    da.match(db)
    yield da


def test_to_from_strawberry_type(simple_match_da: DocumentArray):
    DocumentArray.from_strawberry_type(simple_match_da.to_strawberry_type())


def test_query_simple_match(simple_match_da):
    @strawberry.type
    class Query:
        docs: List[StrawberryDocument] = strawberry.field(
            resolver=lambda: simple_match_da.to_strawberry_type()
        )

    schema = strawberry.Schema(query=Query)

    r = schema.execute_sync(
        '''
{
    docs {
        matches {
            id
            scores  {
                score {
                    value
                }
            }
        }
    }
}
'''
    ).data
    print(r)
    assert r
    assert len(r['docs']) == 10
    assert len(r['docs'][0]['matches']) == 10
    assert r['docs'][0]['matches'][0]['scores'][0]['score']['value']


def test_query_random_docs():
    @strawberry.type
    class Query:
        docs: List[StrawberryDocument] = strawberry.field(
            resolver=lambda: random_docs(10).to_strawberry_type()
        )

    schema = strawberry.Schema(query=Query)

    r = schema.execute_sync(
        '''
    {
        docs {
            id
            chunks {
                id
            }
        }
    }
    '''
    ).data
    print(r)
    assert r
    assert len(r['docs']) == 10
