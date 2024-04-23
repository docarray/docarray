import os

import numpy as np
import pytest

from docarray.index import MongoAtlasDocumentIndex

from . import NestedDoc, SimpleDoc, SimpleSchema


def mongo_env_var(var: str):
    return os.environ[var]


@pytest.fixture(scope='session')
def mongodb_index_config():
    return {
        "mongo_connection_uri": mongo_env_var("MONGODB_URI"),
        "database_name": mongo_env_var("DATABASE_NAME"),
    }


@pytest.fixture
def simple_index(mongodb_index_config):

    index = MongoAtlasDocumentIndex[SimpleSchema](**mongodb_index_config)
    return index


@pytest.fixture
def nested_index(mongodb_index_config):
    index = MongoAtlasDocumentIndex[NestedDoc](**mongodb_index_config)
    return index


@pytest.fixture(scope='module')
def random_simple_documents():
    N_DIM = 10
    docs_text = [
        "Text processing with Python is a valuable skill for data analysis.",
        "Gardening tips for a beautiful backyard oasis.",
        "Explore the wonders of deep-sea diving in tropical locations.",
        "The history and art of classical music compositions.",
        "An introduction to the world of gourmet cooking.",
        "Integer pharetra, leo quis aliquam hendrerit, arcu ante sagittis massa, nec tincidunt arcu.",
        "Sed luctus convallis velit sit amet laoreet. Morbi sit amet magna pellentesque urna tincidunt",
        "luctus enim interdum lacinia. Morbi maximus diam id justo egestas pellentesque. Suspendisse",
        "id laoreet odio gravida vitae. Vivamus feugiat nisi quis est pellentesque interdum. Integer",
        "eleifend eros non, accumsan lectus. Curabitur porta auctor tellus at pharetra. Phasellus ut condimentum",
    ]
    return [
        SimpleSchema(embedding=np.random.rand(N_DIM), number=i, text=docs_text[i])
        for i in range(10)
    ]


@pytest.fixture
def nested_documents():
    N_DIM = 10
    docs = [
        NestedDoc(
            d=SimpleDoc(embedding=np.random.rand(N_DIM)),
            embedding=np.random.rand(N_DIM),
        )
        for _ in range(10)
    ]
    docs.append(
        NestedDoc(
            d=SimpleDoc(embedding=np.zeros(N_DIM)),
            embedding=np.ones(N_DIM),
        )
    )
    docs.append(
        NestedDoc(
            d=SimpleDoc(embedding=np.ones(N_DIM)),
            embedding=np.zeros(N_DIM),
        )
    )
    docs.append(
        NestedDoc(
            d=SimpleDoc(embedding=np.zeros(N_DIM)),
            embedding=np.ones(N_DIM),
        )
    )
    return docs


@pytest.fixture
def simple_index_with_docs(simple_index, random_simple_documents):
    """
    Setup and teardown of simple_index. Accesses the underlying MongoDB collection directly.
    """
    simple_index._doc_collection.delete_many({})
    simple_index.index(random_simple_documents)
    yield simple_index, random_simple_documents
    simple_index._doc_collection.delete_many({})


@pytest.fixture
def nested_index_with_docs(nested_index, nested_documents):
    """
    Setup and teardown of simple_index. Accesses the underlying MongoDB collection directly.
    """
    nested_index._doc_collection.delete_many({})
    nested_index.index(nested_documents)
    yield nested_index, nested_documents
    nested_index._doc_collection.delete_many({})
