import logging
import os

import numpy as np
import pytest

from docarray.index import MongoDBAtlasDocumentIndex

from . import NestedDoc, SimpleDoc, SimpleSchema


@pytest.fixture(scope='session')
def mongodb_index_config():
    return {
        "mongo_connection_uri": os.environ["MONGODB_URI"],
        "database_name": os.environ["MONGODB_DATABASE"],
    }


@pytest.fixture
def simple_index(mongodb_index_config):

    index = MongoDBAtlasDocumentIndex[SimpleSchema](
        index_name="bespoke_name", **mongodb_index_config
    )
    return index


@pytest.fixture
def nested_index(mongodb_index_config):
    index = MongoDBAtlasDocumentIndex[NestedDoc](**mongodb_index_config)
    return index


@pytest.fixture(scope='module')
def n_dim():
    return 10


@pytest.fixture(scope='module')
def embeddings(n_dim):
    """A consistent, reasonable, mock of vector embeddings, in [-1, 1]."""
    x = np.linspace(-np.pi, np.pi, n_dim)
    y = np.arange(n_dim)
    return np.sin(x[np.newaxis, :] + y[:, np.newaxis])


@pytest.fixture(scope='module')
def random_simple_documents(n_dim, embeddings):
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
        SimpleSchema(embedding=embeddings[i], number=i, text=docs_text[i])
        for i in range(len(docs_text))
    ]


@pytest.fixture
def nested_documents(n_dim):
    docs = [
        NestedDoc(
            d=SimpleDoc(embedding=np.random.rand(n_dim)),
            embedding=np.random.rand(n_dim),
        )
        for _ in range(10)
    ]
    docs.append(
        NestedDoc(
            d=SimpleDoc(embedding=np.zeros(n_dim)),
            embedding=np.ones(n_dim),
        )
    )
    docs.append(
        NestedDoc(
            d=SimpleDoc(embedding=np.ones(n_dim)),
            embedding=np.zeros(n_dim),
        )
    )
    docs.append(
        NestedDoc(
            d=SimpleDoc(embedding=np.zeros(n_dim)),
            embedding=np.ones(n_dim),
        )
    )
    return docs


@pytest.fixture
def simple_index_with_docs(simple_index, random_simple_documents):
    """
    Setup and teardown of simple_index. Accesses the underlying MongoDB collection directly.
    """
    simple_index._collection.delete_many({})
    simple_index._logger.setLevel(logging.DEBUG)
    simple_index.index(random_simple_documents)
    yield simple_index, random_simple_documents
    simple_index._collection.delete_many({})


@pytest.fixture
def nested_index_with_docs(nested_index, nested_documents):
    """
    Setup and teardown of simple_index. Accesses the underlying MongoDB collection directly.
    """
    nested_index._collection.delete_many({})
    nested_index.index(nested_documents)
    yield nested_index, nested_documents
    nested_index._collection.delete_many({})
