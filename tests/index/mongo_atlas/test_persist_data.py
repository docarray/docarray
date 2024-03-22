from docarray.index import MongoAtlasDocumentIndex

from .fixtures import *  # noqa
from .helpers import assert_when_ready


def create_index(uri, database, collection_name, schema):
    return MongoAtlasDocumentIndex[schema](
        mongo_connection_uri=uri,
        database_name=database,
        collection_name=collection_name,
    )


def test_persist(
    clean_database, mongo_fixture_env, simple_schema, random_simple_documents
):
    index = create_index(*mongo_fixture_env, simple_schema)

    assert index.num_docs() == 0

    index.index(random_simple_documents)

    def pred():
        # check if there are elements in the database and if the index is up to date.
        assert index.num_docs() == len(random_simple_documents)
        assert (
            len(
                index.find(
                    random_simple_documents[0].embedding,
                    search_field='embedding',
                    limit=1,
                ).documents
            )
            > 0
        )

    assert_when_ready(pred)

    doc_before = index.find(
        random_simple_documents[0].embedding, search_field='embedding', limit=1
    ).documents[0]
    del index

    index = create_index(*mongo_fixture_env, simple_schema)
    doc_after = index.find(
        random_simple_documents[0].embedding, search_field='embedding', limit=1
    ).documents[0]

    assert index.num_docs() == len(random_simple_documents)
    assert doc_before.id == doc_after.id
    assert (doc_before.embedding == doc_after.embedding).all()
