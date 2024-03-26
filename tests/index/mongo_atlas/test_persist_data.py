from docarray.index import MongoAtlasDocumentIndex

from .fixtures import (  # noqa: F401
    mongo_fixture_env,
    random_simple_documents,
    simple_schema,
)
from .helpers import assert_when_ready


def create_index(uri, database, schema):
    return MongoAtlasDocumentIndex[schema](
        mongo_connection_uri=uri,
        database_name=database,
    )


def test_persist(
    mongo_fixture_env, simple_schema, random_simple_documents  # noqa: F811
):
    index = create_index(*mongo_fixture_env, simple_schema)
    index._doc_collection.delete_many({})

    def cleaned_database():
        assert index.num_docs() == 0

    assert_when_ready(cleaned_database)

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
