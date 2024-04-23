from docarray.index import MongoAtlasDocumentIndex

from . import SimpleSchema, assert_when_ready


def test_persist(mongodb_index_config, random_simple_documents):  # noqa: F811
    index = MongoAtlasDocumentIndex[SimpleSchema](**mongodb_index_config)
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

    index = MongoAtlasDocumentIndex[SimpleSchema](**mongodb_index_config)

    doc_after = index.find(
        random_simple_documents[0].embedding, search_field='embedding', limit=1
    ).documents[0]

    assert index.num_docs() == len(random_simple_documents)
    assert doc_before.id == doc_after.id
    assert (doc_before.embedding == doc_after.embedding).all()
