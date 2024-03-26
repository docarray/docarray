from .fixtures import simple_index_with_docs  # noqa: F401


def test_filter(simple_index_with_docs):  # noqa: F811

    db, base_docs = simple_index_with_docs

    docs = db.filter(filter_query={"number": {"$lt": 1}})
    assert len(docs) == 1
    assert docs[0].number == 0

    docs = db.filter(filter_query={"number": {"$gt": 8}})
    assert len(docs) == 1
    assert docs[0].number == 9

    docs = db.filter(filter_query={"number": {"$lt": 8, "$gt": 3}})
    assert len(docs) == 4

    docs = db.filter(filter_query={"text": {"$regex": "introduction"}})
    assert len(docs) == 1
    assert 'introduction' in docs[0].text.lower()

    docs = db.filter(filter_query={"text": {"$not": {"$regex": "Explore"}}})
    assert len(docs) == 9
    assert all("Explore" not in doc.text for doc in docs)
