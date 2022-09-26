from docarray import Document, DocumentArray


def test_get_additional(start_storage):
    da = DocumentArray(
        storage='weaviate', config={'n_dim': 3}
    )

    with da:
        da.extend(
            [
                Document(embedding=[0, 0, 0]),
                Document(embedding=[2, 2, 2]),
                Document(embedding=[4, 4, 4]),
                Document(embedding=[2, 2, 2]),
                Document(embedding=[4, 4, 4]),
            ]
        )

    additional = ["creationTimeUnix", "lastUpdateTimeUnix"]
    results = da.find(
        DocumentArray([Document(embedding=[2, 2, 2])]),
        limit=1,
        additional=additional,
    )

    for res in results:
        assert res[:, "tags__creationTimeUnix"][0] is not None
        assert res[:, "tags__lastUpdateTimeUnix"][0] is not None
