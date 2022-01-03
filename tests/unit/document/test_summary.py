from docarray import Document


def test_single_doc_summary():
    # empty doc
    Document().summary()
    # nested doc
    Document(
        chunks=[
            Document(),
            Document(chunks=[Document()]),
            Document(),
        ],
        matches=[Document(), Document()],
    ).summary()
