from docarray import DocumentArray, Document


def test_data_type(start_storage):
    elastic_doc = DocumentArray(
        storage='elasticsearch',
        config={
            'n_dim': 3,
            'columns': [
                ('is_true', 'bool'),
                ('test_long', 'long'),
                ('test_double', 'double'),
            ],
            'distance': 'l2_norm',
            'index_name': 'test_data_type',
        },
    )

    with elastic_doc:
        elastic_doc.extend(
            [
                Document(
                    id=1,
                    test_bool=True,
                    test_long=372_036_854_775_807,
                    test_double=1_000_000_000_000_000,
                )
            ]
        )

    assert elastic_doc[0].tags['test_bool'] is True
    assert elastic_doc[0].tags['test_long'] == 372_036_854_775_807
    assert elastic_doc[0].tags['test_double'] == 1_000_000_000_000_000
