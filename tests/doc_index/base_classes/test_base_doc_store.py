from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDocument, DocumentArray
from docarray.doc_index.abstract_doc_index import BaseDocumentIndex
from docarray.typing import ID, NdArray

pytestmark = pytest.mark.doc_index


class SimpleDoc(BaseDocument):
    tens: NdArray[10] = Field(dim=1000)


class FlatDoc(BaseDocument):
    tens_one: NdArray = Field(dim=10)
    tens_two: NdArray = Field(dim=50)


class NestedDoc(BaseDocument):
    d: SimpleDoc


class DeepNestedDoc(BaseDocument):
    d: NestedDoc


class FakeQueryBuilder:
    ...


def _identity(*x, **y):
    return x, y


class DummyDocIndex(BaseDocumentIndex):
    @dataclass
    class RuntimeConfig(BaseDocumentIndex.RuntimeConfig):
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: {str: {'hi': 'there'}, np.ndarray: {'you': 'good?'}}
        )

    @dataclass
    class DBConfig(BaseDocumentIndex.DBConfig):
        work_dir: str = '.'

    class QueryBuilder(BaseDocumentIndex.QueryBuilder):
        def build(self):
            return self._queries

    def python_type_to_db_type(self, x):
        return str

    _index = _identity
    num_docs = _identity
    _del_items = _identity
    _get_items = _identity
    execute_query = _identity
    _find = _identity
    _find_batched = _identity
    _filter = _identity
    _filter_batched = _identity
    _text_search = _identity
    _text_search_batched = _identity


def test_parametrization():
    with pytest.raises(ValueError):
        DummyDocIndex()

    store = DummyDocIndex[SimpleDoc]()
    assert store._schema is SimpleDoc


def test_build_query():
    store = DummyDocIndex[SimpleDoc]()
    q = store.build_query()
    assert isinstance(q, store.QueryBuilder)


def test_create_columns():
    # Simple doc
    store = DummyDocIndex[SimpleDoc]()
    assert list(store._column_infos.keys()) == ['id', 'tens']

    assert store._column_infos['id'].docarray_type == ID
    assert store._column_infos['id'].db_type == str
    assert store._column_infos['id'].n_dim is None
    assert store._column_infos['id'].config == {'hi': 'there'}

    assert issubclass(store._column_infos['tens'].docarray_type, NdArray)
    assert store._column_infos['tens'].db_type == str
    assert store._column_infos['tens'].n_dim == 10
    assert store._column_infos['tens'].config == {'dim': 1000, 'hi': 'there'}

    # Flat doc
    store = DummyDocIndex[FlatDoc]()
    assert list(store._column_infos.keys()) == ['id', 'tens_one', 'tens_two']

    assert store._column_infos['id'].docarray_type == ID
    assert store._column_infos['id'].db_type == str
    assert store._column_infos['id'].n_dim is None
    assert store._column_infos['id'].config == {'hi': 'there'}

    assert issubclass(store._column_infos['tens_one'].docarray_type, NdArray)
    assert store._column_infos['tens_one'].db_type == str
    assert store._column_infos['tens_one'].n_dim is None
    assert store._column_infos['tens_one'].config == {'dim': 10, 'hi': 'there'}

    assert issubclass(store._column_infos['tens_two'].docarray_type, NdArray)
    assert store._column_infos['tens_two'].db_type == str
    assert store._column_infos['tens_two'].n_dim is None
    assert store._column_infos['tens_two'].config == {'dim': 50, 'hi': 'there'}

    # Nested doc
    store = DummyDocIndex[NestedDoc]()
    assert list(store._column_infos.keys()) == ['id', 'd__id', 'd__tens']

    assert store._column_infos['id'].docarray_type == ID
    assert store._column_infos['id'].db_type == str
    assert store._column_infos['id'].n_dim is None
    assert store._column_infos['id'].config == {'hi': 'there'}

    assert issubclass(store._column_infos['d__tens'].docarray_type, NdArray)
    assert store._column_infos['d__tens'].db_type == str
    assert store._column_infos['d__tens'].n_dim == 10
    assert store._column_infos['d__tens'].config == {'dim': 1000, 'hi': 'there'}


def test_docs_validation():
    class OtherSimpleDoc(SimpleDoc):
        ...

    class OtherFlatDoc(FlatDoc):
        ...

    class OtherNestedDoc(NestedDoc):
        ...

    # SIMPLE
    store = DummyDocIndex[SimpleDoc]()
    in_list = [SimpleDoc(tens=np.random.random((10,)))]
    assert isinstance(store._validate_docs(in_list), DocumentArray[BaseDocument])
    in_da = DocumentArray[SimpleDoc](in_list)
    assert store._validate_docs(in_da) == in_da
    in_other_list = [OtherSimpleDoc(tens=np.random.random((10,)))]
    assert isinstance(store._validate_docs(in_other_list), DocumentArray[BaseDocument])
    in_other_da = DocumentArray[OtherSimpleDoc](in_other_list)
    assert store._validate_docs(in_other_da) == in_other_da

    with pytest.raises(ValueError):
        store._validate_docs(
            [
                FlatDoc(
                    tens_one=np.random.random((10,)), tens_two=np.random.random((50,))
                )
            ]
        )
    with pytest.raises(ValueError):
        store._validate_docs(
            DocumentArray[FlatDoc](
                [
                    FlatDoc(
                        tens_one=np.random.random((10,)),
                        tens_two=np.random.random((50,)),
                    )
                ]
            )
        )

    # FLAT
    store = DummyDocIndex[FlatDoc]()
    in_list = [
        FlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))
    ]
    assert isinstance(store._validate_docs(in_list), DocumentArray[BaseDocument])
    in_da = DocumentArray[FlatDoc](
        [FlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))]
    )
    assert store._validate_docs(in_da) == in_da
    in_other_list = [
        OtherFlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))
    ]
    assert isinstance(store._validate_docs(in_other_list), DocumentArray[BaseDocument])
    in_other_da = DocumentArray[OtherFlatDoc](
        [
            OtherFlatDoc(
                tens_one=np.random.random((10,)), tens_two=np.random.random((50,))
            )
        ]
    )
    assert store._validate_docs(in_other_da) == in_other_da
    with pytest.raises(ValueError):
        store._validate_docs([SimpleDoc(tens=np.random.random((10,)))])
    with pytest.raises(ValueError):
        assert not store._validate_docs(
            DocumentArray[SimpleDoc]([SimpleDoc(tens=np.random.random((10,)))])
        )

    # NESTED
    store = DummyDocIndex[NestedDoc]()
    in_list = [NestedDoc(d=SimpleDoc(tens=np.random.random((10,))))]
    assert isinstance(store._validate_docs(in_list), DocumentArray[BaseDocument])
    in_da = DocumentArray[NestedDoc](
        [NestedDoc(d=SimpleDoc(tens=np.random.random((10,))))]
    )
    assert store._validate_docs(in_da) == in_da
    in_other_list = [OtherNestedDoc(d=OtherSimpleDoc(tens=np.random.random((10,))))]
    assert isinstance(store._validate_docs(in_other_list), DocumentArray[BaseDocument])
    in_other_da = DocumentArray[OtherNestedDoc](
        [OtherNestedDoc(d=OtherSimpleDoc(tens=np.random.random((10,))))]
    )

    assert store._validate_docs(in_other_da) == in_other_da
    with pytest.raises(ValueError):
        store._validate_docs([SimpleDoc(tens=np.random.random((10,)))])
    with pytest.raises(ValueError):
        store._validate_docs(
            DocumentArray[SimpleDoc]([SimpleDoc(tens=np.random.random((10,)))])
        )


def test_docs_validation_unions():
    class OptionalDoc(BaseDocument):
        tens: Optional[NdArray[10]] = Field(dim=1000)

    class UnionDoc(BaseDocument):
        tens: Union[NdArray[10], str] = Field(dim=1000)

    # OPTIONAL
    store = DummyDocIndex[SimpleDoc]()
    in_list = [OptionalDoc(tens=np.random.random((10,)))]
    assert isinstance(store._validate_docs(in_list), DocumentArray[BaseDocument])
    in_da = DocumentArray[OptionalDoc](in_list)
    assert store._validate_docs(in_da) == in_da

    with pytest.raises(ValueError):
        store._validate_docs([OptionalDoc(tens=None)])

    # OTHER UNION
    store = DummyDocIndex[SimpleDoc]()
    in_list = [UnionDoc(tens=np.random.random((10,)))]
    assert isinstance(store._validate_docs(in_list), DocumentArray[BaseDocument])
    in_da = DocumentArray[UnionDoc](in_list)
    assert isinstance(store._validate_docs(in_da), DocumentArray[BaseDocument])

    with pytest.raises(ValueError):
        store._validate_docs([UnionDoc(tens='hello')])


def test_get_value():
    t = np.random.random((10,))

    doc = SimpleDoc(tens=t)
    assert np.all(DummyDocIndex._get_values_by_column([doc], 'tens')[0] == t)

    doc = FlatDoc(tens_one=t, tens_two=np.random.random((50,)))
    assert np.all(DummyDocIndex._get_values_by_column([doc], 'tens_one')[0] == t)

    doc = NestedDoc(d=SimpleDoc(tens=t))
    assert np.all(DummyDocIndex._get_values_by_column([doc], 'd__tens')[0] == t)

    doc = DeepNestedDoc(d=NestedDoc(d=SimpleDoc(tens=t)))
    assert np.all(DummyDocIndex._get_values_by_column([doc], 'd__d__tens')[0] == t)

    vals = DummyDocIndex._get_values_by_column([doc, doc], 'd__d__tens')
    assert np.all(vals[0] == t)
    assert np.all(vals[1] == t)


def test_get_data_by_columns():
    store = DummyDocIndex[SimpleDoc]()
    docs = [SimpleDoc(tens=np.random.random((10,))) for _ in range(10)]
    data_by_columns = store._get_col_value_dict(docs)
    assert list(data_by_columns.keys()) == ['id', 'tens']
    assert list(data_by_columns['id']) == [doc.id for doc in docs]
    assert list(data_by_columns['tens']) == [doc.tens for doc in docs]

    store = DummyDocIndex[FlatDoc]()
    docs = [
        FlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))
        for _ in range(10)
    ]
    data_by_columns = store._get_col_value_dict(docs)
    assert list(data_by_columns.keys()) == ['id', 'tens_one', 'tens_two']
    assert list(data_by_columns['id']) == [doc.id for doc in docs]
    assert list(data_by_columns['tens_one']) == [doc.tens_one for doc in docs]
    assert list(data_by_columns['tens_two']) == [doc.tens_two for doc in docs]

    store = DummyDocIndex[NestedDoc]()
    docs = [NestedDoc(d=SimpleDoc(tens=np.random.random((10,)))) for _ in range(10)]
    data_by_columns = store._get_col_value_dict(docs)
    assert list(data_by_columns.keys()) == ['id', 'd__id', 'd__tens']
    assert list(data_by_columns['id']) == [doc.id for doc in docs]
    assert list(data_by_columns['d__id']) == [doc.d.id for doc in docs]
    assert list(data_by_columns['d__tens']) == [doc.d.tens for doc in docs]

    store = DummyDocIndex[DeepNestedDoc]()
    docs = [
        DeepNestedDoc(d=NestedDoc(d=SimpleDoc(tens=np.random.random((10,)))))
        for _ in range(10)
    ]
    data_by_columns = store._get_col_value_dict(docs)
    assert list(data_by_columns.keys()) == ['id', 'd__id', 'd__d__id', 'd__d__tens']
    assert list(data_by_columns['id']) == [doc.id for doc in docs]
    assert list(data_by_columns['d__id']) == [doc.d.id for doc in docs]
    assert list(data_by_columns['d__d__id']) == [doc.d.d.id for doc in docs]
    assert list(data_by_columns['d__d__tens']) == [doc.d.d.tens for doc in docs]


def test_transpose_data_by_columns():
    store = DummyDocIndex[SimpleDoc]()
    docs = [SimpleDoc(tens=np.random.random((10,))) for _ in range(10)]
    data_by_columns = store._get_col_value_dict(docs)
    data_by_rows = list(store._transpose_col_value_dict(data_by_columns))
    assert len(data_by_rows) == len(docs)
    for doc, row in zip(docs, data_by_rows):
        assert doc.id == row['id']
        assert np.all(doc.tens == row['tens'])

    store = DummyDocIndex[FlatDoc]()
    docs = [
        FlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))
        for _ in range(10)
    ]
    data_by_columns = store._get_col_value_dict(docs)
    data_by_rows = list(store._transpose_col_value_dict(data_by_columns))
    assert len(data_by_rows) == len(docs)
    for doc, row in zip(docs, data_by_rows):
        assert doc.id == row['id']
        assert np.all(doc.tens_one == row['tens_one'])
        assert np.all(doc.tens_two == row['tens_two'])

    store = DummyDocIndex[NestedDoc]()
    docs = [NestedDoc(d=SimpleDoc(tens=np.random.random((10,)))) for _ in range(10)]
    data_by_columns = store._get_col_value_dict(docs)
    data_by_rows = list(store._transpose_col_value_dict(data_by_columns))
    assert len(data_by_rows) == len(docs)
    for doc, row in zip(docs, data_by_rows):
        assert doc.id == row['id']
        assert doc.d.id == row['d__id']
        assert np.all(doc.d.tens == row['d__tens'])

    store = DummyDocIndex[DeepNestedDoc]()
    docs = [
        DeepNestedDoc(d=NestedDoc(d=SimpleDoc(tens=np.random.random((10,)))))
        for _ in range(10)
    ]
    data_by_columns = store._get_col_value_dict(docs)
    data_by_rows = list(store._transpose_col_value_dict(data_by_columns))
    assert len(data_by_rows) == len(docs)
    for doc, row in zip(docs, data_by_rows):
        assert doc.id == row['id']
        assert doc.d.id == row['d__id']
        assert doc.d.d.id == row['d__d__id']
        assert np.all(doc.d.d.tens == row['d__d__tens'])
