from dataclasses import dataclass, field
from typing import Any, Dict, Type

import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDocument, DocumentArray
from docarray.storage.abstract_doc_store import BaseDocumentIndex
from docarray.typing import ID, NdArray


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
    __delitem__ = _identity
    __getitem__ = _identity
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
    assert list(store._columns.keys()) == ['id', 'tens']

    assert store._columns['id'].docarray_type == ID
    assert store._columns['id'].db_type == str
    assert store._columns['id'].n_dim is None
    assert store._columns['id'].config == {'hi': 'there'}

    assert issubclass(store._columns['tens'].docarray_type, NdArray)
    assert store._columns['tens'].db_type == str
    assert store._columns['tens'].n_dim == 10
    assert store._columns['tens'].config == {'dim': 1000, 'hi': 'there'}

    # Flat doc
    store = DummyDocIndex[FlatDoc]()
    assert list(store._columns.keys()) == ['id', 'tens_one', 'tens_two']

    assert store._columns['id'].docarray_type == ID
    assert store._columns['id'].db_type == str
    assert store._columns['id'].n_dim is None
    assert store._columns['id'].config == {'hi': 'there'}

    assert issubclass(store._columns['tens_one'].docarray_type, NdArray)
    assert store._columns['tens_one'].db_type == str
    assert store._columns['tens_one'].n_dim is None
    assert store._columns['tens_one'].config == {'dim': 10, 'hi': 'there'}

    assert issubclass(store._columns['tens_two'].docarray_type, NdArray)
    assert store._columns['tens_two'].db_type == str
    assert store._columns['tens_two'].n_dim is None
    assert store._columns['tens_two'].config == {'dim': 50, 'hi': 'there'}

    # Nested doc
    store = DummyDocIndex[NestedDoc]()
    assert list(store._columns.keys()) == ['id', 'd__id', 'd__tens']

    assert store._columns['id'].docarray_type == ID
    assert store._columns['id'].db_type == str
    assert store._columns['id'].n_dim is None
    assert store._columns['id'].config == {'hi': 'there'}

    assert issubclass(store._columns['d__tens'].docarray_type, NdArray)
    assert store._columns['d__tens'].db_type == str
    assert store._columns['d__tens'].n_dim == 10
    assert store._columns['d__tens'].config == {'dim': 1000, 'hi': 'there'}


def test_is_schema_compatible():
    class OtherSimpleDoc(SimpleDoc):
        ...

    class OtherFlatDoc(FlatDoc):
        ...

    class OtherNestedDoc(NestedDoc):
        ...

    store = DummyDocIndex[SimpleDoc]()
    assert store._is_schema_compatible([SimpleDoc(tens=np.random.random((10,)))])
    assert store._is_schema_compatible(
        DocumentArray[SimpleDoc]([SimpleDoc(tens=np.random.random((10,)))])
    )
    assert store._is_schema_compatible([OtherSimpleDoc(tens=np.random.random((10,)))])
    assert store._is_schema_compatible(
        DocumentArray[OtherSimpleDoc]([OtherSimpleDoc(tens=np.random.random((10,)))])
    )
    assert not store._is_schema_compatible(
        [FlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))]
    )
    assert not store._is_schema_compatible(
        DocumentArray[FlatDoc](
            [
                FlatDoc(
                    tens_one=np.random.random((10,)), tens_two=np.random.random((50,))
                )
            ]
        )
    )

    store = DummyDocIndex[FlatDoc]()
    assert store._is_schema_compatible(
        [FlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))]
    )
    assert store._is_schema_compatible(
        DocumentArray[FlatDoc](
            [
                FlatDoc(
                    tens_one=np.random.random((10,)), tens_two=np.random.random((50,))
                )
            ]
        )
    )
    assert store._is_schema_compatible(
        [
            OtherFlatDoc(
                tens_one=np.random.random((10,)), tens_two=np.random.random((50,))
            )
        ]
    )
    assert store._is_schema_compatible(
        DocumentArray[OtherFlatDoc](
            [
                OtherFlatDoc(
                    tens_one=np.random.random((10,)), tens_two=np.random.random((50,))
                )
            ]
        )
    )
    assert not store._is_schema_compatible([SimpleDoc(tens=np.random.random((10,)))])
    assert not store._is_schema_compatible(
        DocumentArray[SimpleDoc]([SimpleDoc(tens=np.random.random((10,)))])
    )

    store = DummyDocIndex[NestedDoc]()
    assert store._is_schema_compatible(
        [NestedDoc(d=SimpleDoc(tens=np.random.random((10,))))]
    )
    assert store._is_schema_compatible(
        DocumentArray[NestedDoc]([NestedDoc(d=SimpleDoc(tens=np.random.random((10,))))])
    )
    assert store._is_schema_compatible(
        [OtherNestedDoc(d=OtherSimpleDoc(tens=np.random.random((10,))))]
    )
    assert store._is_schema_compatible(
        DocumentArray[OtherNestedDoc](
            [OtherNestedDoc(d=OtherSimpleDoc(tens=np.random.random((10,))))]
        )
    )
    assert not store._is_schema_compatible([SimpleDoc(tens=np.random.random((10,)))])
    assert not store._is_schema_compatible(
        DocumentArray[SimpleDoc]([SimpleDoc(tens=np.random.random((10,)))])
    )


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
