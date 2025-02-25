import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.array.any_array import AnyDocArray
from docarray.documents import ImageDoc
from docarray.index.abstract import BaseDocIndex, _raise_not_composable
from docarray.typing import ID, ImageBytes, ImageUrl, NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import torch_imported
from docarray.utils._internal._typing import safe_issubclass

pytestmark = pytest.mark.index


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000)


class FlatDoc(BaseDoc):
    tens_one: NdArray = Field(dim=10)
    tens_two: NdArray = Field(dim=50)


class NestedDoc(BaseDoc):
    d: SimpleDoc


class DeepNestedDoc(BaseDoc):
    d: NestedDoc


class SubindexDoc(BaseDoc):
    d: DocList[SimpleDoc]


class SubSubindexDoc(BaseDoc):
    d_root: DocList[SubindexDoc]


class FakeQueryBuilder:
    ...


def _identity(*x, **y):
    return x, y


class DummyDocIndex(BaseDocIndex):
    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)
        for col_name, col in self._column_infos.items():
            if safe_issubclass(col.docarray_type, AnyDocArray):
                sub_db_config = copy.deepcopy(self._db_config)
                self._subindices[col_name] = self.__class__[col.docarray_type.doc_type](
                    db_config=sub_db_config, subindex=True
                )

    @property
    def index_name(self):
        return 'dummy'

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        pass

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        work_dir: str = '.'
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: {
                str: {'hi': 'there'},
                np.ndarray: {'you': 'good?'},
                'varchar': {'good': 'bye'},
                AbstractTensor: {'dim': 1000},
            }
        )

    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def build(self):
            return None

        find = _raise_not_composable('find')
        filter = _raise_not_composable('filter')
        text_search = _raise_not_composable('text_search')
        find_batched = _raise_not_composable('find_batched')
        filter_batched = _raise_not_composable('find_batched')
        text_search_batched = _raise_not_composable('text_search')

    def python_type_to_db_type(self, x):
        return str

    def num_docs(self):
        return 3

    def _doc_exists(self, doc_id: str) -> bool:
        return False

    _index = _identity
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

    index = DummyDocIndex[SimpleDoc]()
    assert index._schema is SimpleDoc

    index = DummyDocIndex[SubindexDoc]()
    assert index._schema is SubindexDoc
    assert list(index._subindices['d']._schema._docarray_fields().keys()) == [
        'id',
        'tens',
        'parent_id',
    ]

    index = DummyDocIndex[SubSubindexDoc]()
    assert index._schema is SubSubindexDoc
    assert list(index._subindices['d_root']._schema._docarray_fields().keys()) == [
        'id',
        'd',
        'parent_id',
    ]
    assert list(
        index._subindices['d_root']._subindices['d']._schema._docarray_fields().keys()
    ) == [
        'id',
        'tens',
        'parent_id',
    ]


def test_build_query():
    index = DummyDocIndex[SimpleDoc]()
    q = index.build_query()
    assert isinstance(q, index.QueryBuilder)


def test_create_columns():
    # Simple doc
    index = DummyDocIndex[SimpleDoc]()
    assert list(index._column_infos.keys()) == ['id', 'tens']

    assert index._column_infos['id'].docarray_type == ID
    assert index._column_infos['id'].db_type == str
    assert index._column_infos['id'].n_dim is None
    assert index._column_infos['id'].config['hi'] == 'there'

    assert safe_issubclass(index._column_infos['tens'].docarray_type, AbstractTensor)
    assert index._column_infos['tens'].db_type == str
    assert index._column_infos['tens'].n_dim == 10
    assert index._column_infos['tens'].config == {'dim': 1000, 'hi': 'there'}

    # Flat doc
    index = DummyDocIndex[FlatDoc]()
    assert list(index._column_infos.keys()) == ['id', 'tens_one', 'tens_two']

    assert index._column_infos['id'].docarray_type == ID
    assert index._column_infos['id'].db_type == str
    assert index._column_infos['id'].n_dim is None
    assert index._column_infos['id'].config['hi'] == 'there'

    assert safe_issubclass(
        index._column_infos['tens_one'].docarray_type, AbstractTensor
    )
    assert index._column_infos['tens_one'].db_type == str
    assert index._column_infos['tens_one'].n_dim is None
    assert index._column_infos['tens_one'].config == {'dim': 10, 'hi': 'there'}

    assert safe_issubclass(
        index._column_infos['tens_two'].docarray_type, AbstractTensor
    )
    assert index._column_infos['tens_two'].db_type == str
    assert index._column_infos['tens_two'].n_dim is None
    assert index._column_infos['tens_two'].config == {'dim': 50, 'hi': 'there'}

    # Nested doc
    index = DummyDocIndex[NestedDoc]()
    assert list(index._column_infos.keys()) == ['id', 'd__id', 'd__tens']

    assert index._column_infos['id'].docarray_type == ID
    assert index._column_infos['id'].db_type == str
    assert index._column_infos['id'].n_dim is None
    assert index._column_infos['id'].config['hi'] == 'there'

    assert safe_issubclass(index._column_infos['d__tens'].docarray_type, AbstractTensor)
    assert index._column_infos['d__tens'].db_type == str
    assert index._column_infos['d__tens'].n_dim == 10
    assert index._column_infos['d__tens'].config == {'dim': 1000, 'hi': 'there'}

    # Subindex doc
    index = DummyDocIndex[SubindexDoc]()
    assert list(index._column_infos.keys()) == ['id', 'd']
    assert list(index._subindices['d']._column_infos.keys()) == [
        'id',
        'tens',
        'parent_id',
    ]

    assert safe_issubclass(index._column_infos['d'].docarray_type, AnyDocArray)
    assert index._column_infos['d'].db_type is None
    assert index._column_infos['d'].n_dim is None
    assert index._column_infos['d'].config == {}

    assert index._subindices['d']._column_infos['id'].docarray_type == ID
    assert index._subindices['d']._column_infos['id'].db_type == str
    assert index._subindices['d']._column_infos['id'].n_dim is None
    assert index._subindices['d']._column_infos['id'].config['hi'] == 'there'

    assert safe_issubclass(
        index._subindices['d']._column_infos['tens'].docarray_type, AbstractTensor
    )
    assert index._subindices['d']._column_infos['tens'].db_type == str
    assert index._subindices['d']._column_infos['tens'].n_dim == 10
    assert index._subindices['d']._column_infos['tens'].config == {
        'dim': 1000,
        'hi': 'there',
    }

    assert index._subindices['d']._column_infos['parent_id'].docarray_type == ID
    assert index._subindices['d']._column_infos['parent_id'].db_type == str
    assert index._subindices['d']._column_infos['parent_id'].n_dim is None
    assert index._subindices['d']._column_infos['parent_id'].config == {'hi': 'there'}

    # SubSubindex doc
    index = DummyDocIndex[SubSubindexDoc]()
    assert list(index._column_infos.keys()) == ['id', 'd_root']
    assert list(index._subindices['d_root']._column_infos.keys()) == [
        'id',
        'd',
        'parent_id',
    ]
    assert list(index._subindices['d_root']._subindices['d']._column_infos.keys()) == [
        'id',
        'tens',
        'parent_id',
    ]

    assert safe_issubclass(
        index._subindices['d_root']._column_infos['d'].docarray_type, AnyDocArray
    )
    assert index._subindices['d_root']._column_infos['d'].db_type is None
    assert index._subindices['d_root']._column_infos['d'].n_dim is None
    assert index._subindices['d_root']._column_infos['d'].config == {}

    assert (
        index._subindices['d_root']._subindices['d']._column_infos['id'].docarray_type
        == ID
    )
    assert (
        index._subindices['d_root']._subindices['d']._column_infos['id'].db_type == str
    )
    assert (
        index._subindices['d_root']._subindices['d']._column_infos['id'].n_dim is None
    )
    assert (
        index._subindices['d_root']._subindices['d']._column_infos['id'].config['hi']
        == 'there'
    )
    assert safe_issubclass(
        index._subindices['d_root']
        ._subindices['d']
        ._column_infos['tens']
        .docarray_type,
        AbstractTensor,
    )
    assert (
        index._subindices['d_root']._subindices['d']._column_infos['tens'].db_type
        == str
    )
    assert (
        index._subindices['d_root']._subindices['d']._column_infos['tens'].n_dim == 10
    )
    assert index._subindices['d_root']._subindices['d']._column_infos[
        'tens'
    ].config == {
        'dim': 1000,
        'hi': 'there',
    }

    assert (
        index._subindices['d_root']
        ._subindices['d']
        ._column_infos['parent_id']
        .docarray_type
        == ID
    )
    assert (
        index._subindices['d_root']._subindices['d']._column_infos['parent_id'].db_type
        == str
    )
    assert (
        index._subindices['d_root']._subindices['d']._column_infos['parent_id'].n_dim
        is None
    )
    assert index._subindices['d_root']._subindices['d']._column_infos[
        'parent_id'
    ].config == {'hi': 'there'}


def test_flatten_schema():
    index = DummyDocIndex[SimpleDoc]()
    fields = SimpleDoc._docarray_fields()
    assert set(index._flatten_schema(SimpleDoc)) == {
        ('id', ID, fields['id']),
        ('tens', AbstractTensor, fields['tens']),
    }

    index = DummyDocIndex[FlatDoc]()
    fields = FlatDoc._docarray_fields()
    assert set(index._flatten_schema(FlatDoc)) == {
        ('id', ID, fields['id']),
        ('tens_one', AbstractTensor, fields['tens_one']),
        ('tens_two', AbstractTensor, fields['tens_two']),
    }

    index = DummyDocIndex[NestedDoc]()
    fields = NestedDoc._docarray_fields()
    fields_nested = SimpleDoc._docarray_fields()
    assert set(index._flatten_schema(NestedDoc)) == {
        ('id', ID, fields['id']),
        ('d__id', ID, fields_nested['id']),
        ('d__tens', AbstractTensor, fields_nested['tens']),
    }

    index = DummyDocIndex[DeepNestedDoc]()
    fields = DeepNestedDoc._docarray_fields()
    fields_nested = NestedDoc._docarray_fields()
    fields_nested_nested = SimpleDoc._docarray_fields()
    assert set(index._flatten_schema(DeepNestedDoc)) == {
        ('id', ID, fields['id']),
        ('d__id', ID, fields_nested['id']),
        ('d__d__id', ID, fields_nested_nested['id']),
        ('d__d__tens', AbstractTensor, fields_nested_nested['tens']),
    }

    index = DummyDocIndex[SubindexDoc]()
    fields = SubindexDoc._docarray_fields()
    assert set(index._flatten_schema(SubindexDoc)) == {
        ('id', ID, fields['id']),
        ('d', DocList[SimpleDoc], fields['d']),
    }
    assert [
        field_name
        for field_name, _, _ in index._subindices['d']._flatten_schema(
            index._subindices['d']._schema
        )
    ] == ['id', 'tens', 'parent_id']
    assert [
        type_
        for _, type_, _ in index._subindices['d']._flatten_schema(
            index._subindices['d']._schema
        )
    ] == [ID, AbstractTensor, ID]

    index = DummyDocIndex[SubSubindexDoc]()
    fields = SubSubindexDoc._docarray_fields()
    assert set(index._flatten_schema(SubSubindexDoc)) == {
        ('id', ID, fields['id']),
        ('d_root', DocList[SubindexDoc], fields['d_root']),
    }
    assert [
        field_name
        for field_name, _, _ in index._subindices['d_root']
        ._subindices['d']
        ._flatten_schema(index._subindices['d_root']._subindices['d']._schema)
    ] == ['id', 'tens', 'parent_id']
    assert [
        type_
        for _, type_, _ in index._subindices['d_root']
        ._subindices['d']
        ._flatten_schema(index._subindices['d_root']._subindices['d']._schema)
    ] == [ID, AbstractTensor, ID]


def test_flatten_schema_union():
    class MyDoc(BaseDoc):
        image: ImageDoc

    index = DummyDocIndex[MyDoc]()
    fields = MyDoc._docarray_fields()
    fields_image = ImageDoc._docarray_fields()

    if torch_imported:
        from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor

    assert set(index._flatten_schema(MyDoc)) == {
        ('id', ID, fields['id']),
        ('image__id', ID, fields_image['id']),
        ('image__url', ImageUrl, fields_image['url']),
        ('image__tensor', AbstractTensor, fields_image['tensor']),
        ('image__embedding', AbstractTensor, fields_image['embedding']),
        ('image__bytes_', ImageBytes, fields_image['bytes_']),
    }

    class MyDoc2(BaseDoc):
        tensor: Union[NdArray, str]

    with pytest.raises(ValueError):
        _ = DummyDocIndex[MyDoc2]()

    class MyDoc3(BaseDoc):
        tensor: Union[NdArray, ImageTorchTensor]

    index = DummyDocIndex[MyDoc3]()
    fields = MyDoc3._docarray_fields()
    assert set(index._flatten_schema(MyDoc3)) == {
        ('id', ID, fields['id']),
        ('tensor', AbstractTensor, fields['tensor']),
    }


def test_columns_db_type_with_user_defined_mapping(tmp_path):
    class MyDoc(BaseDoc):
        tens: NdArray[10] = Field(dim=1000, col_type=np.ndarray)

    index = DummyDocIndex[MyDoc](work_dir=str(tmp_path))

    assert index._column_infos['tens'].db_type == np.ndarray


def test_columns_db_type_with_user_defined_mapping_additional_params(tmp_path):
    class MyDoc(BaseDoc):
        tens: NdArray[10] = Field(dim=1000, col_type='varchar', max_len=1024)

    index = DummyDocIndex[MyDoc](work_dir=str(tmp_path))

    assert index._column_infos['tens'].db_type == 'varchar'
    assert index._column_infos['tens'].config['max_len'] == 1024


def test_columns_illegal_mapping(tmp_path):
    class MyDoc(BaseDoc):
        tens: NdArray[10] = Field(dim=1000, col_type='non_valid_type')

    with pytest.raises(
        ValueError, match='The given col_type is not a valid db type: non_valid_type'
    ):
        DummyDocIndex[MyDoc](work_dir=str(tmp_path))


def test_docs_validation():
    class OtherSimpleDoc(SimpleDoc):
        ...

    class OtherFlatDoc(FlatDoc):
        ...

    class OtherNestedDoc(NestedDoc):
        ...

    # SIMPLE
    index = DummyDocIndex[SimpleDoc]()
    in_list = [SimpleDoc(tens=np.random.random((10,)))]
    assert isinstance(index._validate_docs(in_list), DocList)
    assert isinstance(index._validate_docs(in_list)[0], BaseDoc)

    in_da = DocList[SimpleDoc](in_list)
    assert index._validate_docs(in_da) == in_da
    in_other_list = [OtherSimpleDoc(tens=np.random.random((10,)))]
    assert isinstance(index._validate_docs(in_other_list), DocList)
    assert isinstance(index._validate_docs(in_other_list)[0], BaseDoc)
    in_other_da = DocList[OtherSimpleDoc](in_other_list)
    assert index._validate_docs(in_other_da) == in_other_da

    with pytest.raises(ValueError):
        index._validate_docs(
            [
                FlatDoc(
                    tens_one=np.random.random((10,)), tens_two=np.random.random((50,))
                )
            ]
        )
    with pytest.raises(ValueError):
        index._validate_docs(
            DocList[FlatDoc](
                [
                    FlatDoc(
                        tens_one=np.random.random((10,)),
                        tens_two=np.random.random((50,)),
                    )
                ]
            )
        )

    # FLAT
    index = DummyDocIndex[FlatDoc]()
    in_list = [
        FlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))
    ]
    assert isinstance(index._validate_docs(in_list), DocList)
    assert isinstance(index._validate_docs(in_list)[0], BaseDoc)
    in_da = DocList[FlatDoc](
        [FlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))]
    )
    assert index._validate_docs(in_da) == in_da
    in_other_list = [
        OtherFlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))
    ]
    assert isinstance(index._validate_docs(in_other_list), DocList)
    assert isinstance(index._validate_docs(in_other_list)[0], BaseDoc)
    in_other_da = DocList[OtherFlatDoc](
        [
            OtherFlatDoc(
                tens_one=np.random.random((10,)), tens_two=np.random.random((50,))
            )
        ]
    )
    assert index._validate_docs(in_other_da) == in_other_da
    with pytest.raises(ValueError):
        index._validate_docs([SimpleDoc(tens=np.random.random((10,)))])
    with pytest.raises(ValueError):
        assert not index._validate_docs(
            DocList[SimpleDoc]([SimpleDoc(tens=np.random.random((10,)))])
        )

    # NESTED
    index = DummyDocIndex[NestedDoc]()
    in_list = [NestedDoc(d=SimpleDoc(tens=np.random.random((10,))))]
    assert isinstance(index._validate_docs(in_list), DocList)
    assert isinstance(index._validate_docs(in_list)[0], BaseDoc)
    in_da = DocList[NestedDoc]([NestedDoc(d=SimpleDoc(tens=np.random.random((10,))))])
    assert index._validate_docs(in_da) == in_da
    in_other_list = [OtherNestedDoc(d=OtherSimpleDoc(tens=np.random.random((10,))))]
    assert isinstance(index._validate_docs(in_other_list), DocList)
    assert isinstance(index._validate_docs(in_other_list)[0], BaseDoc)
    in_other_da = DocList[OtherNestedDoc](
        [OtherNestedDoc(d=OtherSimpleDoc(tens=np.random.random((10,))))]
    )

    assert index._validate_docs(in_other_da) == in_other_da
    with pytest.raises(ValueError):
        index._validate_docs([SimpleDoc(tens=np.random.random((10,)))])
    with pytest.raises(ValueError):
        index._validate_docs(
            DocList[SimpleDoc]([SimpleDoc(tens=np.random.random((10,)))])
        )


def test_docs_validation_unions():
    class OptionalDoc(BaseDoc):
        tens: Optional[NdArray[10]] = Field(dim=1000)

    class MixedUnionDoc(BaseDoc):
        tens: Union[NdArray[10], str] = Field(dim=1000)

    class TensorUnionDoc(BaseDoc):
        tens: Union[NdArray[10], AbstractTensor] = Field(dim=1000)

    # OPTIONAL
    index = DummyDocIndex[SimpleDoc]()
    in_list = [OptionalDoc(tens=np.random.random((10,)))]
    assert isinstance(index._validate_docs(in_list), DocList)
    assert isinstance(index._validate_docs(in_list)[0], BaseDoc)
    in_da = DocList[OptionalDoc](in_list)
    assert index._validate_docs(in_da) == in_da

    with pytest.raises(ValueError):
        index._validate_docs([OptionalDoc(tens=None)])

    # MIXED UNION
    index = DummyDocIndex[SimpleDoc]()
    in_list = [MixedUnionDoc(tens=np.random.random((10,)))]
    assert isinstance(index._validate_docs(in_list), DocList)
    assert isinstance(index._validate_docs(in_list)[0], BaseDoc)
    in_da = DocList[MixedUnionDoc](in_list)
    assert isinstance(index._validate_docs(in_da), DocList)
    assert isinstance(index._validate_docs(in_da)[0], BaseDoc)

    with pytest.raises(ValueError):
        index._validate_docs([MixedUnionDoc(tens='hello')])

    # TENSOR UNION
    index = DummyDocIndex[TensorUnionDoc]()
    in_list = [SimpleDoc(tens=np.random.random((10,)))]
    assert isinstance(index._validate_docs(in_list), DocList)
    assert isinstance(index._validate_docs(in_list)[0], BaseDoc)
    in_da = DocList[SimpleDoc](in_list)
    assert index._validate_docs(in_da) == in_da

    index = DummyDocIndex[SimpleDoc]()
    in_list = [TensorUnionDoc(tens=np.random.random((10,)))]
    assert isinstance(index._validate_docs(in_list), DocList)
    assert isinstance(index._validate_docs(in_list)[0], BaseDoc)
    in_da = DocList[TensorUnionDoc](in_list)
    assert index._validate_docs(in_da) == in_da


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

    doc = SubindexDoc(
        d=DocList[SimpleDoc](
            [
                SimpleDoc(
                    tens=t,
                )
            ]
        ),
    )
    assert np.all(DummyDocIndex._get_values_by_column([doc], 'd')[0].tens == t)

    doc = SubSubindexDoc(
        d_root=DocList[SubindexDoc](
            [
                SubindexDoc(
                    d=DocList[SimpleDoc](
                        [
                            SimpleDoc(
                                tens=t,
                            )
                        ]
                    ),
                )
            ]
        )
    )
    assert np.all(
        DummyDocIndex._get_values_by_column([doc], 'd_root')[0].d[0][0].tens == t
    )
    index = DummyDocIndex[SubSubindexDoc]()
    assert np.all(
        index._subindices['d_root']._get_values_by_column(doc.d_root, 'd')[0][0].tens
        == t
    )


def test_get_data_by_columns():
    index = DummyDocIndex[SimpleDoc]()
    docs = [SimpleDoc(tens=np.random.random((10,))) for _ in range(10)]
    data_by_columns = index._get_col_value_dict(docs)
    assert list(data_by_columns.keys()) == ['id', 'tens']
    assert list(data_by_columns['id']) == [doc.id for doc in docs]
    assert list(data_by_columns['tens']) == [doc.tens for doc in docs]

    index = DummyDocIndex[FlatDoc]()
    docs = [
        FlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))
        for _ in range(10)
    ]
    data_by_columns = index._get_col_value_dict(docs)
    assert list(data_by_columns.keys()) == ['id', 'tens_one', 'tens_two']
    assert list(data_by_columns['id']) == [doc.id for doc in docs]
    assert list(data_by_columns['tens_one']) == [doc.tens_one for doc in docs]
    assert list(data_by_columns['tens_two']) == [doc.tens_two for doc in docs]

    index = DummyDocIndex[NestedDoc]()
    docs = [NestedDoc(d=SimpleDoc(tens=np.random.random((10,)))) for _ in range(10)]
    data_by_columns = index._get_col_value_dict(docs)
    assert list(data_by_columns.keys()) == ['id', 'd__id', 'd__tens']
    assert list(data_by_columns['id']) == [doc.id for doc in docs]
    assert list(data_by_columns['d__id']) == [doc.d.id for doc in docs]
    assert list(data_by_columns['d__tens']) == [doc.d.tens for doc in docs]

    index = DummyDocIndex[DeepNestedDoc]()
    docs = [
        DeepNestedDoc(d=NestedDoc(d=SimpleDoc(tens=np.random.random((10,)))))
        for _ in range(10)
    ]
    data_by_columns = index._get_col_value_dict(docs)
    assert list(data_by_columns.keys()) == ['id', 'd__id', 'd__d__id', 'd__d__tens']
    assert list(data_by_columns['id']) == [doc.id for doc in docs]
    assert list(data_by_columns['d__id']) == [doc.d.id for doc in docs]
    assert list(data_by_columns['d__d__id']) == [doc.d.d.id for doc in docs]
    assert list(data_by_columns['d__d__tens']) == [doc.d.d.tens for doc in docs]

    index = DummyDocIndex[SubindexDoc]()
    docs = [
        SubindexDoc(
            d=DocList[SimpleDoc](
                [
                    SimpleDoc(
                        tens=np.random.random((10,)),
                    )
                ]
            ),
        )
        for _ in range(5)
    ]
    data_by_columns = index._get_col_value_dict(docs)
    assert list(data_by_columns.keys()) == ['id', 'd']
    assert list(data_by_columns['id']) == [doc.id for doc in docs]
    assert list(data_by_columns['d']) == [doc.d for doc in docs]

    index = DummyDocIndex[SubSubindexDoc]()
    docs = [
        SubSubindexDoc(
            d_root=DocList[SubindexDoc](
                [
                    SubindexDoc(
                        d=DocList[SimpleDoc](
                            [
                                SimpleDoc(
                                    tens=np.random.random((10,)),
                                )
                                for _ in range(2)
                            ]
                        ),
                    )
                    for _ in range(2)
                ]
            )
        )
        for _ in range(2)
    ]
    data_by_columns = index._get_col_value_dict(docs)
    assert list(data_by_columns.keys()) == ['id', 'd_root']
    assert list(data_by_columns['id']) == [doc.id for doc in docs]
    assert [
        doc
        for subsub_doc in list(data_by_columns['d_root'])
        for sub_doc in subsub_doc
        for doc in sub_doc.d
    ] == [doc for doc in docs for sub_doc in doc.d_root for doc in sub_doc.d]


def test_transpose_data_by_columns():
    index = DummyDocIndex[SimpleDoc]()
    docs = [SimpleDoc(tens=np.random.random((10,))) for _ in range(10)]
    data_by_columns = index._get_col_value_dict(docs)
    data_by_rows = list(index._transpose_col_value_dict(data_by_columns))
    assert len(data_by_rows) == len(docs)
    for doc, row in zip(docs, data_by_rows):
        assert doc.id == row['id']
        assert np.all(doc.tens == row['tens'])

    index = DummyDocIndex[FlatDoc]()
    docs = [
        FlatDoc(tens_one=np.random.random((10,)), tens_two=np.random.random((50,)))
        for _ in range(10)
    ]
    data_by_columns = index._get_col_value_dict(docs)
    data_by_rows = list(index._transpose_col_value_dict(data_by_columns))
    assert len(data_by_rows) == len(docs)
    for doc, row in zip(docs, data_by_rows):
        assert doc.id == row['id']
        assert np.all(doc.tens_one == row['tens_one'])
        assert np.all(doc.tens_two == row['tens_two'])

    index = DummyDocIndex[NestedDoc]()
    docs = [NestedDoc(d=SimpleDoc(tens=np.random.random((10,)))) for _ in range(10)]
    data_by_columns = index._get_col_value_dict(docs)
    data_by_rows = list(index._transpose_col_value_dict(data_by_columns))
    assert len(data_by_rows) == len(docs)
    for doc, row in zip(docs, data_by_rows):
        assert doc.id == row['id']
        assert doc.d.id == row['d__id']
        assert np.all(doc.d.tens == row['d__tens'])

    index = DummyDocIndex[DeepNestedDoc]()
    docs = [
        DeepNestedDoc(d=NestedDoc(d=SimpleDoc(tens=np.random.random((10,)))))
        for _ in range(10)
    ]
    data_by_columns = index._get_col_value_dict(docs)
    data_by_rows = list(index._transpose_col_value_dict(data_by_columns))
    assert len(data_by_rows) == len(docs)
    for doc, row in zip(docs, data_by_rows):
        assert doc.id == row['id']
        assert doc.d.id == row['d__id']
        assert doc.d.d.id == row['d__d__id']
        assert np.all(doc.d.d.tens == row['d__d__tens'])

    index = DummyDocIndex[SubindexDoc]()
    docs = [
        SubindexDoc(
            d=DocList[SimpleDoc](
                [
                    SimpleDoc(
                        tens=np.random.random((10,)),
                    )
                ]
            ),
        )
        for _ in range(5)
    ]
    data_by_columns = index._get_col_value_dict(docs)
    data_by_rows = list(index._transpose_col_value_dict(data_by_columns))
    assert len(data_by_rows) == len(docs)
    for doc, row in zip(docs, data_by_rows):
        assert doc.id == row['id']
        assert doc.d == row['d']

    index = DummyDocIndex[SubSubindexDoc]()
    docs = [
        SubSubindexDoc(
            d_root=DocList[SubindexDoc](
                [
                    SubindexDoc(
                        d=DocList[SimpleDoc](
                            [
                                SimpleDoc(
                                    tens=np.random.random((10,)),
                                )
                                for _ in range(5)
                            ]
                        ),
                    )
                    for _ in range(5)
                ]
            )
        )
        for _ in range(5)
    ]
    data_by_columns = index._get_col_value_dict(docs)
    data_by_rows = list(index._transpose_col_value_dict(data_by_columns))
    assert len(data_by_rows) == len(docs)
    for doc, row in zip(docs, data_by_rows):
        assert doc.id == row['id']
        assert [doc for sub_doc in doc.d_root for doc in sub_doc.d] == [
            doc for sub_doc in row['d_root'] for doc in sub_doc.d
        ]


def test_convert_dict_to_doc():
    index = DummyDocIndex[SimpleDoc]()
    doc_dict = {'id': 'simple', 'tens': np.random.random((10,))}
    doc = index._convert_dict_to_doc(doc_dict, index._schema)
    assert doc.id == doc_dict['id']
    assert np.all(doc.tens == doc_dict['tens'])

    index = DummyDocIndex[FlatDoc]()
    doc_dict = {
        'id': 'nested',
        'tens_one': np.random.random((10,)),
        'tens_two': np.random.random((50,)),
    }
    doc = index._convert_dict_to_doc(doc_dict, index._schema)
    assert doc.id == doc_dict['id']
    assert np.all(doc.tens_one == doc_dict['tens_one'])
    assert np.all(doc.tens_two == doc_dict['tens_two'])

    index = DummyDocIndex[NestedDoc]()
    doc_dict = {'id': 'nested', 'd__id': 'simple', 'd__tens': np.random.random((10,))}
    doc_dict_copy = doc_dict.copy()
    doc = index._convert_dict_to_doc(doc_dict, index._schema)
    assert doc.id == doc_dict_copy['id']
    assert doc.d.id == doc_dict_copy['d__id']
    assert np.all(doc.d.tens == doc_dict_copy['d__tens'])

    index = DummyDocIndex[DeepNestedDoc]()
    doc_dict = {
        'id': 'deep',
        'd__id': 'nested',
        'd__d__id': 'simple',
        'd__d__tens': np.random.random((10,)),
    }
    doc_dict_copy = doc_dict.copy()
    doc = index._convert_dict_to_doc(doc_dict, index._schema)
    assert doc.id == doc_dict_copy['id']
    assert doc.d.id == doc_dict_copy['d__id']
    assert doc.d.d.id == doc_dict_copy['d__d__id']
    assert np.all(doc.d.d.tens == doc_dict_copy['d__d__tens'])

    class MyDoc(BaseDoc):
        image: ImageDoc

    index = DummyDocIndex[MyDoc]()
    doc_dict = {
        'id': 'root',
        'image__id': 'nested',
        'image__tensor': np.random.random((128,)),
    }
    doc = index._convert_dict_to_doc(doc_dict, index._schema)

    if torch_imported:
        from docarray.typing.tensor.image.image_torch_tensor import ImageTorchTensor

    class MyDoc2(BaseDoc):
        tens: Union[NdArray, ImageTorchTensor]

    index = DummyDocIndex[MyDoc2]()
    doc_dict = {
        'id': 'root',
        'tens': np.random.random((128,)),
    }
    doc_dict_copy = doc_dict.copy()
    doc = index._convert_dict_to_doc(doc_dict, index._schema)
    assert doc.id == doc_dict_copy['id']
    assert np.all(doc.tens == doc_dict_copy['tens'])

    index = DummyDocIndex[SubindexDoc]()
    doc_dict = {
        'id': 'subindex',
        'parent_id': 'root',
        'tens': np.random.random((10,)),
    }
    doc_dict_copy = doc_dict.copy()
    doc = index._subindices['d']._convert_dict_to_doc(
        doc_dict, index._subindices['d']._schema
    )
    assert isinstance(doc, SimpleDoc)
    assert doc.id == doc_dict['id']
    assert np.all(doc.tens == doc_dict_copy['tens'])

    index = DummyDocIndex[SubSubindexDoc]()
    doc_dict = {
        'id': 'subsubindex',
        'parent_id': 'subindex',
        'tens': np.random.random((10,)),
    }
    doc_dict_copy = doc_dict.copy()
    doc = (
        index._subindices['d_root']
        ._subindices['d']
        ._convert_dict_to_doc(
            doc_dict, index._subindices['d_root']._subindices['d']._schema
        )
    )
    assert isinstance(doc, SimpleDoc)
    assert doc.id == doc_dict['id']
    assert np.all(doc.tens == doc_dict_copy['tens'])


def test_validate_search_fields():
    index = DummyDocIndex[SimpleDoc]()
    assert list(index._column_infos.keys()) == ['id', 'tens']

    # 'tens' is a valid field
    assert index._validate_search_field(search_field='tens')
    # should not fail when an empty string or None is passed
    assert index._validate_search_field(search_field='')
    index._validate_search_field(search_field=None)
    # 'ten' is not a valid field
    with pytest.raises(ValueError):
        index._validate_search_field('ten')


def test_len():
    index = DummyDocIndex[SimpleDoc]()
    count = len(index)
    assert count == 3


def test_update_subindex_data():
    index = DummyDocIndex[SubindexDoc]()
    docs = [
        SubindexDoc(
            id=f'{i}',
            d=DocList[SimpleDoc](
                [
                    SimpleDoc(
                        tens=np.random.random((10,)),
                    )
                    for _ in range(5)
                ]
            ),
        )
        for i in range(5)
    ]
    index._update_subindex_data(docs)
    for doc in docs:
        for subdoc in doc.d:
            assert subdoc.parent_id == doc.id

    index = DummyDocIndex[SubSubindexDoc]()
    docs = [
        SubSubindexDoc(
            d_root=DocList[SubindexDoc](
                [
                    SubindexDoc(
                        d=DocList[SimpleDoc](
                            [
                                SimpleDoc(
                                    tens=np.random.random((10,)),
                                )
                                for _ in range(5)
                            ]
                        ),
                    )
                    for _ in range(5)
                ]
            )
        )
        for _ in range(5)
    ]
    index._update_subindex_data(docs)
    for doc in docs:
        for subdoc in doc.d_root:
            assert subdoc.parent_id == doc.id
