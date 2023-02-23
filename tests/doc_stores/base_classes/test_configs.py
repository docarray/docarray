from dataclasses import dataclass, field
from typing import Any, Dict, Type

from pydantic import Field

from docarray import BaseDocument
from docarray.storage.abstract_doc_store import BaseDocumentIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDocument):
    tens: NdArray[10] = Field(dim=1000)


class FakeQueryBuilder:
    ...


@dataclass
class DBConfig(BaseDocumentIndex.DBConfig):
    work_dir: str = '.'
    other: int = 5


@dataclass
class RuntimeConfig(BaseDocumentIndex.RuntimeConfig):
    default_column_config: Dict[Type, Dict[str, Any]] = field(
        default_factory=lambda: {
            str: {
                'dim': 128,
                'space': 'l2',
            },
        }
    )
    default_ef: int = 50


def _identity(*x, **y):
    return x, y


class DummyDocIndex(BaseDocumentIndex):
    DBConfig = DBConfig
    RuntimeConfig = RuntimeConfig

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


def test_defaults():
    store = DummyDocIndex[SimpleDoc]()
    assert store._db_config.other == 5
    assert store._db_config.work_dir == '.'
    assert store._runtime_config.default_column_config[str] == {
        'dim': 128,
        'space': 'l2',
    }


def test_set_by_class():
    # change all settings
    store = DummyDocIndex[SimpleDoc](DBConfig(work_dir='hi', other=10))
    assert store._db_config.other == 10
    assert store._db_config.work_dir == 'hi'
    store.configure(RuntimeConfig(default_column_config={}, default_ef=10))
    assert store._runtime_config.default_column_config == {}

    # change only some settings
    store = DummyDocIndex[SimpleDoc](DBConfig(work_dir='hi'))
    assert store._db_config.other == 5
    assert store._db_config.work_dir == 'hi'
    store.configure(RuntimeConfig(default_column_config={}))
    assert store._runtime_config.default_column_config == {}


def test_set_by_kwargs():
    # change all settings
    store = DummyDocIndex[SimpleDoc](work_dir='hi', other=10)
    assert store._db_config.other == 10
    assert store._db_config.work_dir == 'hi'
    store.configure(default_column_config={}, default_ef=10)
    assert store._runtime_config.default_column_config == {}

    # change only some settings
    store = DummyDocIndex[SimpleDoc](work_dir='hi')
    assert store._db_config.other == 5
    assert store._db_config.work_dir == 'hi'
    store.configure(default_column_config={})
    assert store._runtime_config.default_column_config == {}


def test_default_column_config():
    store = DummyDocIndex[SimpleDoc]()
    assert store._runtime_config.default_column_config == {
        str: {
            'dim': 128,
            'space': 'l2',
        },
    }
