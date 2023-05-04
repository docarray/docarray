(sqlite)=
# SQLite

You can use SQLite as a document store for DocumentArray. It's suitable for accessing a large number of Documents which can't fit in memory.

## Usage

```python
from docarray import DocumentArray

da = DocumentArray(storage='sqlite')  # with default config

da1 = DocumentArray(
    storage='sqlite', config={'connection': 'example.db'}
)  # with customize config
```

To reconnect a formerly persisted database, you can need to specify *both* `connection` and `table_name` in `config`:

```python
from docarray import DocumentArray

da = DocumentArray(
    storage='sqlite', config={'connection': 'example.db', 'table_name': 'mine'}
)

da.summary()
```

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name               | Description                                                                                                                            | Default            |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| `connection`       | SQLite database filename                                                                                                               | a random temp file |
| `table_name`       | SQLite table name                                                                                                                      | a random name      |
| `serialize_config` | [Serialization config of each Document](../../../fundamentals/document/serialization.md)                                               | None               |
| `conn_config`      | [Connection config pass to `sqlite3.connect`](https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection)                       | None               |
| `journal_mode`     | [SQLite Pragma: journal mode](https://www.sqlite.org/pragma.html#pragma_journal_mode)                                                  | `'DELETE'`         |
| `synchronous`      | [SQLite Pragma: synchronous](https://www.sqlite.org/pragma.html#pragma_synchronous)                                                    | `'OFF'`            |
| `list_like`        | Controls if ordering of Documents is persisted in the Database. Disabling this breaks list-like features, but can improve performance. | True               |
| `root_id`          | Boolean flag indicating whether to store `root_id` in the tags of chunk level Documents                                                | True               |
