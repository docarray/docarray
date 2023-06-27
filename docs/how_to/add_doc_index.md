# Add a new Document Index

In DocArray a _Document Index_ is a class that takes documents, optionally persists them,
and makes them searchable. Different Document Indexes leverage different backends, like Weaviate, Qdrant, HNSWLib etc.

This document shows covers adding a new Document Index to DocArray.

This can be broken down into a number of steps:

1. Install and user instructions
2. Create a new class that inherits from `BaseDocIndex`
3. Declare default configurations for your Document Index
4. Implement abstract methods for indexing, searching, and deleting
5. Implement a Query Builder for your Document Index

In general, the steps above can be followed in roughly that order.

However, a Document Index implementation is usually very interconnected, so you will probably have to jump between these steps a bit,
both in your implementation and in the guide below.

For an end-to-end example of this process, check out the [existing HNSWLib Document Index implementation](https://github.com/docarray/docarray/pull/1124).

!!! warning
    **Caution**: The HNSWLib Document Index implementation can be used as a reference, but it is special in some key ways.
    For example, HNSWLib can only index vectors, so it uses SQLite to store the rest of the documents alongside it.
    This is _not_ how you should store documents in your implementation! You can find guidance on how you _should_ do it below.


## Installation and user instructions

Add the library required for your Index via poetry:

```shell
poetry add {my_index_lib}
```

The `pyproject.toml` file should now look like this:

```toml
[tool.poetry.dependencies]
my_index_lib = ">=123.456.789"
```

Mark the library as optional and manually create an `extra` for it:

```toml
[tool.poetry.dependencies]
my_index_lib = {version = ">=0.6.2", optional = true }

[tool.poetry.extras]
my_index_extra = ["my_index_lib"]
```

In case the user tries to use your Index without the correct installs, we want to throw an error with corresponding instructions.

To enable this, first, add instructions to the `INSTALL_INSTRUCTIONS` dictionary in `docarray/utils/misc.py`, such as 

```python
{'my_index_lib': '"docarray[my_index_extra]"'}
```

Next, ensure you add a case to the `__getattr__()` in `docarray/index/__init__.py` for your new Index. By doing so, the user will be given the instructions when trying to import `MyIndex` without the correct libraries installed.

```python
if TYPE_CHECKING:
    from docarray.index.backends.my_index import MyIndex  # noqa: F401


def __getattr__(name: str):
    if name == 'HnswDocumentIndex':
        import_library('hnswlib', raise_error=True)
        from docarray.index.backends.my_index import MyIndex  # noqa

        __all__.append('MyIndex')
        return MyIndex
```

Additionally, wrap the required imports in the file where the `MyIndex` class will be located, like it is done in `docarray/index/backends/hnswlib.py`.

## Create a new Document Index class

To get started, create a new class that inherits from `BaseDocIndex` and `typing.Generic`:

```python
TSchema = TypeVar('TSchema', bound=BaseDoc)


class MyDocumentIndex(BaseDocIndex, Generic[TSchema]):
    ...
```

Here, `TSchema` is a type variable representing the schema of the Document Index, which is a `Document`.
You can use it in type hints of methods and attributes.

### Create the constructor

You can write an `__init__` method for your Document Index where you set up all the needed bells and whistles:

```python
def __init__(self, db_config=None, **kwargs):
    super().__init__(db_config=db_config, **kwargs)
    ...
```

Ensure you call the `super().__init__` method, which will do some basic initialization for you.

### Set up your backend

Your backend (database or similar) should represent documents in the following way:

- Every field of a document is a column in the database.
- Column types follow a default that you define, based on the type hint of the associated field, but can also be configured by the user.
- Every row in your database thus represents a document.
- **Nesting:** The most common way to handle nested documents (and the one where the `AbstractDocumentIndex` will hold your hand the most), is to flatten out nested documents. But if your backend natively supports nesting representations, then feel free to leverage those!

!!! warning
    Don't take too much inspiration from the HNSWLib Document Index implementation on this point, as it is a bit of a special case.

Also, check the Document Index is being set up "fresh", meaning no data was previously persisted.
Then create a new database table (or the equivalent concept in you backend) for the documents, otherwise, the Document Index should connect to the existing database and table.
You can determine this based on `self._db_config` (see below).

!!! note
    If you are integrating a database, your Document Index should always assume there is already a database running that it can connect to.
    It should _not_ spawn a new database instance.

To help with all of this, `super().__init__` inject a few helpful attributes for you (more info in the dedicated sections below):

- `self._schema`
- `self._db_config`
- `self._runtime_config`
- `self._column_infos`

### The `_schema`

When a user instantiates a Document Index, they do so in a parametric way:

```python
class Inner(BaseDoc):
    embedding: NdArray[512]


class MyDoc(BaseDoc):
    tensor: NdArray[100]
    other_tensor: NdArray = Field(dim=10, space='cosine')
    description: str
    inner: Inner


store = MyDocumentIndex[MyDoc]()
```

In this case, `store` would have a class attribute `_schema` that is the `MyDoc` class.
This is done automatically for you, and you can use it in your implementation.

### The `_db_config`

The `_db_config` is a dataclass that contains all "static" configurations of your Document Index.
Users can pass these configurations to the `__init__` method of your Document Index, and `self._db_config` will be populated
for you, so that you can use it in your implementation.

You can declare allowed fields and default values for your `_db_config`, but you will see that later.

### The `_runtime_config`

The `_runtime_config` is a dataclass that contains all "dynamic" configurations of your Document Index.
Users can pass these configurations to the `.configure()` method of your Document Index, and `self._runtime_config` will be populated
for you, so that you can use it in your implementation.

You can declare allowed fields and default values for your `_runtime_config `, but you will see that later.

### The `_column_infos`

`self._column_infos` is a dictionary that contains information about all columns in your Document Index instance.

This information is automatically extracted from `self._schema`, and populated for you.

Concretely, `self._column_infos: Dict[str, _ColumnInfo]` maps from a column name to a `_ColumnInfo` dataclass.

For the `MyDoc` schema above, the column names would be `tensor`, `other_tensor`, `description`, `id`, `inner__embedding`, and `inner__id`.
These are the key of `self._column_infos`.

The values of `self._column_infos` are `_ColumnInfo` dataclasses, which have the following form:

```python
@dataclass
class _ColumnInfo:
    docarray_type: Type
    db_type: Any
    n_dim: Optional[int]
    config: Dict[str, Any]
```

- `docarray_type` is the type of the column in DocArray, e.g. `AbstractTensor` or `str`
- `db_type` is the type of the column in the Document Index, e.g. `np.ndarray` or `str`. You can customize the mapping from `docarray_type` to `db_type`, as we will see later.
- `config` is a dictionary of configurations for the column. For example, the `other_tensor` column above would contain the `space` and `dim` configurations.
- `n_dim` is the dimensionality of the column, e.g. `100` for a 100-dimensional vector. See further guidance on this below.

Again, these are automatically populated for you, so you can just use them in your implementation.

!!! note
    `_ColumnInfo.docarray_type` contains the python type as specified in `self._schema`, whereas 
    `_ColumnInfo.db_type` contains the data type of a particular database column.

    By default, it holds that `_ColumnInfo.docarray_type == self.python_type_to_db_type(_ColumnInfo.db_type)`, as we will see later.
    However, you should not rely on this, because a user can manually specify a different db_type. 
    Therefore, your implementation should rely on `_ColumnInfo.db_type` and not directly call `python_type_to_db_type()`.

!!! warning
    If a subclass of `AbstractTensor` appears in the Document Index's schema (i.e. `TorchTensor`, `NdArray`, or `TensorFlowTensor`), then `_ColumnInfo.docarray_type` will simply show `AbstractTensor` instead of the specific subclass. This is because the abstract class normalizes all input data of type `AbstractTensor` to `np.ndarray` anyways, which should make your life easier. Just be sure to properly handle `AbstractTensor` as a possible value or `_ColumnInfo.docarray_type`, and you won't have to worry about the differences between torch, tf, and np.

### Properly handle `n_dim`

`_ColumnInfo.n_dim` is automatically obtained from type parametrizations of the form `NdArray[100]`;
if there isn't such a parametrization, `n_dim` of the columns will be `None`.

You should also provide another way of defining the dimensionality of your columns, specifically by exposing a parameter in `Field(...)` (see example schema at the top).

This leads to four possible scenarios:

**Scenario 1: Only `n_dim` is defined**

Imagine the user defines this schema:

```python
class MyDoc(BaseDoc):
    tensor: NdArray[100]


index = MyDocumentIndex[MyDoc]()
```

In that case, the following will be true: `self._column_infos['tensor'].n_dim == 100` and `self._column_infos['tensor'].config == {}`.
The `tensor` column in your backend should be configured to have dimensionality `100`.

**Scenario 2: Only `Field(...)` is defined**

Now, imagine the user defines _this_ schema:

```python
class MyDoc(BaseDoc):
    tensor: NdArray = Field(dim=50)


index = MyDocumentIndex[MyDoc]()
```

In that case, `self._column_infos['tensor'].n_dim is None` and `self._column_infos['tensor'].config['dim'] == 50`.
The `tensor` column in your backend should be configured to have dimensionality `50`.

**Scenario 3: Both `n_dim` and `Field(...)` are defined**

Now, imagine this schema:

```python
class MyDoc(BaseDoc):
    tensor: NdArray[100] = Field(dim=50)


index = MyDocumentIndex[MyDoc]()
```

In this case, `self._column_infos['tensor'].n_dim == 100` and `self._column_infos['tensor'].config['dim'] == 50`.
The `tensor` column in your backend should be configured to have dimensionality `100`, as **`n_dim` takes precedence over `Field(...)`**.

**Scenario 4: Neither `n_dim` nor `Field(...)` are defined**

Finally, imagine this:

```python
class MyDoc(BaseDoc):
    tensor: NdArray


index = MyDocumentIndex[MyDoc]()
```

In this case, `self._column_infos['tensor'].n_dim is None` and `self._column_infos['tensor'].config == {}`.
If your backend can handle tensor/embedding columns without defined dimensionality, you should leverage that mechanism.
Otherwise, raise an Exception.

## Declare default configurations

We have already made reference to the `_db_config` and `_runtime_config` attributes.

To define what can be stored in them, and what the default values are, you need to create two inner classes:

```python
@dataclass
class DBConfig(BaseDocIndex.DBConfig):
    default_column_config: Dict[Type, Dict[str, Any]] = ...


@dataclass
class RuntimeConfig(BaseDocIndex.RuntimeConfig):
    ...
```

!!! note
    - `DBConfig` inherits from `BaseDocIndex.DBConfig` and `RuntimeConfig` inherits from `BaseDocIndex.RuntimeConfig`
    - All fields in each dataclass need to have default values. Choose these sensibly, as they will be used if the user does not specify a value.

### The `DBConfig` class

The `DBConfig` class defines the static configurations of your Document Index.
These are configurations that are tied to the database (or library) running in the background, such as `host`, `port`, etc.
Here you should put everything that the user cannot or should not change after initialization.

!!! note
    Every `DBConfig` needs to contain a `default_column_config` field.
    This is a dictionary that, for each possible column type in your database, defines a default configuration for that column type.
    This will automatically be passed to a `_ColumnInfo` whenever a user does not manually specify a configuration for that column.

    For example, in the `MyDoc` schema above, the `tensor` `_ColumnInfo` would have a default configuration specified for `np.ndarray` columns.

What is actually contained in these type-dependant configurations is up to you (and database specific).
For example, for `np.ndarray` columns you could define the configurations `index_type` and `metric_type`,
and for `varchar` columns you could define a `max_length` configuration.

It is probably best to see this in action, so you should check out the `HnswDocumentIndex` implementation.

### The `RuntimeConfig` class

The `RuntimeConfig` class defines the dynamic configurations of your Document Index.
These are configurations that can be changed at runtime, for example default behaviours such as batch sizes, consistency levels, etc.

It is a common pattern to allow such parameters both in the `RuntimeConfig`, where they will act as global defaults, and
in specific methods (`index`, `find`, etc.), where they will act as local overrides.


## Implement abstract methods for indexing, searching, and deleting

After you've done the basic setup above, you can jump into the good stuff: implementing the actual indexing, searching, and deleting.

In general, the following is true:

- For every method that you need to implement, there is a public variant (e.g. `index`) and a private variant (e.g. `_index`)
- You should usually implement the private variant, which is called by the already-implemented public variant. This should make your life easier, because some preprocessing and data normalization will already be done for you.
- You can, however, also implement the public variant directly, if you want to do something special.

!!! warning
    While implementing the public variant directly is a perfectly fine thing to do, it may create more maintenance work for you in the future, because the public variant defined in the `BaseDocIndex` might change in the future, and you will have to update your implementation accordingly.

Further:

- You don't absolutely have to implement everything. If a feature (e.g. `text_search`) is not supported by your backend, just raise a `NotImplementedError` in the corresponding method.
- Many methods come in a "singular" variant (e.g. `find`) and a "batched" variant (e.g. `find_batched`).
  - The "singular" variant expects a single input, be it an ANN query, a text query, a filter, etc., and return matches and scores for that single input
  - The "batched" variant expects a batch of inputs, and returns of matches and scores for each input
- Your implementations of, e.g., `_find()`, `_index()` etc. are allowed to take additional optional keyword arguments.
  These can then be used to control DB specific behaviours, such as consistency levels, batch sizes, etc. As mentioned above, it is good practice to mirror these arguments in `self.RuntimeConfig`.

Overall, you're asked to implement the methods that appear after the `Abstract methods; Subclasses must implement these`
comment in the `BaseDocIndex` class.
The details of each method should become clear from the docstrings and type hints.

### The `python_type_to_db_type()` method

This method is slightly special, because 

1. It is not exposed to the user
2. You absolutely have to implement it

It is intended to take a type of a field in the store's schema (e.g. `AbstractTensor` for `tensor`), and return the corresponding type in the database (e.g. `np.ndarray`).

The `BaseDocIndex` class uses this information to create and populate the `_ColumnInfo`s in `self._column_infos`.

If the user wants to change the default behaviour, one can set the db type by using the `col_type` field:

```python
class MySchema(BaseDoc):
    my_num: float = Field(col_type='float64')
    my_text: str = Field(..., col_type='varchar', max_len=2048)
```

In this case, the `db_type` of `my_num` will be `'float64'` and the `db_type` of `my_text` will be `'varchar'`. 
Additional information regarding the `col_type`, such as `max_len` for `varchar` will be stored in the `_ColumnsInfo.config`.
The given `col_type` has to be a valid `db_type`, meaning that has to be described in the index's `DBConfig.default_column_config`.

### The `_index()` method

When indexing documents, your implementation should behave in the following way:

- Every field in the Document is mapped to a column in the database
- This includes the `id` field, which is mapped to the primary key of the database (if your backend has such a concept)
- The configuration of that column can be found in `self._column_infos[field_name].config`
- In DocArray <=0.21, we used to store a serialized representation of every document. This is not needed anymore, as every row in your database table should fully represent a single indexed document.

To handle nested documents, the public `index()` method already flattens every incoming document for you.
This means that `_index()` already receives a flattened representation of the data, and you don't need to worry about that.

Concretely, the `_index()` method takes as input a dictionary of column names to column data, flattened out.

!!! note
    If you (or your backend) prefer to do bulk indexing on row-wise data, then you can use the `self._transpose_col_value_dict()`
    helper method. Inside of `_index()` you can use this to transform `column_to_data` into a row-wise view of the data.

**If your backend has native nesting capabilities:** You can also ignore most of the above, and implement the public `index()` method directly.
That way you have full control over whether the input data gets flattened or not.

**The `.id` field:** Every Document has an `.id` field, which is intended to act as a unique identifier or primary key
in your backend, if such a concept exists in your case. In your implementation you can assume that `.id`s are **unique** and **non-empty**.
(Strictly speaking, this uniqueness property is not guaranteed, since a user could override the auto-generated `.id` field with a custom value.
If your implementation encounters a duplicate `.id`, it is okay to fail and raise an Exception.)

### The `_filter_by_parent_id()` method

The default implementatin return `None`. You can choose to override this function with database specific filter API when needed. 
This function should return a list of ids of subindex level documents given the id of root document.

### The `index_name()` property

The `index_name` property is used in the initialization of subindices, and the default implementation is empty. This function should return the name of the index. And if the property of the index name in your backend is not `index_name`, you need to convert it as the first step in `__init__()`, like `index_name` is assigned to `work_dir` in `docarray/index/backends/hnswlib.py`.


## Implement a Query Builder for your Document Index

Every Document Index exposes a Query Builder interface which the user can use to build composed, hybrid queries.

For you as a backend integrator, there are three main things that are related to this:

- The `QueryBuilder` class that you need to implement
- The `execute_query()` method that you need to implement
- The `build_query()` method that just returns an instance of your `QueryBuilder`. You _don't_ need to implement this yourself.

Overall, this interface is very flexible, meaning that not a whole lot of structure is imposed on you.
You can decide what happens in the `QueryBuilder` class, and how the query is executed in the `execute_query()` method.
But there are a few things that you should stick to.

### Implement the `QueryBuilder` class

The QueryBuilder is what accumulates partial queries and builds them into a single query, ready to be executed.

Your Query Builder has to be an inner class of your Document Index, its class name has to be `QueryBuilder`, and it has to inherit from the Base Query Builder:

```python
class QueryBuilder(BaseDocIndex.QueryBuilder):
    ...
```

The Query Builder exposes the following interface:

- The same query related methods as the `BaseDocIndex` class (e.g. `filter`, `find`, `text_search`, and their batched variants)
- The `build()` method

Its goal is to enable an interface for composing complex queries, like this:

```python
index = MyDocumentIndex[MyDoc]()
q = index.build_query().find(...).filter(...).text_search(...).build()
index.execute_query(q)
```

How the individual calls to `find`, `filter`, and `text_search` are combined is up to your backend.

### Implement individual query methods

It is up to you how you implement the individual query methods, e.g. `find`, `filter`, and `text_search` of the query builder.

However, there are a few common strategies that you could use: If your backend natively supports a query builder pattern,
then you could wrap that with the DocumentIndex Query Builder interface; or you could set these methods to simply collect
arguments passed to it and defer the actual query building to the `build()` method (the `HNSWLibIndex does this); or you
could eagerly build intermediate queries at every call.

No matter what you do, you should stick to one design principle: **Every call to `find`, `filter`, `text_search` etc.
should return a new instance of the Query Builder**, with updated state.

!!! note "If your backend does not support all operations"
    Most backends do not support compositions of all query operations, which is completely fine.
    If that is the case, you should handle that in the following way:

    - If an operation **is** supported by the Document Index that you are implementing, but **is not** supported by the Query Builder, you should use the pre-defined `_raise_not_composable()` helper method to raise a `NotImplementedError`.
    - If an operation **is not** supported by the Document Index that you are implementing, and **is not** supported by the Query Builder, you should use the pre-defined `_raise_not_supported()` helper method to raise a `NotImplementedError`.
    - If an operation **is** supported by the Document Index that you are implementing, and **is** supported by the Query Builder, but **is not** supported in combination with a certain other operation, you should raise a `RuntimeError`. Depending on how your Query Builder is set up, you might want to do that either eagerly during the conflicting method call, or lazily inside of `.build()`.

### Implement the `build()` method

It is up to you how you implement the `build()` method, and this will depend on how you implemented the individual query
methods in the section above.

Depending on this, `build()` could wrap a similar method of an underlying native query builder; or it could combine
the collected arguments and build an actual query; or it could even be a no-op.
The important thing is that it returns a query object that can be immediately executed by the `execute_query()` method.

What exactly this query object is, is up to you. It could be a string, a dictionary, a custom object, or anything else.

### Implement the `execute_query()` method

The `execute_query()` method of your Document Index has to fulfill two requirements:
1. Be able to execute a query that was built by the `build()` method of the corresponding `QueryBuilder` class, and return the results.
2. Be able to execute a native query object of your backend, as a simple pass-through, and return the results. This is intended for users to be able to use the native query language of your backend and manually construct their own queries, if they want to.

How you achieve this is up to you. If 1. and 2. condense down to the same thing because `build()` returns a native query
is also up to you.

The only thing to keep in mind is that any heavy lifting, combining of various inputs and queries, and potential validation
should happen in the `build()` method, and not in `execute_query()`.
