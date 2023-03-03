# Add a new Document Index

In DocArray there exists the concept of _Document Index_, a class that takes `Document`s, optionally persists them,
and makes them searchable.

There are different Document Indexes leveraging different backends, such as Weaviate, Qdrant, HNSWLib etc.

This document shows how to add a new Document Index to DocArray.

That process can be broken down into a number of basic steps:

1. Create a new class that inherits from `BaseDocumentIndex`
2. Declare default configurations for your Document Index
3. Implement abstract methods for indexing, searching, and deleting
4. Implement a Query Builder for your Document Index

In general, the steps above can be followed in roughly that order.

However, a Document Index implementation is usually very interconnected, so you will probably have to jump between these steps a bit,
both in your implementation and in the guide below.

For an end-to-end example of this process, you can check out the [existing HNSWLib Document Index implementation](https://github.com/docarray/docarray/pull/1124).

**Caution**: The HNSWLib Document Index implementation can be used as a reference, but it is special in some key ways.
For example, HNSWLib can only index vectors, so it uses SQLite to store the rest of the Documents alongside it.
This is _not_ how you should store Documents in your implementation! You can find guidance on how you _should_ do it below.

## Create a new Document Index class

To get started, create a new class that inherits from `BaseDocumentIndex` and `typing.Generic`:


```python
TSchema = TypeVar('TSchema', bound=BaseDocument)


class MyDocumentIndex(BaseDocumentIndex, Generic[TSchema]):
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

Make sure that you call the `super().__init__` method, which will do some basic initialization for you.

### Set up your backend

Your backend (database or similar) should represent Documents in the following way:
- Every field of a Document is a column in the database
- Column types follow a default that you define, based on the type hint of the associated field, but can also be configures by the user
- Every row in your database thus represents a Document
- **Nesting:** The most common way to handle nested Document (and the one where the `AbstractDocumentIndex` will hold your hand the most), is to flatten out nested Documents. But if your backend natively supports nesting representations, then feel free to leverage those!

**Caution**: Don't take too much inspiration from the HNSWLib Document Index implementation on this point, as it is a bit of a special case.


Also, you should check if the Document Index is being set up "fresh", meaning no data was previously persisted.
Then you should create a new database table (or the equivalent concept in you backend) for the Documents.
Otherwise, the Document Index should connect to the existing database and table.
You can determine this based on `self._db_config` (see below).

**Note:** If you are integrating a database, your Document Index should always assume that there is already a database running that it can connect to.
It should _not_ spawn a new database instance.

To help you with all of this, `super().__init__` inject a few helpful attributes for you (more info in the dedicated sections below):

- `self._schema`
- `self._db_config`
- `self._runtime_config`
- `self._columns`

### The `_schema`

When a user instantiates a Document Index, they do so in a parametric way, like so:

```python
class Inner(BaseDocument):
    embedding: NdArray[512]


class MyDoc(BaseDocument):
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

### The `_columns`

`self._columns` is a dictionary that contains information about all columns in your Document Index instance.

This information is automatically extracted from `self._schema`, and populated for you.

Concretely, `self._columns: Dict[str, _Column]` maps from a column name to a `_Column` dataclass.

For the `MyDoc` schema above, the column names would be `tensor`, `other_tensor`, `description`, `id`, `inner__embedding`, and `inner__id`.
These are the key of `self._columns`.

The values of `self._columns` are `_Column` dataclasses, which have the following form:

```python
@dataclass
class _Column:
    docarray_type: Type
    db_type: Any
    n_dim: Optional[int]
    config: Dict[str, Any]
```

- `docarray_type` is the type of the column in DocArray, e.g. `NdArray` or `str`
- `db_type` is the type of the column in the Document Index, e.g. `np.ndarray` or `str`. You can customize the mapping from `docarray_type` to `db_type`, as we will see later.
- `config` is a dictionary of configurations for the column. For example, for the `other_tensor` column above, this would contain the `space` and `dim` configurations.
- `n_dim` is the dimensionality of the column, e.g. `100` for a 100-dimensional vector. See further guidance on this below.

Again, these are automatically populated for you, so you can just use them in your implementation.

### Properly handle `n_dim`

`_Column.n_dim` is automatically obtained from type parametrizations of the form `NdArray[100]`;
if there isn't such a parametrization, `n_dim` of the columns will be `None`.

You should also provide another way of defining the dimensionality of your columns, specifically by exposing a parameter in `Field(...)` (see example schema at the top).

This leads to four possible scenarios:

**Scenario 1: Only `n_dim` is defined**

Imagine the user defines a schema like the following:

```python
class MyDoc(BaseDocument):
    tensor: NdArray[100]


index = MyDocumentIndex[MyDoc]()
```

In that case, the following will be true: `self._columns['tensor'].n_dim == 100` and `self._columns['tensor'].config == {}`.
The `tensor` column in your backend should be configured to have dimensionality 100.

**Scenario 2: Only `Field(...)` is defined**

Imagine the user defines a schema like the following:

```python
class MyDoc(BaseDocument):
    tensor: NdArray = Field(dim=50)


index = MyDocumentIndex[MyDoc]()
```

In that case, the following will be true: `self._columns['tensor'].n_dim is None` and `self._columns['tensor'].config['dim'] == 50`.
The `tensor` column in your backend should be configured to have dimensionality 50.

**Scenario 3: Both `n_dim` and `Field(...)` are defined**

Imagine the user defines a schema like the following:

```python
class MyDoc(BaseDocument):
    tensor: NdArray[100] = Field(dim=50)


index = MyDocumentIndex[MyDoc]()
```

In that case, the following will be true: `self._columns['tensor'].n_dim == 100` and `self._columns['tensor'].config['dim'] == 50`.
The `tensor` column in your backend should be configured to have dimensionality 100, as **`n_dim` takes precedence over `Field(...)`**.

**Scenario 4: Neither `n_dim` nor `Field(...)` are defined**

Imagine the user defines a schema like the following:

```python
class MyDoc(BaseDocument):
    tensor: NdArray


index = MyDocumentIndex[MyDoc]()
```

In that case, the following will be true: `self._columns['tensor'].n_dim is None` and `self._columns['tensor'].config == {}`.
If your backend can handle tensor/embedding columns without defined dimensionality, you should leverage that mechanism.
Otherwise, raise an Exception.

## Declare default configurations

We already made reference to the `_db_config` and `_runtime_config` attributes.

In order to define what can be stored in them, and what the default values are, you need to create two inner classes:

```python
@dataclass
class DBConfig(BaseDocumentIndex.DBConfig):
    ...


@dataclass
class RuntimeConfig(BaseDocumentIndex.RuntimeConfig):
    default_column_config: Dict[Type, Dict[str, Any]] = ...
```

Note that:
- `DBConfig` inherits from `BaseDocumentIndex.DBConfig` and `RuntimeConfig` inherits from `BaseDocumentIndex.RuntimeConfig`
- All fields in each dataclass need to have default values. Choose these sensibly, as they will be used if the user does not specify a value.

### The `DBConfig` class

The `DBConfig` class is used to define the static configurations of your Document Index.
These are configurations that are tied to the database (or library) running in the background, such as `host`, `port`, etc.
Here you should put everything that the user cannot or should not change after initialization.

### The `RuntimeConfig` class

The `RuntimeConfig` class is used to define the dynamic configurations of your Document Index.
These are configurations that can be changed at runtime, for example default behaviours such as batch sizes, consistency levels, etc.

It is a common pattern to allow such parameters both in the `RuntimeConfig`, where they will act as global defaults, and
in specific methods (`index`, `find`, etc.), where they will act as local overrides.

**Important**: Every `RuntimeConfig` needs to contain a `default_column_config` field.
This is a dictionary that, for each possible column type in your database, defines a default configuration for that column type.
This will automatically be passed to a `_Column` whenever a user does not manually specify a configuration for that column.

For example, in the `MyDoc` schema above, the `tensor` `_Column` would have a default configuration specified for `np.ndarray` columns.

What is actually contained in these type-dependant configurations is up to you (and database specific).
For example, for `np.ndarray` columns you could define the configurations `index_type` and `metric_type`,
and for `varchar` columns you could define a `max_length` configuration.

It is probably best to see this in action, so you should check out the `HnswDocumentIndex` implementation.

## Implement abstract methods for indexing, searching, and deleting

After you've done the basic setup above, you can jump into the good stuff: implementing the actual indexing, searching, and deleting.

In general, the following is true:
- For every method that you need to implement, there is a public variant (e.g. `index`) and a private variant (e.g. `_index`)
- You should usually implement the private variant, which is called by the already implemented public variant. This should make your life easier, because some preprocessing and data normalization will already be done for you.
- You can, however, also implement the public variant directly, if you want to do something special.
  - **Caution**: While this is a perfectly fine thing to do, it might create more maintenance work for you in the future, because the public variant defined in the `BaseDocumentIndex` might change in the future, and you will have to update your implementation accordingly.

Further:
- You don't absolutely have to implement everything. If a feature (e.g. `text_search`) is not supported by your backend, just raise a `NotImplementedError` in the corresponding method.
- Many methods come in a "singular" variant (e.g. `find`) and a "batched" variant (e.g. `find_batched`).
  - The "singular" variant expects a single input, be it an ANN query, a text query, a filter, etc., and return matches and scores for that single input
  - The "batched" variant expects a batch of inputs, and returns of matches and scores for each input
- Your implementations of, e.g., `_find()`, `_index()` etc. are allowed to take additional optional keyword arguments.
  These can then be used to control DB specific behaviours, such as consistency levels, batch sizes, etc. As mentioned above, it is good practice to mirror these arguments in `self.RuntimeConfig`.

Overall, you're asked to implement the methods that appear after the `Abstract methods; Subclasses must implement these`
comment in the `BaseDocumentIndex` class.
The details of each method should become clear from the docstrings and type hints.

### The `python_type_to_db_type()` method

This method is slightly special, because 1) it is not exposed to the user, and 2) you absolutely have to implement it.

It is intended to do the following: It takes a type of a field in the store's schema (e.g. `NdArray` for `tensor`), and returns the corresponding type in the database (e.g. `np.ndarray`).
The `BaseDocumentIndex` class uses this information to create and populate the `_Columns` in `self._columns`.

### The `_index()` method

When indexing Documents, your implementation should behave in the following way:

- Every field in the Document is mapped to a column in the database
- This includes the `id` field, which is mapped to the primary key of the database (if your backend has such a concept)
- The configuration of that column can be found in `self._columns[field_name].config`
- In DocArray v1, we used to store a serialized representation of every Document. This is not needed anymore, as every row in your DB table should fully represent a single indexed Document.

To handle nested Documents, the public `index()` method already flattens every incoming Document for you.
This means that `_index()` already receives a flattened representation of the data, and you don't need to worry about that.

Concretely, the `_index()` method takes as input a dictionary of column names to column data, flattened out.

**Note:** It has been brought to my attention that passing a row-wise representation instead of a column-wise representation might be more natural for some (most) backends. This will be addressed shortly.

**If your backend has native nesting capabilities:** You can also ignore most of the above, and implement the public `index()` method directly.
That way you have full control over whether the input data gets flattened or not.

## Implement a Query Builder for your Document Index

This part of the process is still slightly WIP, so let's worry about it once we get there, shall we? ;)

