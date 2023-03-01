# Add a new Document Index

In DocArray there exists the concept of _Document Index_, a class that takes `Document`s, optionally persists them,
and makes them searchable.

There are different Document Indexes leveraging different backends, such as Weaviate, Qdrant, HNSWLit etc.

This Document shows how to add a new Document Index to DocArray.

That process can be broken down into a number of basic steps:

1. Create a new class that inherits from `BaseDocumentIndex`
2. Declare default configurations for your Document Index
3. Implement abstract methods for indexing, searching, and deleting
4. Implement a Query Builder for your Document Index

For an end-to-end example of this process, you can check out the [existing HNSWLib Document Index implementation](https://github.com/docarray/docarray/pull/1124).

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

Overall, the following attributes will be set up automatically and be available to you (more info in the dedicated sections below):
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

You can declare allowed fields and default values for your `_db_config`, but you will see that later.

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
- `n_dim` is the dimensionality of the column, e.g. `100` for a 100-dimensional vector. For columns that are not vectors, this is `None`.
- `config` is a dictionary of configurations for the column. For example, for the `other_tensor` column above, this would contain the `space` and `dim` configurations.

Again, these are automatically populated for you, so you can just use them in your implementation.


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
  - **Caution**: While this is a perfectly fine thing to do, it might create more maintainance work for you in the future, because the public variant defined in the `BaseDocumentIndex` might change in the future, and you will have to update your implementation accordingly.

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

### The `python_type_to_db_type` method

This method is slightly special, because 1) it is not exposed to the user, and 2) you absolutely have to implement it.

It is intended to do the following: It takes a type of a field in the store's schema (e.g. `NdArray` for `tensor`), and returns the corresponding type in the database (e.g. `np.ndarray`).
The `BaseDocumentIndex` class uses this information to create and populate the `_Columns` in `self._columns`.

## Implement a Query Builder for your Document Index

This part of the process is still slightly WIP, so let's worry about it once we get there, shall we? ;)

