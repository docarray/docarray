(find-documentarray)=
# Finding Documents

```{important}

{meth}`~docarray.array.mixins.find.FindMixin.find` supports both **embedding-based nearest-neighbour search** & **document attributes filtering**.
```

In previous chapter, we saw how {meth}`~docarray.array.mixins.match.MatchMixin.match` can be used to match nearest neighbors of query documents using the `.embeddings` computed. Another way to accomplish the same task is to use the {meth}`~docarray.array.mixins.find.FindMixin.find` function. In addition to **matching nearest neighbors** using `embeddings`, the `find` function can also be used to **filter documents based on attributes** by conditions specified in a query dictionary.

```{seealso}
- {meth}`~docarray.array.mixins.match.MatchMixin.match`: find the nearest-neighbour Documents from another DocumentArray (or itself) based on their `.embeddings`.
```

## Searching Nearest-neighbour Documents

Like {meth}`~docarray.array.mixins.match.MatchMixin.match`, the {meth}`~docarray.array.mixins.find.FindMixin.find` method also finds the nearest neighbors of a given collection of query documents. The `find` method works almost the same as `match` and accepts the same options as `match`. For instance, like in the case of `match()`, you can specify the `device` (CPU/GPU) and the `batch_size`. It also supports matching every types of embeddings supported by `match`. 


The **only** difference is that `match()` is invoked with query `DocumentArray`, and takes the index documents as input. On the other hand, `find()` is invoked with the index `DocumentArray`, and takes query documents as input.

That is, the following two invocations are equivalent.

````{tab} .find
```{code-block} python
---
emphasize-lines: 1, 2
---

index_docs.find(
    query_docs,
    device='gpu',
    batch_size=10,
    limit=50,
    metric_name='cosine',
    exclude_self=True,
    only_id=False,
    **kwargs,
)
```
````

````{tab} .match
```{code-block} python
---
emphasize-lines: 1, 2
---

query_docs.match(
    index_docs,
    device='gpu',
    batch_size=10,
    limit=50,
    metric_name='cosine',
    exclude_self=True,
    only_id=False,
    **kwargs,
)
```
````

## Filtering Documents based on Attributes

We can also use {meth}`~docarray.array.mixins.find.FindMixin.find` to filter documents based on attributes with the conditions specified in a `query` dictionary.

The `query` dictionary defines the filtering conditions using the [MongoDB](https://docs.mongodb.com/manual/reference/operator/query/) query language. Let's take a look at how filtering by attributes can be done.

### Filtering by Attributes

As a simple example, let's consider the case when we want to filter documents with `text` equals `'hello'`. This can be done by:

```python
docs.find({'text': {'$eq': 'hello'}})
```

The above will return a `DocumentArray` in which each document has `doc.text == 'hello'`. We can compose multiple conditions using boolean logic operators. For instance, to filter by one or the other condition, we can:

```python
docs.find({'$or': [{'text': {'$eq': 'hello'}}, {'text': {'$eq': 'world'}}]})
```

The above returns a `DocumentArray` in which each `doc` in `docs` satisfies `doc.text == 'hello' or doc.text == 'world'`.


### Filtering by Tags

To filter by data in the `tags` attribute, we can:

```python
docs.find({'tags__number': {'$gt': 3}})
```

The above will return a `DocumentArray` in which each document has `doc.tags['number'] > 3`.


### Filtering Using Tags as Placeholder

We also use `tags` keys as placeholder by:

```python
docs.find({'text': {'$eq': '{tags__name}'}})
```

The above will return a `DocumentArray` in which each document has `doc.text == doc.tags['name']`.


### Supported Operators

Note, that only the following MongoDB's query operators are supported:

| Query Operator | Description                                                                                                |
|----------------|------------------------------------------------------------------------------------------------------------|
| `$eq`          | Equal to (number, string)                                                                                  |
| `$ne`          | Not equal to (number, string)                                                                              |
| `$gt`          | Greater than (number)                                                                                      |
| `$gte`         | Greater than or equal to (number)                                                                          |
| `$lt`          | Less than (number)                                                                                         |
| `$lte`         | Less than or equal to (number)                                                                             |
| `$in`          | Is in an array                                                                                             |
| `$nin`         | Not in an array                                                                                            |
| `$regex`       | Match the specified regular expression                                                                     |
| `$size`        | Match array/dict field that have the specified size. `$size` does not accept ranges of values.             |
| `$exists`      | Matches documents that have the specified field. And empty string content is also cosidered as not exists. |

For boolean logic operators, only the following are supported:


| Boolean Operator | Description                                        |
|------------------|----------------------------------------------------|
| `$and`           | Join query clauses with a logical AND              |
| `$or`            | Join query clauses with a logical OR               |
| `$not`           | Inverts the effect of a query expression           |

