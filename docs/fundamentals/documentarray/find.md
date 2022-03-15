(find-documentarray)=
# Finding Documents

In previous chapter, we saw how `.match` can be used to match nearest neighbors of query documents using the `embeddings` computed. An alternative way to accomplish the same task is to use the `.find` function. In addition to matching nearest neighbors using `embeddings`, the `.find` function can also be used to filter documents based on the attributes specified in a query dictionary.

```{important}

{meth}`~docarray.array.mixins.find.FindMixin.find` supports both **embedding-based nearest-neighbour search** & **document attributes filtering**.
```


```{seealso}
- {meth}`~docarray.array.mixins.match.MatchMixin.match`: find the nearest-neighbour Documents from another DocumentArray (or itself) based on their `.embeddings`.
```

## Searching Nearest-neighbour Documents

Like `.match`, the `.find` method also finds the nearest neighbors of a given collection of documents. You can use `.find` like you would in `.match`. The `.find` method accepts the same options of `.match`. For instance, you can also specify the device (CPU/GPU) and the `batch_size`. It also supports matching all the different types of embeddings as in `.match`. 


## Filtering Documents based on Attributes
