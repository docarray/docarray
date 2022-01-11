# Parallelization

```{seealso}
- {meth}`~docarray.array.mixins.parallel.ParallelMixin.map`: to parallel process Document by Document, return an interator of elements;
- {meth}`~docarray.array.mixins.parallel.ParallelMixin.map_batch`: to parallel process minibatch DocumentArray, return an iterator of DocumentArray;
- {meth}`~docarray.array.mixins.parallel.ParallelMixin.apply`: like `.map()`, modify DocumentArray inplace;
- {meth}`~docarray.array.mixins.parallel.ParallelMixin.apply_batch`: like `.map_batch()`, modify DocumentArray inplace.
```

Working with large DocumentArray in element-wise can be time-consuming. The naive way is to run a for-loop and enumerate all Document one by one. DocArray provides {meth}`~docarray.array.mixins.parallel.ParallelMixin.map` to speed up things quite a lot. It is like Python 
built-in `map()` function but mapping the function to every element of the DocumentArray in parallel. There is also {meth}`~docarray.array.mixins.parallel.ParallelMixin.map_batch` that works on the minibatch level.

`map()` returns an iterator of processed Documents. If you only modify elements in-place, and do not need the return values, you can use {meth}`~docarray.array.mixins.parallel.ParallelMixin.apply` instead:

```python
from docarray import DocumentArray

da = DocumentArray(...)

da.apply(func)
```

This is often more popular than `map()` in practice.


Let's see an example, where we want to preprocess ~6000 image Documents. First we fill the URI to each Document.

```python
from docarray import DocumentArray

docs = DocumentArray.from_files('*.jpg')  # 6016 image Documents with .uri set
```

To load and preprocess `docs`, we have:

```python
def foo(d):
    return (d.load_uri_to_image_blob()
             .set_image_blob_normalization()
             .set_image_blob_channel_axis(-1, 0))
```

This load the image from file into `.blob` do some normalization and set the channel axis. Now, let's compare the time difference when we do things sequentially and use `.apply()`:

````{tab} For-loop

```python
for d in docs:
    foo(d)
```
````

````{tab} Apply in parallel

```python
docs.apply(foo)
```
````

```text
foo-loop ...	foo-loop takes 11.5 seconds
apply ...	apply takes 3.5 seconds
```

One can see a significant speedup with `.apply()`.

By default, parallelization is conducted with `thread` backend, i.e. multi-threading. It also supports `process` backend by setting `.apply(..., backend='process')`.

```{admonition} When to choose process or thread backend?
:class: important

It depends on how your `func` in `.apply(func)` look like, here are some tips:
- First, if you want `func` to modify elements inplace, the you can only use `thread` backend. With `process` backend you can only rely on the return values of `.map()`, the modification happens inside `func` is lost.
- Second, follow what people often suggests: IO-bound `func` uses `thread`, CPU-bound `func` uses `process`.
- Last, ignore the second rule and what people told you. Test it by yourself and use whatever faster. 
```
