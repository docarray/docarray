# Parallelization

```{seealso}
- {meth}`~docarray.array.mixins.parallel.ParallelMixin.map`: to parallel process Document by Document, return an interator of elements;
- {meth}`~docarray.array.mixins.parallel.ParallelMixin.map_batch`: to parallel process minibatch DocumentArray, return an iterator of DocumentArray;
- {meth}`~docarray.array.mixins.parallel.ParallelMixin.apply`: like `.map()`, modify DocumentArray inplace;
- {meth}`~docarray.array.mixins.parallel.ParallelMixin.apply_batch`: like `.map_batch()`, modify DocumentArray inplace.
```

Working with large DocumentArray in element-wise can be time-consuming. The naive way is to run a for-loop and enumerate all Document one by one. DocArray provides {meth}`~docarray.array.mixins.parallel.ParallelMixin.map` to speed up things quite a lot. It is like Python 
built-in `map()` function but mapping the function to every element of the DocumentArray in parallel. There is also {meth}`~docarray.array.mixins.parallel.ParallelMixin.map_batch` that works on the minibatch level.


Let's see an example, where we want to preprocess ~6000 image Documents. First we fill the URI to each Document.

```python
from docarray import DocumentArray

docs = DocumentArray.from_files('*.jpg')  # 6000 image Document with .uri set
```

To load and preprocess `docs`, we have:

```python
def foo(d):
    return (d.load_uri_to_image_blob()
             .set_image_blob_normalization()
             .set_image_blob_channel_axis(-1, 0))
```

This load the image from file into `.blob` do some normalization and set the channel axis. Now, let's compare the time difference when we do things sequentially and use `DocumentArray.map()` with different backends.

````{tab} For-loop

```python
for d in docs:
    foo(d)
```
````

````{tab} Map with process backend

```python
for d in docs.map(foo, backend='process'):
    pass
```
````

````{tab} Map with thread backend

```python
for d in docs.map(foo, backend='thread'):
    pass
```
````

```text
map-process ...	map-process takes 5 seconds (5.55s)
map-thread ...	map-thread takes 10 seconds (10.28s)
foo-loop ...	foo-loop takes 18 seconds (18.52s)
```

One can see a significant speedup with `.map()`.

If you only modify elements in-place, and do not need return values, you can write:

```python
da = DocumentArray(...)

da.apply(func)
```

```{admonition} When to choose process or thread backend?
:class: important

It depends on how your `func` in `.map(func)` look like:
- First, if you want `func` to modify elements inplace, the you can only use `thread` backend. With `process` backend you can only rely on the return values of `.map()`, the modification happens inside `func` is lost.
- Second, follow what people often suggests: IO-bound `func` uses `thread`, CPU-bound `func` uses `process`.
```
