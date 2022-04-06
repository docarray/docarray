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

This is often more popular than `map()` in practice. However, `map()` has its own charm as we shall see in the next section.


Let's see an example, where we want to preprocess ~6000 image Documents. First we fill the URI to each Document.

```python
from docarray import DocumentArray

docs = DocumentArray.from_files('*.jpg')  # 6016 image Documents with .uri set
```

To load and preprocess `docs`, we have:

```python
def foo(d):
    return (
        d.load_uri_to_image_tensor()
        .set_image_tensor_normalization()
        .set_image_tensor_channel_axis(-1, 0)
    )
```

This load the image from file into `.tensor` do some normalization and set the channel axis. Now, let's compare the time difference when we do things sequentially and use `.apply()`:

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

(map-batch)=
## Use `map_batch()` to overlap CPU & GPU computation

As I said, {meth}`~docarray.array.mixins.parallel.ParallelMixin.map` / {meth}`~docarray.array.mixins.parallel.ParallelMixin.map_batch` has its own charm: it returns an iterator (of batch) where the partial result is immediately available, *regardless* if your function is still running. One can leverage this feature to speedup computation, especially when working with a CPU-GPU pipeline.

Let's see an example, say we have a DocumentArray with 1024 Documents, assuming we can run a CPU job for a 16-Document batch in 1 second/core; and we can run a GPU job for a 16-Document batch in 2 second/core. Say we have 4 CPU core and 1 GPU core as the total resources. 

Question: **how long will it take to process 1024 Documents?**


```python
import time

from docarray import DocumentArray

da = DocumentArray.empty(1024)


def cpu_job(da):
    print(f'cpu on {len(da)} docs')
    time.sleep(1)
    return da


def gpu_job(da):
    print(f'GPU on {len(da)} docs')
    time.sleep(2)
```


Before jump to the code, lets first whiteboard it, do a simple math:

```text
CPU time: 1024/16/4 * 1s = 16s
GPU time: 1024/16 * 2s   = 128s
Total time: 16s + 128s   = 144s   
```

So 144s, right? Yes, if we implement with `apply()`, it is around 144s.

However, we can do better. What if we overlap the computation of CPU and GPU? The whole procedure is anyway GPU bounded. If we can make sure GPU works on every batch **right away** when it is ready from CPU, rather than waits until all batches are ready from CPU, then we can save a lot of time. To be precise, we could do it in _129s_.

```{admonition} Why 129s? Why not 128s
:class: tip

Btw, if you immedidately know the answer you should [send your CV to us](https://jobs.jina.ai/).

Because the very first batch must be done by the CPU first, this is inevitible, which makes the 1 second non-overlapping. The rest of the time will be overlapped and dominated by GPU's 128s. Hence, 1s + 128s = 129s.
```

Okay, let's program these two ways and validate our guess:

````{tab} apply in 144s
```python
da.apply_batch(cpu_job, batch_size=16, num_worker=4)
for b in da.batch(batch_size=16):
    gpu_job(b)
```
````

````{tab} map in 129s
```python
for b in da.map_batch(cpu_job, batch_size=16, num_worker=4):
    gpu_job(b)
```
````

Which gives you,

```text
apply: 144.476s
map: 129.326s
```

Hope this sheds the light on solving the data-draining/blocking problem when you use DocArray in a CPU-GPU pipeline. 

## Use `map_batch()` to overlap CPU and network time

Such technique and mindset can be extended to other pipeline that has potential data-blocking issue. For example, in the implementation of {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.push`, you will find code similar to below:

```{code-block} python
---
emphasize-lines: 12
---

def gen():
    
    yield _head
    
    def _get_chunk(_batch):
        return b''.join(
            d._to_stream_bytes(protocol='protobuf', compress='gzip')
            for d in _batch
        ), len(_batch)

    for chunk, num_doc_in_chunk in self.map_batch(_get_chunk, batch_size=32):
        yield chunk
    yield _tail


response = requests.post(
    f'{_get_cloud_api()}/v2/rpc/artifact.upload',
    data=gen(),
    headers=headers,
)
```

This overlaps the time of sending network request (IO-bounded) with the time of serializing DocumentArray (CPU-bounded) and hence improve the performance a lot. 