# What is DocArray?

```{figure} docarray-vs-banner.gif
:scale: 0 %
```

- It is like JSON, but for intensive computation.
- It is like `numpy.ndarray`, but for unstructured data. 
- It is like `pandas.DataFrame`, but for nested and mixed media data with embeddings.
- It is like Protobuf, but for data scientists and deep learning engineers. 

If you are a **data scientist** who works with image, text, video, audio data in Python all day, you should use DocArray: it can greatly accelerate the work on representing, embedding, matching, visualizing, evaluating, sharing data; while stay close with your favorite toolkits, e.g. Torch, Tensorflow, ONNX, PaddlePaddle, JupyterLab, Google Colab.

If you are a **deep learning engineer** who works on scalable deep learning service, you should use DocArray: it can be the basic building block of your system. Its portable data structure can be wired in Protobuf, compressed bytes, JSON; allowing your engineer friends to happily integrate it into the production system.

This is DocArray: a unique one, aiming to be *your data structure for unstructured data*.

## Design 

DocArray consists of two simple concepts:
- **Document**: a data structure for easily representing nested, unstructured data.
- **DocumentArray**: a container for efficiently accessing, processing, and understanding multiple Documents.

DocArray is designed to be extremely intuitive for Python users, no new syntax to learn. If you know how to Python, you know how to DocArray.

DocArray is designed to maximize the local experience, with the requirement of cloud readiness at anytime.

# Comparing to Alternatives

✅ Full support ✔ limited support ❌ no support

|                                 | DocArray     | `numpy.ndarray` | JSON | `pandas.DataFrame` | Protobuf |
|---------------------------------|--------------|--- |------|--- | --- |
| Tensor/matrix data              | ✅|✅| ❌    |✅|✔️️|
| Text data                       |✅|❌| ✅    |✅|✅|
| Media data                      |✅|❌| ❌    |❌|❌|
| Nested data                     |✅|❌| ✅    |❌|✅|
| Mixed data of the above four    |✅|❌| ❌    |❌|❌|
| Easy to (de)serialize           |✅|❌| ✅    |✅|✅|
| Pythonic experience             |✅|✅| ❌    |✔️️|❌|
| IO support for filetypes        |✅|❌| ❌    |❌|❌|
| Deep learning framework support |✅|✅| ❌    |❌|❌|
| multi-core/GPU support          |✅|✔️️| ❌    |❌|❌|
| Rich functions for data types   |✅|❌| ❌    |✅|❌|


There are two other packages that people often compare DocArray to, yet I haven't use them extensively. It would be unfair to put them in the above list, so here is a dedicated section for them. 

## To AwkwardArray

[AwkwardArray](https://awkward-array.org/quickstart.html) is a library for manipulating JSON/dict data via Numpy idioms. Instead of working with Python dynamically typed object, AwkwardArray converts the data into precompiled routines on contiguous data. Hence, it is highly efficient.

DocArray and AwkwardArray are designed with different purposes. DocArray comes from the context of deep learning engineering that works on a stream of multi/cross-modal Documents. AwkwardArray comes from particle physics where with high-performance number-crunching is the priority. Both shares the idea of having generic data structure, but are designed differently to maximize the productivity of their own domains. This results in different sets of feature functions. 

When it comes to the speed, AwkwardArray is fast at column access whereas DocArray is fast at row access (streaming):

```python
import awkward as ak
import numpy as np
from docarray import DocumentArray
from toytime import TimeContext

da = DocumentArray.empty(100_000)
da.embeddings = np.random.random([len(da), 64])

da.texts = [f'hello {j}' for j in range(len(da))]

ak_array = ak.from_iter(da.to_list())

with TimeContext('iter via DocArray'):
    for d in da:
        pass

with TimeContext('iter via awkward'):
    for r in ak_array:
        pass

with TimeContext('access text via DocArray'):
    da.texts

with TimeContext('access text via awkward'):
    ak_array['text']
```

```text
iter via DocArray ...	0.004s
iter via awkward ...	1.664s
access text via DocArray ...	0.031s
access text via awkward ...	0.000s
```

As one can see, you can convert a DocumentArray into AwkwardArray via `.to_list()`.

## To Zarr

[Zarr](https://zarr.readthedocs.io/en/stable/) is a format for the storage of chunked, compressed, N-dimensional arrays. I know Zarr quite long time ago, to me it is the package when a `numpy.ndarray` is so big to fit into memory. Zarr provides a comprehensive set of functions that allows one to chunk, compress, stream large NdArray. Hence, from that perspective, Zarr like `numpy.ndarray` focuses on numerical representation and computation.

In DocArray, the basic element one would work with is a Document, not `ndarray`. The support of `ndarray` is important, but not the full story: in the context of deep learning engineering, `ndarray` is often an intermediate representation of Document for computing, then throw away. Therefore, having a consistent data structure that lives *long enough* to cover creating, storing, computing, transferring, returning and rendering is one of the major motivations of DocArray.

## To Jina Users

Jina 2.0-2.6 *kind of* have their own "DocArray", with very similar `Document` and `DocumentArray` interface. However, it is an old design and codebase. Since then, many redesigns and improvements have been made to boost its efficiency, usability and portability. DocArray is now an independent package that other frameworks such as future Jina 3.x and Finetuner will rely on.

The first good reason to use DocArray is its efficiency. Here is a side-by-side speed comparison of Jina 2.6 vs. DocArray on some common tasks.

```{figure} speedup-vs2.svg
```

The benchmark was conducted on 100K Documents/DocumentArray averaged over 5 repetitions with min & max values removed.

The speedup comes from the complete redesign of Document and DocumentArray.

Beside code refactoring and optimization, many features have been improved, including:
- advanced indexing for both elements and attributes;
- comprehensive serialization protocols;
- unified and improved Pythonic interface; 
- improved visualization on Document and DocumentArray;
- revised documentations and examples
- ... and many more.

To learn DocArray, the recommendation here is to forget about everything in Jina 2.x, although some interfaces may look familiar. Read [the fundamental sections](../fundamentals/document/index.md) from beginning.

```{important}
The new Jina 3.0 (expected in Feb. 2022) will depend on the new DocArray. All Document & Document API from [Jina Docs](https://docs.jina.ai) will be removed. This documentation website of DocArray serves as the single source of truth. 
```
