# What is DocArray?

```{figure} docarray-vs-banner.gif
:scale: 0 %
```

- It's like JSON, but for intensive computation.
- It's like `numpy.ndarray`, but for unstructured data. 
- It's like `pandas.DataFrame`, but for nested and mixed media data with embeddings.
- It's like Protobuf, but for data scientists and deep learning engineers. 

If you are a **data scientist** who works with image, text, video, audio data in Python all day, you should use DocArray: it can greatly accelerate the work on representing, embedding, matching, visualizing, evaluating, sharing data; while staying close to your favorite toolkits, e.g. Torch, TensorFlow, ONNX, PaddlePaddle, JupyterLab, Google Colab.

If you are a **deep learning engineer** who works on scalable deep learning services, you should use DocArray: it can be the basic building block of your system. Its portable data structure can be wired in Protobuf, compressed bytes, JSON; allowing your engineer friends to happily integrate it into the production system.

This is DocArray: a unique one, aiming to be *your data structure for unstructured data*.

## Design 

DocArray consists of three simple concepts:
- **Document**: a data structure for easily representing nested, unstructured data.
- **DocumentArray**: a container for efficiently accessing, processing, and understanding multiple Documents.
- **Dataclass**: a high-level API for intuitively representing multimodal data.

DocArray is designed to be extremely intuitive for Python users, no new syntax to learn. If you know how to Python, you know how to DocArray.

DocArray is designed to maximize the local experience, with the requirement of cloud readiness at any time.

DocArray is designed to represent multimodal data intuitively to face the ever-increasing development of [multi/cross-modal applications](https://docs.jina.ai/get-started/what-is-cross-modal-multi-modal/).

```{figure} dataclass-example.svg
:width: 80%
```


# Comparing to Alternatives

✅ Full support ✔ Limited support ❌ No support

|                                 | DocArray     | `numpy.ndarray` | JSON | `pandas.DataFrame` | Protobuf |
|---------------------------------|--------------|--- |------|--- | --- |
| Tensor/matrix data              | ✅|✅| ❌    |✅|✔️️|
| Text data                       |✅|❌| ✅    |✅|✅|
| Media data                      |✅|❌| ❌    |❌|❌|
| Nested data                     |✅|❌| ✅    |❌|✅|
| Mixed data of the above four    |✅|❌| ❌    |❌|❌|
| Easy to (de)serialize           |✅|❌| ✅    |✅|✅|
| Data validation (of the output) |✅|❌| ❌    |❌|✅|
| Pythonic experience             |✅|✅| ❌    |✔️️|❌|
| IO support for filetypes        |✅|❌| ❌    |❌|❌|
| Deep learning framework support |✅|✅| ❌    |❌|❌|
| multi-core/GPU support          |✅|✔️️| ❌    |❌|❌|
| Rich functions for data types   |✅|❌| ❌    |✅|❌|


There are three other packages that people often compare DocArray to, yet I haven't used them extensively. It would be unfair to put them in the above list, so here is a dedicated section for them. 

## Hugging Face Datasets

[Hugging Face datasets](https://huggingface.co/docs/datasets/) is a library for easily accessing and sharing datasets for NLP, computer vision, and audio tasks. One of the highlights is its efficient loading of large datasets, which is highly appreciated during training.

In DocArray, there will also be a couple of feature releases soon to allow big data loading with constant memory consumption. However, the biggest difference is that DocArray is focused on **data in transit**, whereas HF Datasets is about **data at rest**. DocArray is focused on active data that is subject to frequent change; and allows efficient transfer between threads, processes and microservices. Data in transit often traverses a network or temporarily resides in memory to be read or updated. That's opposed to Datasets, where training data is stored physically and statically, and is subject to very occasional changes. The figure below depicts the differences:

```{figure} compare-data-type.svg
:width: 90%
```


## To AwkwardArray

[AwkwardArray](https://awkward-array.org/quickstart.html) is a library for manipulating JSON/dict data via NumPy idioms. Instead of working with Python dynamically typed objects, AwkwardArray converts data into precompiled routines on contiguous data. Hence, it is highly efficient.

DocArray and AwkwardArray are designed with different purposes: DocArray comes from the context of deep learning engineering that works on a stream of multi/cross-modal Documents. AwkwardArray comes from particle physics where with high-performance number-crunching is the priority. Both shares the idea of having a generic data structure, but are designed differently to maximize the productivity in their own domains. This results in different sets of feature functions. 

When it comes to speed, AwkwardArray is fast at column access whereas DocArray is fast at row access (streaming):

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

As you can see, you can convert a DocumentArray into AwkwardArray via `.to_list()`.

## To Zarr

[Zarr](https://zarr.readthedocs.io/en/stable/) is a format for the storage of chunked, compressed, N-dimensional arrays. I've known Zarr quite long time ago, to me it is the package when a `numpy.ndarray` is too big to fit into memory. Zarr provides a comprehensive set of functions that lets you chunk, compress, and stream a large ndarray. Hence, from that perspective, Zarr, like `numpy.ndarray` focuses on numerical representation and computation.

In DocArray, the basic element you work with is a Document, not `ndarray`. The support of `ndarray` is important, but not the full story: in the context of deep learning engineering, `ndarray` is often an intermediate representation of Document for computing, then thrown away. Therefore, having a consistent data structure that lives *long enough* to cover creating, storing, computing, transferring, returning and rendering is one of the major motivations of DocArray.

## To Jina Users

DocArray focuses on local and monolith developer experience. Jina scales DocArray to the Cloud. More details on their relations can be {ref}`found here<jina-support>`.

Jina 2.0-2.6 *kind of* had its own "DocArray", with very a similar `Document` and `DocumentArray` interface. However, it was an old design and codebase. Since then, many redesigns and improvements have been made to boost its efficiency, usability and portability. DocArray is now an independent package that other frameworks such as Jina 3.x and Finetuner rely on.

The first good reason to use DocArray is its efficiency. Here is a side-by-side speed comparison of Jina 2.6 vs. DocArray on some common tasks:

```{figure} speedup-vs2.svg
```

The benchmark was conducted on 100K Documents/DocumentArray averaged over five repetitions with min & max values removed.

The speedup comes from the complete redesign of Document and DocumentArray.

Beside code refactoring and optimization, many features have been improved, including:
- Advanced indexing for both elements and attributes
- Comprehensive serialization protocols
- Unified and improved Pythonic interface 
- Improved visualization of Document and DocumentArray
- Revised documentation and examples
- ... and many more

When first using DocArray, some Jina 2.x users may realize the static typing seems missing. This is due to a deliberate decision of DocArray: DocArray guarantees the types and constraints of the wire data, not the input data. In other words, only the functions that are listed under {ref}`docarray-serialization` chapter will trigger data validation. 

To learn DocArray, we recommend you forget about everything in Jina 2.x, although some interfaces may look familiar. Read [the fundamentals sections](../fundamentals/document/index.md) from the beginning.
