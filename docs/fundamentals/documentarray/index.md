(documentarray)=
# DocumentArray

This is a Document, we already know it can be a mix in data types and nested in structure:

```{figure} images/docarray-single.svg
:width: 30%
```

Then this is a DocumentArray:

```{figure} images/docarray-array.svg
:width: 80%
```


{class}`~docarray.array.document.DocumentArray` is a list-like container of {class}`~docarray.document.Document` objects. It is **the best way** when working with multiple Documents.

In a nutshell, you can simply consider it as a Python `list`, as it implements **all** list interfaces. That is, if you know how to use Python `list`, you already know how to use DocumentArray.

It is also powerful as Numpy `ndarray` and Pandas `DataFrame`, allowing you to efficiently [access elements](access-elements.md) and [attributes](access-attributes.md) of contained Documents.

What makes it more exciting is those advanced features of DocumentArray. These features greatly accelerate data scientists work on accessing nested elements, evaluating, visualizing, parallel computing, serializing, matching etc. 

## What's next?

Let's see how to construct a DocumentArray {ref}`in the next section<construct-array>`.

```{toctree}
:hidden:

construct
serialization
access-elements
access-attributes
embedding
matching
evaluation
parallelization
visualization
```