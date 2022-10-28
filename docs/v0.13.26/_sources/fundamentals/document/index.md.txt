# Document

{class}`~docarray.document.Document` is the basic data type in DocArray. Whether you're working with text, image, video, audio, 3D meshes or the nested or the combined of them, you can always represent them as Document.

A Document object has a predefined data schema as below, each of the attributes can be set/get with the dot expression as you would do with any Python object.

(doc-fields)=
| Attribute   | Type            | Description                                                                                                                                                  |
|-------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| id          | string          | A hexdigest that represents a unique document ID                                                                                                             |
| blob        | bytes           | the raw binary content of this document, which often represents the original document                                                                        |
| tensor      | `ndarray`-like  | the ndarray of the image/audio/video document                                                                                                                |
| text        | string          | a text document                                                                                                                                              |
| granularity | int             | the depth of the recursive chunk structure                                                                                                                   |
| adjacency   | int             | the width of the recursive match structure                                                                                                                   |
| parent_id   | string          | the parent id from the previous granularity                                                                                                                  |
| weight      | float           | The weight of this document                                                                                                                                  |
| uri         | string          | a uri of the document could be: a local file path, a remote url starts with http or https or data URI scheme                                                 |
| modality    | string          | modality, an identifier to the modality this document belongs to. In the scope of multi/cross modal search                                                   |
| mime_type   | string          | mime type of this document, for blob content, this is required; for other contents, this can be guessed                                                      |
| offset      | float           | the offset of the doc                                                                                                                                        |
| location    | float           | the position of the doc, could be start and end index of a string; could be x,y (top, left) coordinate of an image crop; could be timestamp of an audio clip |
| chunks      | `DocumentArray` | list of the sub-documents of this document (recursive structure)                                                                                             |
| matches     | `DocumentArray` | the matched documents on the same level (recursive structure)                                                                                                |
| embedding   | `ndarray`-like  | the embedding of this document                                                                                                                               |
| tags        | dict            | a structured data value, consisting of field which map to dynamically typed values.                                                                          |
| scores      | `NamedScore`    | Scores performed on the document, each element corresponds to a metric                                                                                       |
| evaluations | `NamedScore`    | Evaluations performed on the document, each element corresponds to a metric                                                                                  |

```{tip}
An `ndarray`-like object can be a Python (nested) List/Tuple, Numpy ndarray, SciPy sparse matrix (spmatrix), TensorFlow dense and sparse tensor, PyTorch dense and sparse tensor, or PaddlePaddle dense tensor.
```

The data schema of the Document is comprehensive and well-organized. One can categorize those attributes into the following groups:

- Content related: `uri`, `text`, `tensor`, `blob`;
- Nest structure related: `chunks`, `matches`, `granularity`, `adjacency`, `parent_id`;
- Common side information or metadata: `id`, `modality`, `mime_type`, `offset`, `location`, `weight`;
  - Further information: `tags`;
- Computational related: `scores`, `evaluations`, `embedding`.

This picture depicts how you may want to construct or comprehend a Document object.

```{figure} images/document-attributes.svg
```


Document also provides a set of functions frequently used in data science and machine learning community.


## What's next?

To start, let's first see how to construct a Document object in {ref}`the next chapter<construct-doc>`.


```{toctree}
:hidden:

construct
serialization
attribute
embedding
nested
visualization
fluent-interface
```