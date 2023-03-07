(dataclass)=
# Dataclass

```{figure} https://docs.docarray.org/_images/dataclass-banner.png
:width: 0 %
:scale: 0 %
```

```{figure} img/dataclass-banner.png
:scale: 0 %
```


```{toctree}
:hidden:

construct
access
new-type
example
```


DocArray's dataclass is a high-level API for representing a multimodal document using
{ref}`nested Document structure<recursive-nested-document>`.
It follows the design and idiom of the standard [Python dataclass](https://docs.python.org/3/library/dataclasses.html),
allowing users to represent a complicated multimodal document intuitively and process it easily via DocArray Document/DocumentArray API. 

In a nutshell, DocArray provides a `@dataclass` decorator and a set of multimodal types in `docarray.typing`,
which allows the multimodal document on the left to be represented as the code snippet on the right:

::::{grid} 2
:gutter: 1
:padding: 0

:::{grid-item-card}
:padding: 0

```{figure} img/image-mmdoc-example.png
:width: 100%
```

:::
:::{grid-item-card}
:padding: 0

```python
from docarray import dataclass, Document
from docarray.typing import Image, Text, JSON


@dataclass
class WPArticle:
    banner: Image
    headline: Text
    meta: JSON


a = WPArticle(
    banner='dog-cat-flight.png',
    headline='Everything to know about flying with pets, from picking your seats to keeping your animal calm',
    meta={
        'author': 'Nathan Diller',
        'column': 'By the Way - A Post Travel Destination',
    },
)

doc = Document(a)
```


:::

::::


Under the hood, `doc` is represented as a {class}`~docarray.document.Document` containing {attr}`~docarray.document.Document.chunks`
for each of `banner`, `headline` and `meta`.

But the beauty of DocArray's dataclass is that as a user you don't have to reason about `chunks` at all.
Instead, you define your data structure in your own words, and reason in the domain you are most familiar with.

Before we continue, let's spend some time understanding the problem and the rationale behind this feature.


## What is multi-modality?


```{tip}
It is highly recommended that you first read through the last two chapters on Document and DocumentArray before moving on, as they help you understand the problem we are solving here.
```

A multimodal document is a document that consists of a mixture of data modalities, such as image, text, audio, etc. Let's see some examples in real-world. Consider an article card (left) from The Washington Post and a sound effect card (right) from the BBC:


::::{grid} 2

:::{grid-item-card}

```{figure} img/image-mmdoc-example.png
:width: 100%
```

:::
:::{grid-item-card}

```{figure} img/sound-mmdoc-example.png
:width: 100%
```

:::

::::


The left card can be seen as a multimodal document: it consists of a sentence, an image, and some tags (i.e. author, name of column). The right card can be seen as a collection of multimodal documents, each of which consists of an audio clip and a description.

In practice, we want to express such multimodal documents with Document and DocumentArray, so that we can process each modality and leverage DocArray's full API, e.g. to embed, search, store and transfer the documents. That's the purpose of DocArray dataclass. 

## Understanding the problem

Given a multimodal document, we want to represent it with our [Document](../document/index.md) object. What we've learned so far is:
- A Document object is the basic IO unit for almost all of [DocArray's API](../document/fluent-interface.md).
- Each Document {ref}`can only contain one <mutual-exclusive>` data modality.
- A Document can be {ref}`nested<recursive-nested-document>` under `.chunks` or `.matches`.

With those in mind, to represent a multimodal document it seems that we need to put each modality in a separate Document and then nest them under a parent Document. For example, the article card from The Washington Post would look like:

::::{grid} 2

:::{grid-item-card}

```{figure} img/image-mmdoc-example.png
:width: 100%
```

:::
:::{grid-item-card}

```{figure} img/mmdoc-example.svg
:width: 100%
```

:::

::::

- `Doc1`, the image Document, containing the image's `.uri`, and `.tensor`.
- `Doc2`, the text Document, containing the card's `.text` field.
- `Doc0`, the container Document of `Doc1` and `Doc2`, also containing meta information like author name, column name in `.tags`.

This representation has many benefits:
- You can process and apply deep learning methods on each Document (aka modality) separately.
- Or _jointly_, by leveraging the nested relationship at the parent level.
- You can enjoy the full DocArray API, [Jina API](https://github.com/jina-ai/jina), [Hub Executors](https://cloud.jina.ai), [CLIP-as-service](https://clip-as-service.jina.ai/) and [Finetuner](https://github.com/jina-ai/finetuner) out of the box, without redesigning the data structure.

## Understanding the challenges

But why do we need a dataclass module? What are the challenges we're trying to solve? 

The first challenge is that such mapping is **arbitrary and implicit**. Given a real-world multimodal document, it's not straightforward to construct such a nested structure for new users of DocArray. The example above is simple, so the answer seems trivial. But what if you want to represent the following newspaper article as one Document? 

::::{grid} 2

:::{grid-item-card}

```{figure} img/complicate-example.png
:width: 100%
```

:::
:::{grid-item-card}

```{figure} img/mmdoc-complicated.svg
:width: 100%
```

:::

::::

The second challenge is accessing the nested sub-Documents. It should be as easy and consistent as constructing the Document in the first place.

The final challenge is playing well with the Jina ecosystem, letting users leverage existing APIs, algorithms and models to handle such multimodal documents. To be specific, users can use multimodal documents as I/O without changing their algorithms and models.   

## What's next?

DocArray's dataclass is designed to tackle these challenges by providing an elegant solution based on Python dataclass. It shares the same idiom as Python dataclass, allowing the user to define a multimodal document by adding type annotations. In the next sections, we'll see how it works.
