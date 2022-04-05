# Dataclass

```{toctree}
:hidden:

construct
access
field
example
```


DocArray's dataclass is a high-level API for representing a multimodal document using {ref}`nested Document structure<recursive-nested-document>`. It follows the design and idiom of the standard [Python dataclass](https://docs.python.org/3/library/dataclasses.html), allowing users to intuitively represent complicated multimodal document for neural search applications. 

In a nutshell, the left multimodal document can be represented with the right code snippet:

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
from docarray import dataclass, Image, Text, JSON, Document


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

d = Document(a)
```


:::

::::



## What is multimodal?


```{tip}
It is highly recommended that you first read through the last two chapters on Document and DocumentArray before moving on, as they help you understand the problem we are solving here.
```

A multimodal document is a document that consists of a mixture of data modalities, such as image, text, audio, etc. Let's see some examples in real-world. Considering an article card (left) from The Washington Post and a sound effect card (right) from BBC:


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


Each card can be seen as a multimodal document: it consists of a sentence, an image, an audio clip and some tags (i.e. author, column section).


In practice, we want to express such multimodal documents via Document and DocumentArray, so that we can process each modality and leverage all DocArray's API, e.g. to embed, search, store and transfer them. That's the purpose of DocArray dataclass. 

## Understanding the problem

Given a multimodal document, we want to represent it via our [Document](../document/index.md) object. What we have learned so far:
- A Document object is the basic IO unit for almost all [DocArray API](../document/fluent-interface.md).
- Each Document {ref}`can only contain one type of <mutual-exclusive>` data modality.
- A Document can be {ref}`nested<recursive-nested-document>` under `.chunks` or `.matches`.

Having those in mind, to represent a multimodal document it seems that we need to put each modality as a separated Document and then nested them under a parent Document. For example, the article card from The Washington Post would be represented as follows:

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

- `Doc1` the image Document, containing `.uri` of the image and `.tensor` representation of that banner image.
- `Doc2` the text Document, containing `.text` field of the card
- `Doc0` the container Document of `Doc1` and `Doc2`, also contains some meta information such as author name, column name in `.tags`.

Having this representation has many benefits, to name a few:
- One can process and apply deep learning methods on each Document (aka modality) separately.
- Or _jointly_, by leveraging the nested relationship at the parent level.
- One can enjoy all DocArray API, [Jina API](https://github.com/jina-ai/jina), [Hub Executors](https://hub.jina.ai), [CLIP-as-service](https://clip-as-service.jina.ai/) and [Finetuner](https://github.com/jina-ai/finetuner) out of the box, without redesigning the data structure.

## Understanding the challenges

But why do we need a dataclass module, what are the challenges here? 

The first challenge is that such mapping is **arbitrary and implicit**. Given a real-world multimodal document, it is not straightforward to construct such nested structure for new users of DocArray. The example above is simple, so the answer seems trivial. But what if I want to represent the following newspaper article as one Document? 

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

Second challenge is accessing the nested sub-Document. We want to provide users an easy way to access the nested sub-Document. It should be as easy and consistent as how they construct such Document in the first place.

Final challenge is how to play well with DocArray and Jina Ecosystem, allowing users to leverage existing API, algorithms and models to handle such multimodal documents. To be specific, user can use multimodal document as the I/O without changing their algorithms and models.   

## What's next

DocArray's dataclass is designed to tackle these challenges by providing an elegant solution based on Python dataclass. It shares the same idiom as Python dataclass, allowing user to define a multimodal document by adding type annotations. In the next sections, we shall see how it works.

