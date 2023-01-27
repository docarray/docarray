<p align="center">
<img src="https://github.com/docarray/docarray/blob/main/docs/_static/logo-light.svg?raw=true" alt="DocArray logo: The data structure for unstructured data" width="150px">
<br>
<b>The data structure for multimodal data</b>
</p>

<p align=center>
<a href="https://pypi.org/project/docarray/"><img src="https://img.shields.io/pypi/v/docarray?style=flat-square&amp;label=Release" alt="PyPI"></a>
<a href="https://codecov.io/gh/docarray/docarray"><img alt="Codecov branch" src="https://img.shields.io/codecov/c/github/docarray/docarray/main?logo=Codecov&logoColor=white&style=flat-square"></a>
<a href="https://bestpractices.coreinfrastructure.org/projects/6554"><img src="https://bestpractices.coreinfrastructure.org/projects/6554/badge"></a>
<a href="https://pypistats.org/packages/docarray"><img alt="PyPI - Downloads from official pypistats" src="https://img.shields.io/pypi/dm/docarray?style=flat-square"></a>
<a href="https://discord.gg/WaMp6PVPgR"><img src="https://dcbadge.vercel.app/api/server/WaMp6PVPgR?theme=default-inverted&style=flat-square"></a>
</p>

<!-- start elevator-pitch -->

> ⬆️ **DocArray v2**: We are currently working on v2 of DocArray. Keep reading here if you are interested in the
> current (stable) version, or check out the [v2 alpha branch](https://github.com/docarray/docarray/tree/feat-rewrite-v2#readme)
> and [v2 roadmap](https://github.com/docarray/docarray/issues/780)!

DocArray is a library for nested, unstructured, multimodal data in transit, including text, image, audio, video, 3D mesh, etc. It allows deep-learning engineers to efficiently process, embed, search, recommend, store, and transfer multimodal data with a Pythonic API.

🚪 **Door to multimodal world**: super-expressive data structure for representing complicated/mixed/nested text, image, video, audio, 3D mesh data. The foundation data structure of [Jina](https://github.com/jina-ai/jina), [CLIP-as-service](https://github.com/jina-ai/clip-as-service), [DALL·E Flow](https://github.com/jina-ai/dalle-flow), [DiscoArt](https://github.com/jina-ai/discoart) etc.

🧑‍🔬 **Data science powerhouse**: greatly accelerate data scientists' work on embedding, k-NN matching, querying, visualizing, evaluating via Torch/TensorFlow/ONNX/PaddlePaddle on CPU/GPU.

🚡 **Data in transit**: optimized for network communication, ready-to-wire at anytime with fast and compressed serialization in Protobuf, bytes, base64, JSON, CSV, DataFrame. Perfect for streaming and out-of-memory data.

🔎 **One-stop k-NN**: Unified and consistent API for mainstream vector databases that allows nearest neighbor search including Elasticsearch, Redis, AnnLite, Qdrant, Weaviate.

👒 **For modern apps**: GraphQL support makes your server versatile on request and response; built-in data validation and JSON Schema (OpenAPI) help you build reliable web services.

🐍 **Pythonic experience**: as easy as a Python list. If you can Python, you can DocArray. Intuitive idioms and type annotation simplify the code you write.

🛸 **IDE integration**: pretty-print and visualization on Jupyter notebook and Google Colab; comprehensive autocomplete and type hints in PyCharm and VS Code.

Read more on [why should you use DocArray](https://docarray.jina.ai/get-started/what-is/) and [comparison to alternatives](https://docarray.jina.ai/get-started/what-is/#comparing-to-alternatives).

<!-- end elevator-pitch -->

DocArray was released under the open-source [Apache License 2.0](https://github.com/docarray/docarray/blob/main/LICENSE) in January 2022. It is currently a sandbox project under [LF AI & Data Foundation](https://lfaidata.foundation/).

## [Documentation](https://docarray.jina.ai)

## Install 

Requires Python 3.7+
```shell
pip install docarray
```
or via Conda:
```shell
conda install -c conda-forge docarray
```
[Commonly used features](https://docarray.jina.ai/#install) can be enabled via `pip install "docarray[common]"`.


## Get Started

DocArray consists of three simple concepts:

- **Document**: a data structure for easily representing nested, unstructured data.
- **DocumentArray**: a container for efficiently accessing, manipulating, and understanding multiple Documents.
- **Dataclass**: a high-level API for intuitively representing multimodal data.

Let's see DocArray in action with some examples.

### Example 1: represent multimodal data in a dataclass

You can easily represent the following news article card with `docarray.dataclass` and type annotation:


<table>
<tr>
<td> 

<img src="https://github.com/docarray/docarray/blob/main/docs/fundamentals/dataclass/img/image-mmdoc-example.png?raw=true" alt="A example multimodal document" width="300px">
     
</td>
<td>

```python
from docarray import dataclass, Document
from docarray.typing import Image, Text, JSON


@dataclass
class WPArticle:
    banner: Image
    headline: Text
    meta: JSON


a = WPArticle(
    banner='https://.../cat-dog-flight.png',
    headline='Everything to know about flying with pets, ...',
    meta={
        'author': 'Nathan Diller',
        'Column': 'By the Way - A Post Travel Destination',
    },
)

d = Document(a)
```

</td>
</tr>
</table>


### Example 2: text matching in 10 lines

Let's search for top-5 similar sentences of <kbd>she smiled too much</kbd> in "Pride and Prejudice":

```python
from docarray import Document, DocumentArray

d = Document(uri='https://www.gutenberg.org/files/1342/1342-0.txt').load_uri_to_text()
da = DocumentArray(Document(text=s.strip()) for s in d.text.split('\n') if s.strip())
da.apply(Document.embed_feature_hashing, backend='process')

q = (
    Document(text='she smiled too much')
    .embed_feature_hashing()
    .match(da, metric='jaccard', use_scipy=True)
)

print(q.matches[:5, ('text', 'scores__jaccard__value')])
```

```text
[['but she smiled too much.', 
  '_little_, she might have fancied too _much_.', 
  'She perfectly remembered everything that had passed in', 
  'tolerably detached tone. While she spoke, an involuntary glance', 
  'much as she chooses.”'], 
  [0.3333333333333333, 0.6666666666666666, 0.7, 0.7272727272727273, 0.75]]
```

Here the feature embedding is done by simple [feature hashing](https://en.wikipedia.org/wiki/Feature_hashing) and distance metric is [Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index). You have better embeddings? Of course you do! We look forward to seeing your results!

### Example 3: external storage for out-of-memory data

When your data is too big, storing in memory is not the best idea. DocArray supports [multiple storage backends](https://docarray.jina.ai/advanced/document-store/) such as SQLite, Weaviate, Qdrant and AnnLite. They're all unified under **the exact same user experience and API**. Take the above snippet: you only need to change one line to use SQLite:

```python
da = DocumentArray(
    (Document(text=s.strip()) for s in d.text.split('\n') if s.strip()),
    storage='sqlite',
)
```

The code snippet can still run **as-is**. All APIs remain the same, the subsequent code then runs in an "in-database" manner. 

Besides saving memory, you can leverage storage backends for persistence and faster retrieval (e.g. on nearest-neighbor queries).

### Example 4: complete workflow of visual search 

Let's use DocArray and the [Totally Looks Like](https://sites.google.com/view/totally-looks-like-dataset) dataset to build a simple meme image search. The dataset contains 6,016 image-pairs stored in `/left` and `/right`. Images that share the same filename appear similar to the human eye. For example:

<table>
<thead>
  <tr>
    <th>left/00018.jpg</th>
    <th>right/00018.jpg</th>
    <th>left/00131.jpg</th>
    <th>right/00131.jpg</th>
  </tr>
</thead>
<tbody>
  <tr align="center">
    <td><img src="https://github.com/docarray/docarray/blob/main/.github/README-img/left-00018.jpg?raw=true" alt="Visualizing top-9 matches using DocArray API" width="50%"></td>
    <td><img src="https://github.com/docarray/docarray/blob/main/.github/README-img/right-00018.jpg?raw=true" alt="Visualizing top-9 matches using DocArray API" width="50%"></td>
    <td><img src="https://github.com/docarray/docarray/blob/main/.github/README-img/left-00131.jpg?raw=true" alt="Visualizing top-9 matches using DocArray API" width="50%"></td>
    <td><img src="https://github.com/docarray/docarray/blob/main/.github/README-img/right-00131.jpg?raw=true" alt="Visualizing top-9 matches using DocArray API" width="50%"></td>
  </tr>
</tbody>
</table>

Given an image from `/left`, can we find the most-similar image to it in `/right`? (without looking at the filename).

### Load images

First we load images. You *can* go to [Totally Looks Like](https://sites.google.com/view/totally-looks-like-dataset)'s website, unzip and load images as below:

```python
from docarray import DocumentArray

left_da = DocumentArray.from_files('left/*.jpg')
```

Or you can simply pull it from Jina AI Cloud:

```python
left_da = DocumentArray.pull('jina-ai/demo-leftda', show_progress=True)
```

**Note**
If you have more than 15GB of RAM and want to try using the whole dataset instead of just the first 1,000 images, remove `[:1000]` when loading the files into the DocumentArrays `left_da` and `right_da`.


You'll see a progress bar to indicate how much has downloaded.

To get a feeling of the data, we can plot them in one sprite image. You need matplotlib and torch installed to run this snippet:

```python
left_da.plot_image_sprites()
```

<p align="center">
<a href="https://docarray.jina.ai"><img src="https://github.com/docarray/docarray/blob/main/.github/README-img/sprite.png?raw=true" alt="Load totally looks like dataset with docarray API" width="60%"></a>
</p>

### Apply preprocessing

Let's do some standard computer vision pre-processing:

```python
from docarray import Document


def preproc(d: Document):
    return (
        d.load_uri_to_image_tensor()  # load
        .set_image_tensor_normalization()  # normalize color
        .set_image_tensor_channel_axis(-1, 0)
    )  # switch color axis for the PyTorch model later


left_da.apply(preproc)
```

Did I mention `apply` works in parallel?

### Embed images

Now let's convert images into embeddings using a pretrained ResNet50:

```python
import torchvision

model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
left_da.embed(model, device='cuda')  # embed via GPU to speed up
```

This step takes ~30 seconds on GPU. Beside PyTorch, you can also use TensorFlow, PaddlePaddle, or ONNX models in `.embed(...)`.

### Visualize embeddings

You can visualize the embeddings via tSNE in an interactive embedding projector. You will need to have pydantic, uvicorn and FastAPI installed to run this snippet:

```python
left_da.plot_embeddings(image_sprites=True)
```

<p align="center">
<a href="https://docarray.jina.ai"><img src="https://github.com/docarray/docarray/blob/main/.github/README-img/tsne.gif?raw=true" alt="Visualizing embedding via tSNE and embedding projector" width="90%"></a>
</p>

Fun is fun, but our goal is to match left images against right images, and so far we have only handled the left. Let's repeat the same procedure for the right:

<table>
<tr>
<th> Pull from Cloud </th> 
<th> Download, unzip, load from local </th>
</tr>
<tr>
<td> 

```python
right_da = (
    DocumentArray.pull('jina-ai/demo-rightda', show_progress=True)
    .apply(preproc)
    .embed(model, device='cuda')[:1000]
)
```
     
</td>
<td>

```python
right_da = (
    DocumentArray.from_files('right/*.jpg')[:1000]
    .apply(preproc)
    .embed(model, device='cuda')
)
```

</td>
</tr>
</table>

### Match nearest neighbors

Now we can match the left to the right and take the top-9 results.

```python
left_da.match(right_da, limit=9)
```

Let's inspect what's inside `left_da` matches now:

```python
for m in left_da[0].matches:
    print(d.uri, m.uri, m.scores['cosine'].value)
```

```text
left/02262.jpg right/03459.jpg 0.21102
left/02262.jpg right/02964.jpg 0.13871843
left/02262.jpg right/02103.jpg 0.18265384
left/02262.jpg right/04520.jpg 0.16477376
...
```

Or shorten the loop to a one-liner using the element and attribute selector:

```python
print(left_da['@m', ('uri', 'scores__cosine__value')])
```

Better see it.

```python
(
    DocumentArray(left_da[8].matches, copy=True)
    .apply(
        lambda d: d.set_image_tensor_channel_axis(
            0, -1
        ).set_image_tensor_inv_normalization()
    )
    .plot_image_sprites()
)
```

<p align="center">
<a href="https://docarray.jina.ai"><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/9nn-left.jpeg?raw=true" alt="Visualizing top-9 matches using DocArray API" height="250px"></a>
<a href="https://docarray.jina.ai"><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/9nn.png?raw=true" alt="Visualizing top-9 matches using DocArray API" height="250px"></a>
</p>

Here we reversed the preprocessing steps (i.e. switching axis and normalizing) on the copied matches, so you can visualize them using image sprites.  

### Quantitative evaluation

Serious as you are, visual inspection is surely not enough. Let's calculate the recall@K. First we construct the groundtruth matches:

```python
groundtruth = DocumentArray(
    Document(uri=d.uri, matches=[Document(uri=d.uri.replace('left', 'right'))])
    for d in left_da
)
```

Here we created a new DocumentArray with real matches by simply replacing the filename, e.g. `left/00001.jpg` to `right/00001.jpg`. That's all we need: if the predicted match has the identical `uri` as the groundtruth match, then it is correct.

Now let's check recall rate from 1 to 5 over the full dataset:

```python
for k in range(1, 6):
    print(
        f'recall@{k}',
        left_da.evaluate(
            groundtruth, hash_fn=lambda d: d.uri, metric='recall_at_k', k=k, max_rel=1
        ),
    )
```

```text
recall@1 0.02726063829787234
recall@2 0.03873005319148936
recall@3 0.04670877659574468
recall@4 0.052194148936170214
recall@5 0.0573470744680851
```

You can also use other metrics like `precision_at_k`, `ndcg_at_k`, `hit_at_k`.

If you think a pretrained ResNet50 is good enough, let me tell you with [Finetuner](https://github.com/jina-ai/finetuner) you can do much better with [just another ten lines of code](https://finetuner.jina.ai/notebooks/image_to_image/).


### Save results

You can save a DocumentArray to binary, JSON, dict, DataFrame, CSV or Protobuf message with/without compression. In its simplest form:

```python
left_da.save('left_da.bin')
```

To reuse that DocumentArray's data, use `left_da = DocumentArray.load('left_da.bin')`.

If you want to transfer a DocumentArray from one machine to another or share it with your colleagues, you can do:

```python
left_da.push('my_shared_da')
```

Now anyone who knows the token `my_shared_da` can pull and work on it.

```python
left_da = DocumentArray.pull('<username>/my_shared_da')
```

Intrigued? That's only scratching the surface of what DocArray is capable of. [Read our docs to learn more](https://docarray.jina.ai).


<!-- start support-pitch -->
## Support & talk to us
- Join our [Discord server](https://discord.gg/WaMp6PVPgR) and chat with other community members about ideas.
- Join our [public meetings](https://calendar.google.com/calendar/u/2?cid=Y180NmJjYjQ3ZjEzN2QzOThjZjhjZmM2MzM0YTYyMjRkMjVhMjY1NTBlMGZkNjZkOGFmOWUyNjZiMDU4ODkyYmIxQGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20) where we discuss the future of the project.

> DocArray is a trademark of LF AI Projects, LLC
