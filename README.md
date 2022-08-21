<p align="center">
<img src="https://github.com/jina-ai/docarray/blob/main/docs/_static/logo-light.svg?raw=true" alt="DocArray logo: The data structure for unstructured data" width="150px">
<br>
<b>The data structure for unstructured multimodal data</b>
</p>

<p align=center>
<a href="https://pypi.org/project/docarray/"><img src="https://img.shields.io/pypi/v/docarray?style=flat-square&amp;label=Release" alt="PyPI"></a>
<a href="https://codecov.io/gh/jina-ai/docarray"><img alt="Codecov branch" src="https://img.shields.io/codecov/c/github/jina-ai/docarray/main?logo=Codecov&logoColor=white&style=flat-square"></a>
<a href="https://pypi.org/project/docarray/"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/docarray?style=flat-square"></a>
<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-3.1k-blueviolet?logo=slack&amp;logoColor=white&style=flat-square"></a>
</p>

<!-- start elevator-pitch -->

DocArray is a library for nested, unstructured, multimodal data in transit, including text, image, audio, video, 3D mesh, etc. It allows deep-learning engineers to efficiently process, embed, search, recommend, store, and transfer the multi-modal data with a Pythonic API.

🚪 **Door to cross-/multi-modal world**: super-expressive data structure for representing complicated/mixed/nested text, image, video, audio, 3D mesh data. The foundation data structure of [Jina](https://github.com/jina-ai/jina), [CLIP-as-service](https://github.com/jina-ai/clip-as-service), [DALL·E Flow](https://github.com/jina-ai/dalle-flow), [DiscoArt](https://github.com/jina-ai/discoart) etc.

🐍 **Pythonic experience**: designed to be as easy as a Python list. If you know how to Python, you know how to DocArray. Intuitive idioms and type annotation simplify the code you write.

🧑‍🔬 **Data science powerhouse**: greatly accelerate data scientists' work on embedding, k-NN matching, querying, visualizing, evaluating via Torch/TensorFlow/ONNX/PaddlePaddle on CPU/GPU.

🚡 **Data in transit**: optimized for network communication, ready-to-wire at anytime with fast and compressed serialization in Protobuf, bytes, base64, JSON, CSV, DataFrame. 

🎡 **Scale to big data**: handle out-of-memory data via on-disk document store while staying with exact same API experience. Supporting classic databases and vector databases to enable faster nearest neighbour search.

👒 **For modern apps**: GraphQL support makes your server versatile on request and response; built-in data validation and JSON Schema (OpenAPI) help you build reliable webservices.

🛸 **Integrate with IDE**: pretty-print and visualization on Jupyter notebook & Google Colab; comprehensive auto-complete and type hint in PyCharm & VS Code.

Read more on [why should you use DocArray](https://docarray.jina.ai/get-started/what-is/) and [comparison to alternatives](https://docarray.jina.ai/get-started/what-is/#comparing-to-alternatives). 

<p align="center">
<a href="#"><img src="https://github.com/jina-ai/jina/blob/master/.github/readme/core-tree-graph.svg?raw=true" alt="Jina in Jina AI neural search ecosystem" width="100%"></a>
</p>

<!-- end elevator-pitch -->


## [Documentation](https://docarray.jina.ai)

## Install 

Requires Python 3.7+ and `numpy` only:
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

### Example 1: represent multimodal data in dataclass

The following news article card can be easily represented via `docarray.dataclass` and type annotation:


<table>
<tr>
<td> 

<img src="https://github.com/jina-ai/docarray/blob/main/docs/fundamentals/dataclass/img/image-mmdoc-example.png?raw=true" alt="A example multimodal document" width="300px">
     
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


### Example 2: a 10-liners text matching

Let's search for top-5 similar sentences of <kbd>she smiled too much</kbd> in "Pride and Prejudice". 

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

When your data is too big, storing in memory is probably not a good idea. DocArray supports [multiple storage backends](https://docarray.jina.ai/advanced/document-store/) such as SQLite, Weaviate, Qdrant and ANNLite. They are all unified under **the exact same user experience and API**. Take the above snippet as an example, you only need to change one line to use SQLite:

```python
da = DocumentArray(
    (Document(text=s.strip()) for s in d.text.split('\n') if s.strip()),
    storage='sqlite',
)
```

The code snippet can still run **as-is**. All APIs remain the same, the code after are then running in a "in-database" manner. 

Besides saving memory, one can leverage storage backends for persistence, faster retrieval (e.g. on nearest-neighbour queries).



### Example 4: a complete workflow of visual search 

Let's use DocArray and the [Totally Looks Like](https://sites.google.com/view/totally-looks-like-dataset) dataset to build a simple meme image search. The dataset contains 6,016 image-pairs stored in `/left` and `/right`. Images that share the same filename are perceptually similar. For example:

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
    <td><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/left-00018.jpg?raw=true" alt="Visualizing top-9 matches using DocArray API" width="50%"></td>
    <td><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/right-00018.jpg?raw=true" alt="Visualizing top-9 matches using DocArray API" width="50%"></td>
    <td><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/left-00131.jpg?raw=true" alt="Visualizing top-9 matches using DocArray API" width="50%"></td>
    <td><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/right-00131.jpg?raw=true" alt="Visualizing top-9 matches using DocArray API" width="50%"></td>
  </tr>
</tbody>
</table>

Our problem is given an image from `/left`, can we find its most-similar image in `/right`? (without looking at the filename of course).

### Load images

First we load images. You *can* go to [Totally Looks Like](https://sites.google.com/view/totally-looks-like-dataset) website, unzip and load images as below:

```python
from docarray import DocumentArray

left_da = DocumentArray.from_files('left/*.jpg')
```

Or you can simply pull it from Jina Cloud:

```python
left_da = DocumentArray.pull('demo-leftda', show_progress=True)
```

You will see a running progress bar to indicate the downloading process.

To get a feeling of the data you will handle, plot them in one sprite image:

```python
left_da.plot_image_sprites()
```

<p align="center">
<a href="https://docarray.jina.ai"><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/sprite.png?raw=true" alt="Load totally looks like dataset with docarray API" width="60%"></a>
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

Now convert images into embeddings using a pretrained ResNet50:

```python
import torchvision

model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
left_da.embed(model, device='cuda')  # embed via GPU to speed up
```

This step takes ~30 seconds on GPU. Beside PyTorch, you can also use TensorFlow, PaddlePaddle, or ONNX models in `.embed(...)`.

### Visualize embeddings

You can visualize the embeddings via tSNE in an interactive embedding projector:

```python
left_da.plot_embeddings()
```

<p align="center">
<a href="https://docarray.jina.ai"><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/tsne.gif?raw=true" alt="Visualizing embedding via tSNE and embedding projector" width="90%"></a>
</p>

Fun is fun, but recall our goal is to match left images against right images and so far we have only handled the left. Let's repeat the same procedure for the right:


<table>
<tr>
<th> Pull from Cloud </th> 
<th> Download, unzip, load from local </th>
</tr>
<tr>
<td> 

```python
right_da = (
    DocumentArray.pull('demo-rightda', show_progress=True)
    .apply(preproc)
    .embed(model, device='cuda')
)
```
     
</td>
<td>

```python
right_da = (
    DocumentArray.from_files('right/*.jpg').apply(preproc).embed(model, device='cuda')
)
```

</td>
</tr>
</table>

### Match nearest neighbours

We can now match the left to the right and take the top-9 results.

```python
left_da.match(right_da, limit=9)
```

Let's inspect what's inside `left_da` matches now:

```python
for d in left_da:
    for m in d.matches:
        print(d.uri, m.uri, m.scores['cosine'].value)
```

```text
left/02262.jpg right/03459.jpg 0.21102
left/02262.jpg right/02964.jpg 0.13871843
left/02262.jpg right/02103.jpg 0.18265384
left/02262.jpg right/04520.jpg 0.16477376
...
```

Or shorten the loop as one-liner using the element & attribute selector:

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

What we did here is revert the preprocessing steps (i.e. switching axis and normalizing) on the copied matches, so that you can visualize them using image sprites.  

### Quantitative evaluation

Serious as you are, visual inspection is surely not enough. Let's calculate the recall@K. First we construct the groundtruth matches:

```python
groundtruth = DocumentArray(
    Document(uri=d.uri, matches=[Document(uri=d.uri.replace('left', 'right'))])
    for d in left_da
)
```

Here we create a new DocumentArray with real matches by simply replacing the filename, e.g. `left/00001.jpg` to `right/00001.jpg`. That's all we need: if the predicted match has the identical `uri` as the groundtruth match, then it is correct.

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

More metrics can be used such as `precision_at_k`, `ndcg_at_k`, `hit_at_k`.

If you think a pretrained ResNet50 is good enough, let me tell you with [Finetuner](https://github.com/jina-ai/finetuner) you could do much better in just 10 extra lines of code. [Here is how](https://finetuner.jina.ai/get-started/totally-looks-like/).


### Save results

You can save a DocumentArray to binary, JSON, dict, DataFrame, CSV or Protobuf message with/without compression. In its simplest form,

```python
left_da.save('left_da.bin')
```

To reuse it, do `left_da = DocumentArray.load('left_da.bin')`.


If you want to transfer a DocumentArray from one machine to another or share it with your colleagues, you can do:


```python
left_da.push('my_shared_da')
```

Now anyone who knows the token `my_shared_da` can pull and work on it.

```python
left_da = DocumentArray.pull('my_shared_da')
```

Intrigued? That's only scratching the surface of what DocArray is capable of. [Read our docs to learn more](https://docarray.jina.ai).


<!-- start support-pitch -->
## Support

- Check out the [Learning Bootcamp](https://learn.jina.ai) to get started with DocArray.
- Join our [Slack community](https://slack.jina.ai) and chat with other community members about ideas.
- Join our [Engineering All Hands](https://youtube.com/playlist?list=PL3UBBWOUVhFYRUa_gpYYKBqEAkO4sxmne) meet-up to discuss your use case and learn Jina's new features.
    - **When?** The second Tuesday of every month
    - **Where?**
      Zoom ([see our public events calendar](https://calendar.google.com/calendar/embed?src=c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com&ctz=Europe%2FBerlin)/[.ical](https://calendar.google.com/calendar/ical/c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com/public/basic.ics))
      and [live stream on YouTube](https://youtube.com/c/jina-ai)
- Subscribe to the latest video tutorials on our [YouTube channel](https://youtube.com/c/jina-ai)

## Join Us

DocArray is backed by [Jina AI](https://jina.ai) and licensed under [Apache-2.0](./LICENSE). [We are actively hiring](https://jobs.jina.ai) AI engineers, solution engineers to build the next neural search ecosystem in open-source.

<!-- end support-pitch -->
