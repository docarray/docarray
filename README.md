# docarray

<!-- start elevator-pitch -->

The data structure for unstructured data.

🌌 **All data types**: super-expressive data structure for representing complicated/mixed/nested text, image, video, audio, 3D mesh data.

🧑‍🔬 **Data science powerhouse**: easy-to-use functions for facilitating data scientists work on embedding, matching, visualizing, evaluating unstructured data via Torch/Tensorflow/ONNX/PaddlePaddle.

🚡 **Portable**: ready to wire with efficient conversion from/to Protobuf, binary, JSON, CSV, dataframe.

<!-- end elevator-pitch -->


## Install 

Requires Python 3.7+ and `numpy`:
```
pip install docarray
```
To install full dependencies, please use `pip install docarray[full]`.

## [Documentation](https://docarray.jina.ai)

## Get Started

Let's use DocArray and ResNet50 to build a meme image search on [Totally Looks Like](https://sites.google.com/view/totally-looks-like-dataset). This dataset contains 6016 image-pairs stored in `/left` and `/right`. Images that shares the same filename are labeled as perceptually similar. For example, 

| `/left` | `/right` | `/left` | `/right` |
|---------|----------|---------|----------|
|

Our problem is given an image from `/left` and find its most-similar image in `/right` (without looking at the filename of course).

### Load images

First load images and preprocess them with standard computer vision techniques:

```python
from docarray import DocumentArray, Document

left_da = DocumentArray.from_files('left/*.jpg')
```

To get a feeling of the data you will handle, plot them in one sprite image:

```python
left_da.plot_image_sprites()
```

<p align="center">
<a href="https://docs.jina.ai"><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/sprite.png?raw=true" alt="Load totally looks like dataset with docarray API" width="70%"></a>
</p>

### Apply preprocessing

Let's do some standard computer vision preprocessing:

```python
def preproc(d: Document):
    return (d.load_uri_to_image_blob()  # load
             .set_image_blob_normalization()  # normalize color 
             .set_image_blob_channel_axis(-1, 0))  # switch color axis for the pytorch model later

left_da.apply(preproc)
```

Did I mention `apply` work in parallel?

### Embed images

Now convert images into embeddings using a pretrained ResNet50:

```python
import torchvision
model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
left_da.embed(model, device='cuda')  # embed via GPU to speedup
```

This step takes ~30 seconds on GPU. Beside Pytorch, you can also use Tensorflow, PaddlePaddle, ONNX models in `.embed(...)`.

### Visualize embeddings

You can visualize the embeddings via tSNE in an interactive embedding projector:

```python
left_da.plot_embeddings()
```

<p align="center">
<a href="https://docs.jina.ai"><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/tsne.gif?raw=true" alt="Visualizing embedding via tSNE and embedding projector" width="90%"></a>
</p>

Fun is fun, but recall our goal is to match left images against right images and so far we have only handled the left. Let's repeat the same procedure for the right:

```python
right_da = (DocumentArray.from_files('right/*.jpg')
                         .apply(preproc)
                         .embed(model, device='cuda'))
```

### Match nearest neighbours

We can now match the left to the right and take the top-9 results.

```python
left_da.match(right_da, limit=9)
```

Let's inspect what's inside `left_da` now:

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

Better see it.

```python
(DocumentArray(left_da[8].matches, copy=True)
    .apply(lambda d: d.set_image_blob_channel_axis(0, -1)
                      .set_image_blob_inv_normalization())
    .plot_image_sprites('result.png'))
```

<p align="center">
<a href="https://docs.jina.ai"><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/9nn-left.jpeg?raw=true" alt="Visualizing top-9 matches using DocArray API" width="40%"></a>
<a href="https://docs.jina.ai"><img src="https://github.com/jina-ai/docarray/blob/main/.github/README-img/9nn.png?raw=true" alt="Visualizing top-9 matches using DocArray API" width="40%"></a>
</p>

What we did here is reversing the preprocessing steps (i.e. switching axis and normalizing) on the copied matches, so that one can visualize them using image sprites.  

### Quantitative evaluation

Serious as you are, visual inspection is surely not enough. Let's calculate the recall@K. First we construct the groundtruth matches:

```python
groundtruth = DocumentArray(
    Document(uri=d.uri, matches=[Document(uri=d.uri.replace('left', 'right'))])
    for d in left_da
)
```

Here we create a new DocumentArray with real matches by simply replacing the filename, e.g. `left/00001.jpg` to `right/00001.jpg`. That's all we need: if the predicted match has the identical `uri` as the groundtruth match, then it is correct.

Now let's check recall rate from 1 to 5:

```python
for k in range(1, 6):
    print(
        f'recall@{k}',
        left_da.evaluate(
            groundtruth,
            hash_fn=lambda d: d.uri,
            metric='recall_at_k',
            k=k,
            max_rel=1,
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