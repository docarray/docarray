(embed-via-model)=
# Embed via Deep Neural Network

```{important}

{meth}`~jina.types.arrays.mixins.embed.EmbedMixin.embed` function supports both CPU & GPU, which can be specified by its `device` argument.
```

```{important}
You can use PyTorch, Keras, ONNX, PaddlePaddle as the embedding model.
```

When a `DocumentArray` has `.blobs` set, you can use a deep neural network to {meth}`~jina.types.arrays.mixins.embed.EmbedMixin.embed` it, which means filling `DocumentArray.embeddings`. For example, our `DocumentArray` looks like the following:

```python
from jina import DocumentArray
import numpy as np

docs = DocumentArray.empty(10)
docs.blobs = np.random.random([10, 128]).astype(np.float32)
```

And our embedding model is a simple MLP in Pytorch/Keras/ONNX/Paddle:

````{tab} PyTorch

```python
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=128,
        out_features=128,
    ),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=128, out_features=32))
```

````

````{tab} Keras
```python
import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32),
    ]
)

```
````

````{tab} ONNX

Preliminary: you need to first export a DNN model to ONNX via API/CLI. 
For example let's use the PyTorch one:

```python
data = torch.rand(1, 128)

torch.onnx.export(model, data, 'mlp.onnx', 
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=['input'],  # the model's input names
    output_names=['output'],  # the model's output names
    dynamic_axes={
        'input': {0: 'batch_size'},  # variable length axes
        'output': {0: 'batch_size'},
    })
```

Then load it as `InferenceSession`:
 
```python
import onnxruntime

model = onnxruntime.InferenceSession('mlp.onnx')
```
````

````{tab} Paddle

```python
import paddle

model = paddle.nn.Sequential(
    paddle.nn.Linear(
        in_features=128,
        out_features=128,
    ),
    paddle.nn.ReLU(),
    paddle.nn.Linear(in_features=128, out_features=32),
)

```
````

Now, you can simply do

```python
docs.embed(model)

print(docs.embeddings)
```

```text
tensor([[-0.1234,  0.0506, -0.0015,  0.1154, -0.1630, -0.2376,  0.0576, -0.4109,
          0.0052,  0.0027,  0.0800, -0.0928,  0.1326, -0.2256,  0.1649, -0.0435,
         -0.2312, -0.0068, -0.0991,  0.0767, -0.0501, -0.1393,  0.0965, -0.2062,
```


```{hint}
By default, `.embeddings` is in the model framework's format. If you want it always be `numpy.ndarray`, use `.embed(..., to_numpy=True)`. 
```

You can also use pretrained model for embedding:

```python
import torchvision
model = torchvision.models.resnet50(pretrained=True)
docs.embed(model)
```

You can also visualize `.embeddings` using Embedding Projector, {ref}`find more details here<visualize-embeddings>`.


```{hint}
On large `DocumentArray`, you can set `batch_size` via `.embed(..., batch_size=128)`
```


