(embed-via-model)=
# Embed via Neural Network

```{important}

{meth}`~docarray.array.mixins.embed.EmbedMixin.embed` supports both CPU & GPU.
```

When DocumentArray has `.tensors` set, you can use a neural network to {meth}`~docarray.array.mixins.embed.EmbedMixin.embed` it into vector representations, i.e. filling `.embeddings`. For example, our DocumentArray looks like the following:

```python
from docarray import DocumentArray
import numpy as np

docs = DocumentArray.empty(10)
docs.tensors = np.random.random([10, 128]).astype(np.float32)
```

Let's use a simple MLP in Pytorch/Keras/ONNX/Paddle as our embedding model:

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

By default, the filled `.embeddings` is in the given model framework's format. If you want it always be `numpy.ndarray`, use `.embed(..., to_numpy=True)`.

You can specify `.embed(..., device='cuda')` when working with GPU. The device name identifier depends on the model framework that you are using.

On large DocumentArray that does not fit into GPU memory, you can set `batch_size` via `.embed(..., batch_size=128)`.

You can use pretrained model from Keras/PyTorch/PaddlePaddle/ONNX for embedding:

```python
import torchvision
model = torchvision.models.resnet50(pretrained=True)
docs.embed(model)
```

After getting `.embeddings`, you can visualize it using {meth}`~docarray.array.mixins.plot.PlotMixin.plot_embeddings`, {ref}`find more details here<visualize-embeddings>`.

Note that `.embed()` only works when you have `.tensors` set, if you have `.texts` set and your model function supports string as the input, then you can always do the following to get embeddings:

```python
from docarray import DocumentArray

da = DocumentArray(...)
da.embeddings = my_text_model(da.texts)
```
