import numpy as np
import pytest
import torch
from pydantic.tools import parse_obj_as, schema_json_of

from docarray import BaseDoc
from docarray.base_doc.io.json import orjson_dumps
from docarray.computation.tensorflow_backend import tnp
from docarray.typing import AnyEmbedding, NdArrayEmbedding, TorchEmbedding
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.typing.tensor.embedding import TensorFlowEmbedding


@pytest.mark.proto
def test_proto_embedding():
    embedding = parse_obj_as(AnyEmbedding, np.zeros((3, 224, 224)))

    embedding._to_node_protobuf()


def test_json_schema():
    schema_json_of(AnyEmbedding)


def test_dump_json():
    tensor = parse_obj_as(AnyEmbedding, np.zeros((3, 224, 224)))
    orjson_dumps(tensor)


@pytest.mark.parametrize(
    'tensor,cls_audio_tensor,cls_tensor',
    [
        (torch.zeros(1000, 2), TorchEmbedding, torch.Tensor),
        (np.zeros((1000, 2)), NdArrayEmbedding, np.ndarray),
    ],
)
def test_torch_ndarray_to_any_embedding(tensor, cls_audio_tensor, cls_tensor):
    class MyAudioDoc(BaseDoc):
        tensor: AnyEmbedding

    doc = MyAudioDoc(tensor=tensor)
    assert isinstance(doc.tensor, cls_audio_tensor)
    assert isinstance(doc.tensor, cls_tensor)
    assert (doc.tensor == tensor).all()


@pytest.mark.tensorflow
def test_tensorflow_to_any_embedding():
    class MyAudioDoc(BaseDoc):
        tensor: AnyEmbedding

    doc = MyAudioDoc(tensor=tf.zeros((1000, 2)))
    assert isinstance(doc.tensor, TensorFlowEmbedding)
    assert isinstance(doc.tensor.tensor, tf.Tensor)
    assert tnp.allclose(doc.tensor.tensor, tf.zeros((1000, 2)))
