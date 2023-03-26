from typing import Optional

import numpy as np
import pytest
import torch
from pydantic.tools import parse_obj_as, schema_json_of

from docarray import BaseDocument
from docarray.base_document.io.json import orjson_dumps
from docarray.typing import AudioTorchTensor, AudioUrl
from docarray.utils.misc import is_tf_available
from tests import TOYDATA_DIR

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.typing.tensor.audio import AudioTensorFlowTensor

VIDEODATA_DIR = TOYDATA_DIR / 'audio-data'
AUDIO_FILES = [
    str(VIDEODATA_DIR / 'hello.wav'),
    str(VIDEODATA_DIR / 'olleh.wav'),
]
REMOTE_AUDIO_FILE = 'https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/olleh.wav?raw=true'  # noqa: E501


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_audio_url(file_url):
    uri = parse_obj_as(AudioUrl, file_url)
    tensor, _ = uri.load()
    assert isinstance(tensor, np.ndarray)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_load_audio_url_to_audio_torch_tensor_field(file_url):
    class MyAudioDoc(BaseDocument):
        audio_url: AudioUrl
        tensor: Optional[AudioTorchTensor]

    doc = MyAudioDoc(audio_url=file_url)
    doc.tensor, _ = doc.audio_url.load()

    assert isinstance(doc.tensor, torch.Tensor)
    assert isinstance(doc.tensor, AudioTorchTensor)


@pytest.mark.tensorflow
@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_load_audio_url_to_audio_tensorflow_tensor_field(file_url):
    class MyAudioDoc(BaseDocument):
        audio_url: AudioUrl
        tensor: Optional[AudioTensorFlowTensor]

    doc = MyAudioDoc(audio_url=file_url)
    doc.tensor, _ = doc.audio_url.load()

    assert isinstance(doc.tensor, AudioTensorFlowTensor)
    assert isinstance(doc.tensor.tensor, tf.Tensor)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_load(file_url):
    url = parse_obj_as(AudioUrl, file_url)
    tensor, _ = url.load()
    assert isinstance(tensor, np.ndarray)


def test_json_schema():
    schema_json_of(AudioUrl)


def test_dump_json():
    url = parse_obj_as(AudioUrl, REMOTE_AUDIO_FILE)
    orjson_dumps(url)


@pytest.mark.parametrize(
    'path_to_file',
    [
        *[file for file in AUDIO_FILES],
        REMOTE_AUDIO_FILE,
    ],
)
def test_validation(path_to_file):
    url = parse_obj_as(AudioUrl, path_to_file)
    assert isinstance(url, AudioUrl)
    assert isinstance(url, str)


@pytest.mark.parametrize(
    'path_to_file',
    [
        'my/local/text/file.txt',
        'my/local/text/file.png',
    ],
)
def test_illegal_validation(path_to_file):
    with pytest.raises(ValueError, match='AudioUrl'):
        parse_obj_as(AudioUrl, path_to_file)


@pytest.mark.proto
@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_proto_audio_url(file_url):
    uri = parse_obj_as(AudioUrl, file_url)
    proto = uri._to_node_protobuf()
    assert 'audio_url' in str(proto)
