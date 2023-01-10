from typing import Optional

import numpy as np
import pytest
import torch
from pydantic.tools import parse_obj_as, schema_json_of

from docarray import BaseDocument
from docarray.document_base.io.json import orjson_dumps
from docarray.typing import AudioNdArray, AudioTorchTensor, AudioUrl
from tests import TOYDATA_DIR

AUDIO_FILES = [
    str(TOYDATA_DIR / 'hello.wav'),
    str(TOYDATA_DIR / 'olleh.wav'),
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
    tensor = uri.load()
    assert isinstance(tensor, np.ndarray)
    assert isinstance(tensor, AudioNdArray)


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
    doc.tensor = doc.audio_url.load()

    assert isinstance(doc.tensor, torch.Tensor)
    assert isinstance(doc.tensor, AudioTorchTensor)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_load(file_url):
    url = parse_obj_as(AudioUrl, file_url)
    tensor = url.load()
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
        'illegal',
        'https://www.google.com',
        'my/local/text/file.txt',
        'my/local/text/file.png',
    ],
)
def test_illegal_validation(path_to_file):
    with pytest.raises(ValueError, match='AudioUrl'):
        parse_obj_as(AudioUrl, path_to_file)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_proto_audio_url(file_url):
    uri = parse_obj_as(AudioUrl, file_url)
    proto = uri._to_node_protobuf()
    assert str(proto).startswith('audio_url')
