from typing import Optional

import numpy as np
import pytest
import torch
from pydantic.tools import parse_obj_as, schema_json_of

from docarray import BaseDocument
from docarray.base_document.io.json import orjson_dumps
from docarray.typing import (
    AudioNdArray,
    NdArray,
    VideoNdArray,
    VideoTorchTensor,
    VideoUrl,
)
from tests import TOYDATA_DIR

LOCAL_VIDEO_FILE = str(TOYDATA_DIR / 'mov_bbb.mp4')
REMOTE_VIDEO_FILE = 'https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'  # noqa: E501


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE],
)
def test_load(file_url):
    url = parse_obj_as(VideoUrl, file_url)
    audio, video, indices = url.load()

    assert isinstance(audio, np.ndarray)
    assert isinstance(audio, AudioNdArray)

    assert isinstance(video, np.ndarray)
    assert isinstance(video, VideoNdArray)

    assert isinstance(indices, np.ndarray)
    assert isinstance(indices, NdArray)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE],
)
def test_load_key_frames(file_url):
    url = parse_obj_as(VideoUrl, file_url)
    key_frames = url.load_key_frames()

    assert isinstance(key_frames, np.ndarray)
    assert isinstance(key_frames, VideoNdArray)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE],
)
def test_load_video_url_to_video_torch_tensor_field(file_url):
    class MyVideoDoc(BaseDocument):
        video_url: VideoUrl
        tensor: Optional[VideoTorchTensor]

    doc = MyVideoDoc(video_url=file_url)
    doc.tensor = doc.video_url.load_key_frames()

    assert isinstance(doc.tensor, torch.Tensor)
    assert isinstance(doc.tensor, VideoTorchTensor)


def test_json_schema():
    schema_json_of(VideoUrl)


def test_dump_json():
    url = parse_obj_as(VideoUrl, REMOTE_VIDEO_FILE)
    orjson_dumps(url)


@pytest.mark.parametrize(
    'path_to_file',
    [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE],
)
def test_validation(path_to_file):
    url = parse_obj_as(VideoUrl, path_to_file)
    assert isinstance(url, VideoUrl)
    assert isinstance(url, str)


@pytest.mark.parametrize(
    'path_to_file',
    [
        'illegal',
        'https://www.google.com',
        'my/local/text/file.txt',
        'my/local/text/file.png',
        'my/local/file.mp3',
    ],
)
def test_illegal_validation(path_to_file):
    with pytest.raises(ValueError, match='VideoUrl'):
        parse_obj_as(VideoUrl, path_to_file)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE],
)
def test_proto_video_url(file_url):
    uri = parse_obj_as(VideoUrl, file_url)
    proto = uri._to_node_protobuf()
    assert str(proto).startswith('video_url')
