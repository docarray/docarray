from typing import Optional

import numpy as np
import pytest
import torch
from pydantic.tools import parse_obj_as, schema_json_of

from docarray import BaseDoc
from docarray.base_doc.io.json import orjson_dumps
from docarray.typing import (
    AudioNdArray,
    NdArray,
    VideoBytes,
    VideoNdArray,
    VideoTorchTensor,
    VideoUrl,
)
from docarray.utils._internal.misc import is_tf_available
from tests import TOYDATA_DIR

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.typing.tensor.video import VideoTensorFlowTensor

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
    video, audio, indices = url.load()

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
@pytest.mark.parametrize(
    'field, attr_cls',
    [
        ('video', VideoNdArray),
        ('audio', AudioNdArray),
        ('key_frame_indices', NdArray),
    ],
)
def test_load_one_of_named_tuple_results(file_url, field, attr_cls):
    url = parse_obj_as(VideoUrl, file_url)
    result = getattr(url.load(), field)

    assert isinstance(result, np.ndarray)
    assert isinstance(result, attr_cls)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE],
)
def test_load_video_url_to_video_torch_tensor_field(file_url):
    class MyVideoDoc(BaseDoc):
        video_url: VideoUrl
        tensor: Optional[VideoTorchTensor]

    doc = MyVideoDoc(video_url=file_url)
    doc.tensor = doc.video_url.load().video

    assert isinstance(doc.tensor, torch.Tensor)
    assert isinstance(doc.tensor, VideoTorchTensor)


@pytest.mark.tensorflow
@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE],
)
def test_load_video_url_to_video_tensorflow_tensor_field(file_url):
    class MyVideoDoc(BaseDoc):
        video_url: VideoUrl
        tensor: Optional[VideoTensorFlowTensor]

    doc = MyVideoDoc(video_url=file_url)
    doc.tensor = doc.video_url.load().video

    assert isinstance(doc.tensor, VideoTensorFlowTensor)
    assert isinstance(doc.tensor.tensor, tf.Tensor)


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


@pytest.mark.proto
@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE],
)
def test_proto_video_url(file_url):
    uri = parse_obj_as(VideoUrl, file_url)
    proto = uri._to_node_protobuf()
    assert 'video_url' in str(proto)


def test_load_bytes():
    file_url = LOCAL_VIDEO_FILE
    uri = parse_obj_as(VideoUrl, file_url)
    video_bytes = uri.load_bytes()
    assert isinstance(video_bytes, bytes)
    assert isinstance(video_bytes, VideoBytes)
    assert len(video_bytes) > 0
