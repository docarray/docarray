import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDocument
from docarray.documents import VideoDoc
from docarray.typing import AudioNdArray, NdArray, VideoNdArray
from docarray.utils.misc import is_tf_available
from tests import TOYDATA_DIR

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp

VIDEODATA_DIR = TOYDATA_DIR / 'video-data'

VIDEO_FILE = [
    str(VIDEODATA_DIR / 'mov_bbb.mp4'),
    str(VIDEODATA_DIR / 'mov_bbb.avi'),
    str(VIDEODATA_DIR / 'mov_bbb.wmv'),
    str(VIDEODATA_DIR / 'mov_bbb.rm'),
    'https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/video-data/mov_bbb.mp4?raw=true',  # noqa: E501
]

NON_VIDEO_FILES = [
    str(TOYDATA_DIR / 'captions.csv'),
    str(TOYDATA_DIR / 'cube.ply'),
    str(TOYDATA_DIR / 'test.glb'),
    str(TOYDATA_DIR / 'test.png'),
    'illegal',
    'https://www.github.com',
]


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', VIDEO_FILE)
def test_video(file_url):
    vid = VideoDoc(url=file_url)
    vid.tensor, vid.audio.tensor, vid.key_frame_indices = vid.url.load()

    assert isinstance(vid.tensor, VideoNdArray)
    assert isinstance(vid.audio.tensor, AudioNdArray)
    assert isinstance(vid.key_frame_indices, NdArray)


def test_video_np():
    video = parse_obj_as(VideoDoc, np.zeros((10, 10, 3)))
    assert (video.tensor == np.zeros((10, 10, 3))).all()


def test_video_torch():
    video = parse_obj_as(VideoDoc, torch.zeros(10, 10, 3))
    assert (video.tensor == torch.zeros(10, 10, 3)).all()


@pytest.mark.tensorflow
def test_video_tensorflow():
    video = parse_obj_as(VideoDoc, tf.zeros((10, 10, 3)))
    assert tnp.allclose(video.tensor.tensor, tf.zeros((10, 10, 3)))


def test_video_shortcut_doc():
    class MyDoc(BaseDocument):
        video: VideoDoc
        video2: VideoDoc
        video3: VideoDoc

    doc = MyDoc(
        video='http://myurl.mp4',
        video2=np.zeros((10, 10, 3)),
        video3=torch.zeros(10, 10, 3),
    )
    assert doc.video.url == 'http://myurl.mp4'
    assert (doc.video2.tensor == np.zeros((10, 10, 3))).all()
    assert (doc.video3.tensor == torch.zeros(10, 10, 3)).all()


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', NON_VIDEO_FILES)
def test_non_video(file_url):
    with pytest.raises(Exception):
        audio = VideoDoc(url=file_url)
        _, _ = audio.url.load()
