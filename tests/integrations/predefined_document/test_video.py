import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDocument
from docarray.documents import Video
from docarray.typing import AudioNdArray, NdArray, VideoNdArray
from docarray.utils.misc import is_tf_available
from tests import TOYDATA_DIR

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp


LOCAL_VIDEO_FILE = str(TOYDATA_DIR / 'mov_bbb.mp4')
REMOTE_VIDEO_FILE = 'https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'  # noqa: E501


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE])
def test_video(file_url):
    vid = Video(url=file_url)
    vid.tensor, vid.audio.tensor, vid.key_frame_indices = vid.url.load()

    assert isinstance(vid.tensor, VideoNdArray)
    assert isinstance(vid.audio.tensor, AudioNdArray)
    assert isinstance(vid.key_frame_indices, NdArray)


def test_video_np():
    video = parse_obj_as(Video, np.zeros((10, 10, 3)))
    assert (video.tensor == np.zeros((10, 10, 3))).all()


def test_video_torch():
    video = parse_obj_as(Video, torch.zeros(10, 10, 3))
    assert (video.tensor == torch.zeros(10, 10, 3)).all()


@pytest.mark.tensorflow
def test_video_tensorflow():
    video = parse_obj_as(Video, tf.zeros((10, 10, 3)))
    assert tnp.allclose(video.tensor.tensor, tf.zeros((10, 10, 3)))


def test_video_shortcut_doc():
    class MyDoc(BaseDocument):
        video: Video
        video2: Video
        video3: Video

    doc = MyDoc(
        video='http://myurl.mp4',
        video2=np.zeros((10, 10, 3)),
        video3=torch.zeros(10, 10, 3),
    )
    assert doc.video.url == 'http://myurl.mp4'
    assert (doc.video2.tensor == np.zeros((10, 10, 3))).all()
    assert (doc.video3.tensor == torch.zeros(10, 10, 3)).all()
