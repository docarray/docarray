import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDoc
from docarray.documents import AudioDoc, VideoDoc
from docarray.typing import AudioNdArray, NdArray, VideoNdArray
from docarray.utils._internal.misc import is_tf_available
from tests import TOYDATA_DIR

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp


LOCAL_VIDEO_FILE = str(TOYDATA_DIR / 'mov_bbb.mp4')
REMOTE_VIDEO_FILE = 'https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true'  # noqa: E501


pytestmark = [pytest.mark.video]


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE])
def test_video(file_url):
    vid = VideoDoc(url=file_url)
    tensor, audio_tensor, key_frame_indices = vid.url.load()
    vid.tensor = tensor
    vid.audio = AudioDoc(tensor=audio_tensor)
    vid.key_frame_indices = key_frame_indices

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
    class MyDoc(BaseDoc):
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
