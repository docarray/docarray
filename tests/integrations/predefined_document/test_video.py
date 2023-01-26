import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDocument
from docarray.documents import Video
from docarray.typing import AudioNdArray, NdArray, VideoNdArray
from tests import TOYDATA_DIR

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
    image = parse_obj_as(Video, np.zeros((10, 10, 3)))
    assert (image.tensor == np.zeros((10, 10, 3))).all()


def test_video_torch():
    image = parse_obj_as(Video, torch.zeros(10, 10, 3))
    assert (image.tensor == torch.zeros(10, 10, 3)).all()


def test_video_shortcut_doc():
    class MyDoc(BaseDocument):
        image: Video
        image2: Video
        image3: Video

    doc = MyDoc(
        image='http://myurl.mp4',
        image2=np.zeros((10, 10, 3)),
        image3=torch.zeros(10, 10, 3),
    )
    assert doc.image.url == 'http://myurl.mp4'
    assert (doc.image2.tensor == np.zeros((10, 10, 3))).all()
    assert (doc.image3.tensor == torch.zeros(10, 10, 3)).all()
