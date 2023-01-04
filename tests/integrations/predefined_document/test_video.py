import os

import numpy as np
import pytest

from docarray import Video
from docarray.typing import VideoNdArray
from tests import TOYDATA_DIR

LOCAL_VIDEO_FILE = str(TOYDATA_DIR / 'mov_bbb.mp4')
REMOTE_VIDEO_FILE = 'https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'  # noqa: E501


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE])
def test_video(file_url):
    video = Video(url=file_url)
    video.tensor, video.key_frame_indices = video.url.load()

    assert isinstance(video.tensor, np.ndarray)
    assert isinstance(video.tensor, VideoNdArray)
    assert isinstance(video.key_frame_indices, np.ndarray)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [LOCAL_VIDEO_FILE, REMOTE_VIDEO_FILE])
def test_save_video_ndarray(file_url, tmpdir):
    tmp_file = str(tmpdir / 'tmp.mp4')

    video = Video(url=file_url)
    video.tensor, _ = video.url.load()

    assert isinstance(video.tensor, np.ndarray)
    assert isinstance(video.tensor, VideoNdArray)

    video.tensor.save_to_file(tmp_file)
    assert os.path.isfile(tmp_file)

    video_from_file = Video(url=tmp_file)
    video_from_file.tensor = video_from_file.url.load()
    assert np.allclose(video.tensor, video_from_file.tensor)
