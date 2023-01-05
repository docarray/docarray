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
@pytest.mark.parametrize('file_url', [LOCAL_VIDEO_FILE])  # , REMOTE_VIDEO_FILE])
def test_save_video_ndarray(file_url, tmpdir):
    tmp_file = str(TOYDATA_DIR / 'tmp.mp4')

    video_1 = Video(url=file_url)
    assert video_1.url == file_url

    video_1.tensor, _ = video_1.url.load()
    assert isinstance(video_1.tensor, np.ndarray)
    assert isinstance(video_1.tensor, VideoNdArray)

    # from PIL import Image
    # Image.fromarray(video_1.tensor[0]).show()

    video_1.tensor.save_to_file(tmp_file)
    assert os.path.isfile(tmp_file)
    print(f"video_1.tensor[0][:2] = {video_1.tensor[0][:2]}")

    video_2 = Video(url=tmp_file)
    video_2.tensor, _ = video_2.url.load()
    video_2.tensor.save_to_file(str(TOYDATA_DIR / 'tmp_2.mp4'))

    # video_3 = Video(url=str(tmpdir / f'tmp_2.mp4'))
    # video_3.tensor, _ = video_3.url.load()
    # video_3.tensor.save_to_file(str(tmpdir / f'tmp_3.mp4'))
    #
    # video_4 = Video(url=str(tmpdir / f'tmp_3.mp4'))
    # video_4.tensor, _ = video_4.url.load()
    # video_4.tensor.save_to_file(str(tmpdir / f'tmp_4.mp4'))
    #
    # video_5 = Video(url=str(tmpdir / f'tmp_4.mp4'))
    # video_5.tensor, _ = video_5.url.load()
    # video_5.tensor.save_to_file(str(tmpdir / f'tmp_5.mp4'))
    #
    # video_6 = Video(url=str(tmpdir / f'tmp_5.mp4'))
    # video_6.tensor, _ = video_6.url.load()
    # video_6.tensor.save_to_file(str(tmpdir / f'tmp_6.mp4'))
    #
    print(f"video_2.tensor[0][:2] = {video_2.tensor[0][:2]}")
    # print(f"video_3.tensor[0][:2] = {video_3.tensor[0][:2]}")
    # print(f"video_4.tensor[0][:2] = {video_3.tensor[0][:2]}")
    # print(f"video_5.tensor[0][:2] = {video_3.tensor[0][:2]}")
    # print(f"video_6.tensor[0][:2] = {video_3.tensor[0][:2]}")

    # Image.fromarray(video_1.tensor[0]).show()
    assert isinstance(video_1.tensor, np.ndarray)
    assert isinstance(video_1.tensor, VideoNdArray)
    assert video_1.tensor.shape == video_2.tensor.shape
    assert np.allclose(video_1.tensor, video_2.tensor, atol=100)
