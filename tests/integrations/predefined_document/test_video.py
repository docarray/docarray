import pytest

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
    vid.audio_tensor, vid.video_tensor, vid.key_frame_indices = vid.url.load()

    assert isinstance(vid.audio_tensor, AudioNdArray)
    assert isinstance(vid.video_tensor, VideoNdArray)
    assert isinstance(vid.key_frame_indices, NdArray)
