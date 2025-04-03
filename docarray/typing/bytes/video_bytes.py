import warnings
from io import BytesIO
from typing import TYPE_CHECKING, List, NamedTuple, TypeVar

import numpy as np
from pydantic import parse_obj_as

from docarray.typing.bytes.base_bytes import BaseBytes
from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor import AudioNdArray, NdArray, VideoNdArray
from docarray.utils._internal.misc import import_library

T = TypeVar('T', bound='VideoBytes')


class VideoLoadResult(NamedTuple):
    video: VideoNdArray
    audio: AudioNdArray
    key_frame_indices: NdArray


@_register_proto(proto_type_name='video_bytes')
class VideoBytes(BaseBytes):
    """
    Bytes that store a video and that can be load into a video tensor
    """

    def load(self, **kwargs) -> VideoLoadResult:
        """
        Load the video from the bytes into a VideoLoadResult object consisting of:

        - a [`VideoNdArray`][docarray.typing.VideoNdArray] (`VideoLoadResult.video`)
        - an [`AudioNdArray`][docarray.typing.AudioNdArray] (`VideoLoadResult.audio`)
        - an [`NdArray`][docarray.typing.NdArray] containing the key frame indices (`VideoLoadResult.key_frame_indices`).

        ---

        ```python
        from docarray import BaseDoc
        from docarray.typing import AudioNdArray, NdArray, VideoNdArray, VideoUrl


        class MyDoc(BaseDoc):
            video_url: VideoUrl


        doc = MyDoc(
            video_url='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true'
        )

        video, audio, key_frame_indices = doc.video_url.load()
        assert isinstance(video, VideoNdArray)
        assert isinstance(audio, AudioNdArray)
        assert isinstance(key_frame_indices, NdArray)
        ```

        ---


        :param kwargs: supports all keyword arguments that are being supported by
            av.open() as described [here](https://pyav.org/docs/stable/api/_globals.html?highlight=open#av.open)
        :return: a `VideoLoadResult` instance with video, audio and keyframe indices
        """
        if TYPE_CHECKING:
            import av
        else:
            av = import_library('av')

        with av.open(BytesIO(self), **kwargs) as container:
            audio_frames: List[np.ndarray] = []
            video_frames: List[np.ndarray] = []
            keyframe_indices: List[int] = []

            for frame in container.decode():
                if type(frame) == av.audio.frame.AudioFrame:
                    audio_frames.append(frame.to_ndarray())
                elif type(frame) == av.video.frame.VideoFrame:
                    if frame.key_frame == 1:
                        curr_index = len(video_frames)
                        keyframe_indices.append(curr_index)

                    video_frames.append(frame.to_ndarray(format='rgb24'))

        # Pad audio arrays to same shape if sample rates differ
        if len({arr.shape for arr in audio_frames}) > 1:
            warnings.warn('Audio frames have different sample rates')
            audio_frames = self._pad_arrays_to_same_shape(audio_frames)

        if len(audio_frames) == 0:
            audio = parse_obj_as(AudioNdArray, np.array(audio_frames))
        else:
            audio = parse_obj_as(AudioNdArray, np.stack(audio_frames))

        video = parse_obj_as(VideoNdArray, np.stack(video_frames))
        indices = parse_obj_as(NdArray, keyframe_indices)

        return VideoLoadResult(video=video, audio=audio, key_frame_indices=indices)

    @staticmethod
    def _pad_arrays_to_same_shape(arrays: List[np.ndarray]) -> List[np.ndarray]:
        # Calculate the maximum number of samples in any array
        max_samples = max(arr.shape[1] for arr in arrays)

        # Pad arrays with fewer samples
        for i, arr in enumerate(arrays):
            if arr.shape[1] < max_samples:
                arrays[i] = np.pad(arr, ((0, 0), (0, max_samples - arr.shape[1])))

        return arrays