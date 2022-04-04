from typing import Union, BinaryIO, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...typing import T


class VideoDataMixin:
    """Provide helper functions for :class:`Document` to support video data."""

    def load_uri_to_video_tensor(self: 'T', only_keyframes: bool = False) -> 'T':
        """Convert a :attr:`.uri` to a video ndarray :attr:`.tensor`.

        :param only_keyframes: only keep the keyframes in the video
        :return: Document itself after processed
        """
        import av

        with av.open(self.uri) as container:
            if only_keyframes:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = 'NONKEY'

            frames = []
            for frame in container.decode(video=0):
                img = frame.to_image()
                frames.append(np.asarray(img))

        self.tensor = np.moveaxis(np.stack(frames), 1, 2)
        return self

    def save_video_tensor_to_file(
        self: 'T', file: Union[str, BinaryIO], frame_rate: int = 30, codec: str = 'h264'
    ) -> 'T':
        """Save :attr:`.tensor` as a video mp4/h264 file.

        :param file: The file to open, which can be either a string or a file-like object.
        :param frame_rate: frames per second
        :param codec: the name of a decoder/encoder
        :return: itself after processed
        """
        if (
            self.tensor.ndim != 4
            or self.tensor.shape[-1] != 3
            or self.tensor.dtype != np.uint8
        ):
            raise ValueError(
                f'expects `.tensor` with dtype=uint8 and ndim=4 and the last dimension is 3, '
                f'but receiving {self.tensor.shape} in {self.tensor.dtype}'
            )

        video_tensor = np.moveaxis(np.clip(self.tensor, 0, 255), 1, 2)

        import av

        with av.open(file, mode='w') as container:
            stream = container.add_stream(codec, rate=frame_rate)
            stream.width = self.tensor.shape[1]
            stream.height = self.tensor.shape[2]
            stream.pix_fmt = 'yuv420p'

            for b in video_tensor:
                frame = av.VideoFrame.from_ndarray(b, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
        return self
