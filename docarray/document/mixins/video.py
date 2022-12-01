import threading
import time
from typing import (
    Union,
    BinaryIO,
    TYPE_CHECKING,
    Generator,
    Type,
    Dict,
    Optional,
    Tuple,
)

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from docarray.typing import T
    from docarray import Document


class VideoDataMixin:
    """Provide helper functions for :class:`Document` to support video data."""

    @classmethod
    def generator_from_webcam(
        cls: Type['T'],
        height_width: Optional[Tuple[int, int]] = None,
        show_window: bool = True,
        window_title: str = 'webcam',
        fps: int = 30,
        exit_key: int = 27,
        exit_event=None,
        tags: Optional[Dict] = None,
    ) -> Generator['T', None, None]:
        """
        Create a generator that yields a :class:`Document` object from the webcam.

        This feature requires the `opencv-python` package.

        :param height_width: the shape of the video frame, if not provided, the shape will be determined from the first frame.
            Note that this is restricted by the hardware of the camera.
        :param show_window: if to show preview window of the webcam video
        :param window_title: the window title of the preview window
        :param fps: expected frames per second, note that this is not guaranteed, as the actual fps depends on the hardware limit
        :param exit_key: the key to press to exit the preview window
        :param exit_event: the multiprocessing/threading/asyncio event that once set to exit the preview window
        :param tags: the tags to attach to the document
        :return: a generator that yields a :class:`Document` object from a webcam
        """
        import cv2

        if exit_event is None:
            exit_event = threading.Event()

        vc = cv2.VideoCapture(0)
        prev_frame_time = time.perf_counter()
        actual_fps = 0

        try:
            while not exit_event.is_set():
                rval, frame = vc.read()
                d = cls(tensor=frame, tags=tags)  # type: Document
                if height_width:
                    d.set_image_tensor_shape(height_width)

                yield d

                key = cv2.waitKey(1000 // (fps + fps - actual_fps))

                if show_window:
                    new_frame_time = time.perf_counter()

                    actual_fps = int(1 / (new_frame_time - prev_frame_time))
                    prev_frame_time = new_frame_time

                    # converting the fps into integer

                    # putting the FPS count on the frame
                    cv2.putText(
                        d.tensor,
                        f'FPS {actual_fps:0.0f}/{fps}',
                        (7, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )

                    # displaying the frame with fps
                    cv2.imshow(window_title, d.tensor)

                if key == exit_key or not rval:
                    break
        finally:
            vc.release()
            if show_window:
                cv2.destroyWindow(window_title)

    def load_uri_to_video_tensor(
        self: 'T', only_keyframes: bool = False, **kwargs
    ) -> 'T':
        """Convert a :attr:`.uri` to a video ndarray :attr:`.tensor`.

        :param only_keyframes: if True keep only the keyframes, if False keep all frames and store the
            indices of the keyframes in :attr:`.tags`
        :param kwargs: supports all keyword arguments that are being supported by av.open() as
            described in: https://pyav.org/docs/stable/api/_globals.html?highlight=open#av.open
        :return: Document itself after processed
        """
        import av

        with av.open(self.uri, **kwargs) as container:
            if only_keyframes:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = 'NONKEY'

            frames = []
            keyframe_indices = []
            for i, frame in enumerate(container.decode(video=0)):

                img = frame.to_image()
                frames.append(img)
                if not only_keyframes and frame.key_frame == 1:
                    keyframe_indices.append(i)

        self.tensor = np.moveaxis(np.stack(frames), 1, 2)
        if not only_keyframes:
            self.tags['keyframe_indices'] = keyframe_indices

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
