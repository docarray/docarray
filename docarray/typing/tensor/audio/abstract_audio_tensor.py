import warnings
from abc import ABC
from typing import TYPE_CHECKING, Any, BinaryIO, Dict, TypeVar, Union

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import import_library, is_notebook

if TYPE_CHECKING:
    from docarray.typing.bytes.audio_bytes import AudioBytes

T = TypeVar('T', bound='AbstractAudioTensor')

MAX_INT_16 = 2**15


class AbstractAudioTensor(AbstractTensor, ABC):
    def to_bytes(self) -> 'AudioBytes':
        """
        Convert audio tensor to [`AudioBytes`][docarray.typing.AudioBytes].
        """
        from docarray.typing.bytes.audio_bytes import AudioBytes

        tensor = self.get_comp_backend().to_numpy(self)
        tensor = (tensor * MAX_INT_16).astype('<h')
        return AudioBytes(tensor.tobytes())

    def save(
        self: 'T',
        file_path: Union[str, BinaryIO],
        format: str = 'wav',
        frame_rate: int = 44100,
        sample_width: int = 2,
        pydub_args: Dict[str, Any] = {},
    ) -> None:
        """
        Save audio tensor to an audio file. Mono/stereo is preserved.

        :param file_path: path to an audio file. If file is a string, open the file by
            that name, otherwise treat it as a file-like object.
        :param format: format for the audio file ('mp3', 'wav', 'raw', 'ogg' or other ffmpeg/avconv supported files)
        :param frame_rate: sampling frequency
        :param sample_width: sample width in bytes
        :param pydub_args: dictionary of additional arguments for pydub.AudioSegment.export function
        """
        pydub = import_library('pydub', raise_error=True)  # noqa: F841
        from pydub import AudioSegment

        comp_backend = self.get_comp_backend()
        channels = 2 if comp_backend.n_dim(array=self) > 1 else 1  # type: ignore

        segment = AudioSegment(
            self.to_bytes(),
            frame_rate=frame_rate,
            sample_width=sample_width,
            channels=channels,
        )
        segment.export(file_path, format=format, **pydub_args)

    def display(self, rate=44100):
        """
        Play audio data from tensor in notebook.
        """
        if is_notebook():
            from IPython.display import Audio, display

            audio_np = self.get_comp_backend().to_numpy(self)
            display(Audio(audio_np, rate=rate))
        else:
            warnings.warn('Display of audio is only possible in a notebook.')
