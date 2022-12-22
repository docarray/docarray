import wave
from typing import BinaryIO, TypeVar, Union

from docarray.typing import NdArray

T = TypeVar('T', bound='AudioNdArray')


class AudioNdArray(NdArray):
    """ """

    def save_audio_tensor_to_file(
        self: 'T',
        file_path: Union[str, BinaryIO],
        sample_rate: int = 44100,
        sample_width: int = 2,
    ) -> None:
        """
        Save :attr:`.tensor` into a .wav file. Mono/stereo is preserved.

        :param file_path: if file is a string, open the file by that name, otherwise
            treat it as a file-like object.
        :param sample_rate: sampling frequency
        :param sample_width: sample width in bytes
        """

        # Convert to (little-endian) 16 bit integers.
        max_int16 = 2**15
        tensor = (self * max_int16).astype('<h')
        n_channels = 2 if self.ndim > 1 else 1

        with wave.open(file_path, 'w') as f:
            # 2 Channels.
            f.setnchannels(n_channels)
            # 2 bytes per sample.
            f.setsampwidth(sample_width)
            f.setframerate(sample_rate)
            f.writeframes(tensor.tobytes())
