from abc import ABC, abstractmethod

from docarray.typing.tensor.abstract_tensor import AbstractTensor


class AbstractImageTensor(AbstractTensor, ABC):
    @abstractmethod
    def to_bytes(self):
        """
        Convert image tensor to bytes.
        """
        ...
