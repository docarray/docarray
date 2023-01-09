from docarray.computation import AbstractComputationalBackend
from docarray.typing.tensor.abstract_tensor import AbstractTensor


class AbstracttensorCompBackend(AbstractComputationalBackend[AbstractTensor]):
    @staticmethod
    def to_device(tensor: 'AbstractTensor', device: str) -> 'AbstractTensor':
        """Move the tensor to the specified device."""
        return tensor.to(device)
