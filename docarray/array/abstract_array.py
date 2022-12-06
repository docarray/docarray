from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Type, Union

from docarray.document import BaseDocument

if TYPE_CHECKING:
    from docarray.typing import NdArray, TorchTensor


class AbstractDocumentArray(Sequence):

    document_type: Type[BaseDocument]
    _columns: Optional[
        Dict[str, Union['TorchTensor', 'AbstractDocumentArray', 'NdArray', None]]
    ]  # here columns are the holder of the data in tensor modes

    @abstractmethod
    def __init__(self, docs: Iterable[BaseDocument]):
        ...

    @abstractmethod
    def __class_getitem__(
        cls, item: Type[BaseDocument]
    ) -> Type['AbstractDocumentArray']:
        ...

    @abstractmethod
    def is_stacked(self) -> bool:
        ...

    @abstractmethod
    def _column_fields(self) -> List[str]:
        ...
