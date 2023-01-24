from typing import Callable, Dict, Generic, List, Optional, Type, TypeVar

from docarray import BaseDocument, DocumentArray
from docarray.typing import TorchTensor

T_doc = TypeVar('T_doc', bound=BaseDocument)


class TorchDataset(Generic[T_doc]):
    """Torch Dataset from DocumentArray"""

    document_type: Optional[Type[BaseDocument]] = None

    def __init__(
        self, da: 'DocumentArray[T_doc]', preprocessing: Dict[str, Callable]
    ) -> None:
        self.da = da
        self._preprocessing = preprocessing

    def __len__(self):
        return len(self.da)

    def __getitem__(self, item: int):
        doc = self.da[item].copy(deep=True)
        for field, preprocess in self._preprocessing.items():
            preprocess(doc.__getattribute__(field))
        return doc

    @classmethod
    def collate_fn(cls, batch: List[T_doc]):
        doc_type = cls.document_type
        if doc_type:
            batch_da = DocumentArray[doc_type](  # type: ignore
                batch,
                tensor_type=TorchTensor,
            )
        else:
            batch_da = DocumentArray(batch, tensor_type=TorchTensor)
        return batch_da.stack()

    @classmethod
    def __class_getitem__(cls, item: Type[BaseDocument]) -> Type['TorchDataset']:
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'{cls.__name__}[item] item should be a Document not a {item} '
            )

        class _TypedDataset(cls):  # type: ignore
            document_type = item

        return _TypedDataset
