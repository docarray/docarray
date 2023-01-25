from typing import Callable, Dict, Generic, List, Optional, Type, TypeVar

from docarray import BaseDocument, DocumentArray
from docarray.typing import TorchTensor

T_doc = TypeVar('T_doc', bound=BaseDocument)


class MultiModalDataset(Generic[T_doc]):
    """
    A dataset that can be used inside a PyTorch DataLoader.
    In other words, it implements the PyTorch Dataset interface.

    :param da: the DocumentArray to be used as the dataset
    :param preprocessing: a dictionary of field names and preprocessing functions
    """

    document_type: Optional[Type[BaseDocument]] = None
    __typed_ds__: Dict[Type[BaseDocument], Type['MultiModalDataset']] = {}

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
            if len(field) == 0:
                doc = preprocess(doc) or doc
            else:
                acc_path = field.split('.')
                _field_ref = doc
                for attr in acc_path[:-1]:
                    _field_ref = getattr(_field_ref, attr)
                attr = acc_path[-1]
                value = getattr(_field_ref, attr)
                setattr(_field_ref, attr, preprocess(value) or value)
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
    def __class_getitem__(cls, item: Type[BaseDocument]) -> Type['MultiModalDataset']:
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'{cls.__name__}[item] item should be a Document not a {item} '
            )

        if item not in cls.__typed_ds__:

            class _TypedDataset(cls):  # type: ignore
                document_type = item

            cls.__typed_ds__[item] = _TypedDataset

        return cls.__typed_ds__[item]
