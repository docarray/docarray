from typing import Callable, Dict, Generic, List, Optional, Type, TypeVar

from torch.utils.data import Dataset
from typing_inspect import is_union_type

from docarray import BaseDocument, DocumentArray
from docarray.typing import TorchTensor
from docarray.utils._typing import change_cls_name

T_doc = TypeVar('T_doc', bound=BaseDocument)


class MultiModalDataset(Dataset, Generic[T_doc]):
    """
    A dataset that can be used inside a PyTorch DataLoader.
    In other words, it implements the PyTorch Dataset interface.

    :param da: the DocumentArray to be used as the dataset
    :param preprocessing: a dictionary of field names and preprocessing functions

    The preprocessing dictionary passed to the constructor consists of keys that are
    field names and values that are functions that take a single argument and return
    a single argument.

    EXAMPLE USAGE
    .. code-block:: python
    from torch.utils.data import DataLoader
    from docarray import DocumentArray
    from docarray.data import MultiModalDataset
    from docarray.documents import Text


    def prepend_number(text: str):
        return f"Number {text}"


    da = DocumentArray[Text](Text(text=str(i)) for i in range(16))
    ds = MultiModalDataset[Text](da, preprocessing={'text': prepend_number})
    loader = DataLoader(ds, batch_size=4, collate_fn=MultiModalDataset[Text].collate_fn)
    for batch in loader:
        print(batch.text)

    Nested fields can be accessed by using dot notation.
    The document itself can be accessed using the empty string as the key.

    Transformations that operate on reference types (such as Documents) can optionally
    not return a value.

    The transformations will be applied according to their order in the dictionary.

    EXAMPLE USAGE
    .. code-block:: python
    import torch
    from torch.utils.data import DataLoader
    from docarray import DocumentArray, BaseDocument
    from docarray.data import MultiModalDataset
    from docarray.documents import Text


    class Thesis(BaseDocument):
        title: Text


    class Student(BaseDocument):
        thesis: Thesis


    def embed_title(title: Text):
        title.embedding = torch.ones(4)


    def normalize_embedding(thesis: Thesis):
        thesis.title.embedding = thesis.title.embedding / thesis.title.embedding.norm()


    def add_nonsense(student: Student):
        student.thesis.title.embedding = student.thesis.title.embedding + int(
            student.thesis.title.text
        )


    da = DocumentArray[Student](Student(thesis=Thesis(title=str(i))) for i in range(16))
    ds = MultiModalDataset[Student](
        da,
        preprocessing={
            "thesis.title": embed_title,
            "thesis": normalize_embedding,
            "": add_nonsense,
        },
    )
    loader = DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)
    for batch in loader:
        print(batch.thesis.title.embedding)
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
            global _TypedDataset

            class _TypedDataset(cls):  # type: ignore
                document_type = item

            change_cls_name(
                _TypedDataset, f'{cls.__name__}[{item.__name__}]', globals()
            )

            cls.__typed_ds__[item] = _TypedDataset

        MultiModalDataset._init_typed_das(item)
        return cls.__typed_ds__[item]

    @staticmethod
    def _init_typed_das(item: Type[BaseDocument]):
        # Leaf Node
        if not isinstance(item, type) or not issubclass(item, BaseDocument):
            return

        # Document
        DocumentArray[item]().stack()  # type: ignore
        for field_type in item.__annotations__.values():
            if is_union_type(field_type):
                for union_type in field_type.__args__:
                    MultiModalDataset._init_typed_das(union_type)
            MultiModalDataset._init_typed_das(field_type)
