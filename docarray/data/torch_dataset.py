from typing import Callable, Dict, Generic, List, Optional, Type, TypeVar

from torch.utils.data import Dataset

from docarray import BaseDoc, DocList, DocVec
from docarray.typing import TorchTensor
from docarray.utils._internal._typing import change_cls_name, safe_issubclass

T_doc = TypeVar('T_doc', bound=BaseDoc)


class MultiModalDataset(Dataset, Generic[T_doc]):
    """
    A dataset that can be used inside a PyTorch DataLoader.
    In other words, it implements the PyTorch Dataset interface.

    The preprocessing dictionary passed to the constructor consists of keys that are
    field names and values that are functions that take a single argument and return
    a single argument.

    ---

    ```python
    from torch.utils.data import DataLoader
    from docarray import DocList
    from docarray.data import MultiModalDataset
    from docarray.documents import TextDoc


    def prepend_number(text: str):
        return f"Number {text}"


    docs = DocList[TextDoc](TextDoc(text=str(i)) for i in range(16))
    ds = MultiModalDataset[TextDoc](docs, preprocessing={'text': prepend_number})
    loader = DataLoader(ds, batch_size=4, collate_fn=MultiModalDataset[TextDoc].collate_fn)
    for batch in loader:
        print(batch.text)
    ```

    ---

    Nested fields can be accessed by using dot notation.
    The document itself can be accessed using the empty string as the key.

    Transformations that operate on reference types (such as Documents) can optionally
    not return a value.

    The transformations will be applied according to their order in the dictionary.

    ---

    ```python
    import torch
    from torch.utils.data import DataLoader
    from docarray import DocList, BaseDoc
    from docarray.data import MultiModalDataset
    from docarray.documents import TextDoc


    class Thesis(BaseDoc):
        title: TextDoc


    class Student(BaseDoc):
        thesis: Thesis


    def embed_title(title: TextDoc):
        title.embedding = torch.ones(4)


    def normalize_embedding(thesis: Thesis):
        thesis.title.embedding = thesis.title.embedding / thesis.title.embedding.norm()


    def add_nonsense(student: Student):
        student.thesis.title.embedding = student.thesis.title.embedding + int(
            student.thesis.title.text
        )


    docs = DocList[Student](Student(thesis=Thesis(title=str(i))) for i in range(16))
    ds = MultiModalDataset[Student](
        docs,
        preprocessing={
            "thesis.title": embed_title,
            "thesis": normalize_embedding,
            "": add_nonsense,
        },
    )
    loader = DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)
    for batch in loader:
        print(batch.thesis.title.embedding)
    ```

    ---

    :param docs: the `DocList` to be used as the dataset
    :param preprocessing: a dictionary of field names and preprocessing functions
    """

    doc_type: Optional[Type[BaseDoc]] = None
    __typed_ds__: Dict[Type[BaseDoc], Type['MultiModalDataset']] = {}

    def __init__(
        self, docs: 'DocList[T_doc]', preprocessing: Dict[str, Callable]
    ) -> None:
        self.docs = docs
        self._preprocessing = preprocessing

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, item: int):
        doc = self.docs[item].copy(deep=True)
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
        doc_type = cls.doc_type
        if doc_type:
            batch_da = DocVec[doc_type](  # type: ignore
                batch,
                tensor_type=TorchTensor,
            )
        else:
            batch_da = DocVec(batch, tensor_type=TorchTensor)
        return batch_da

    @classmethod
    def __class_getitem__(cls, item: Type[BaseDoc]) -> Type['MultiModalDataset']:
        if not safe_issubclass(item, BaseDoc):
            raise ValueError(
                f'{cls.__name__}[item] item should be a Document not a {item} '
            )

        if item not in cls.__typed_ds__:
            global _TypedDataset

            class _TypedDataset(cls):  # type: ignore
                doc_type = item

            change_cls_name(
                _TypedDataset, f'{cls.__name__}[{item.__name__}]', globals()
            )

            cls.__typed_ds__[item] = _TypedDataset

        return cls.__typed_ds__[item]
