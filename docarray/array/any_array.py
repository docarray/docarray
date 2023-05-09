import random
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np

from docarray.array.any_collections import AnyCollections
from docarray.base_doc import BaseDoc
from docarray.display.document_array_summary import DocArraySummary
from docarray.typing.abstract_type import AbstractType

if TYPE_CHECKING:
    from docarray.typing.tensor.abstract_tensor import AbstractTensor

T = TypeVar('T', bound='AnyDocArray')
T_doc = TypeVar('T_doc', bound=BaseDoc)
IndexIterType = Union[slice, Iterable[int], Iterable[bool], None]


class AnyDocArray(AnyCollections[T_doc], AbstractType):
    doc_type: Type[BaseDoc]
    __typed_da__: Dict[Type['AnyCollections'], Dict[Type[BaseDoc], Type]] = {}

    @overload
    def __getitem__(self: T, item: int) -> T_doc:
        ...

    @overload
    def __getitem__(self: T, item: IndexIterType) -> T:
        ...

    @abstractmethod
    def __getitem__(self, item: Union[int, IndexIterType]) -> Union[T_doc, T]:
        ...

    def __getattr__(self, item: str):
        # Needs to be explicitly defined here for the purpose to disable PyCharm's complaints
        # about not detected properties: https://youtrack.jetbrains.com/issue/PY-47991
        return super().__getattribute__(item)

    @abstractmethod
    def traverse_flat(
        self: 'AnyDocArray',
        access_path: str,
    ) -> Union[List[Any], 'AbstractTensor']:
        """
        Return a List of the accessed objects when applying the `access_path`. If this
        results in a nested list or list of [`DocList`s][docarray.array.doc_list.doc_list.DocList], the list will be flattened
        on the first level. The access path is a string that consists of attribute
        names, concatenated and `"__"`-separated. It describes the path from the first
        level to an arbitrary one, e.g. `'content__image__url'`.


        ```python
        from docarray import BaseDoc, DocList, Text


        class Author(BaseDoc):
            name: str


        class Book(BaseDoc):
            author: Author
            content: Text


        docs = DocList[Book](
            Book(author=Author(name='Jenny'), content=Text(text=f'book_{i}'))
            for i in range(10)  # noqa: E501
        )

        books = docs.traverse_flat(access_path='content')  # list of 10 Text objs

        authors = docs.traverse_flat(access_path='author__name')  # list of 10 strings
        ```

        If the resulting list is a nested list, it will be flattened:

        ```python
        from docarray import BaseDoc, DocList


        class Chapter(BaseDoc):
            content: str


        class Book(BaseDoc):
            chapters: DocList[Chapter]


        docs = DocList[Book](
            Book(chapters=DocList[Chapter]([Chapter(content='some_content') for _ in range(3)]))
            for _ in range(10)
        )

        chapters = docs.traverse_flat(access_path='chapters')  # list of 30 strings
        ```

        If your [`DocList`][docarray.array.doc_list.doc_list.DocList] is in doc_vec mode, and you want to access a field of
        type `AnyTensor`, the doc_vec tensor will be returned instead of a list:

        ```python
        class Image(BaseDoc):
            tensor: TorchTensor[3, 224, 224]


        batch = DocList[Image](
            [
                Image(
                    tensor=torch.zeros(3, 224, 224),
                )
                for _ in range(2)
            ]
        )

        batch_stacked = batch.stack()
        tensors = batch_stacked.traverse_flat(
            access_path='tensor'
        )  # tensor of shape (2, 3, 224, 224)
        ```

        :param access_path: a string that represents the access path ("__"-separated).
        :return: list of the accessed objects, flattened if nested.
        """
        ...

    @staticmethod
    def _traverse(node: Any, access_path: str):
        if access_path:
            curr_attr, _, path_attrs = access_path.partition('__')

            from docarray.array import DocList

            if isinstance(node, (DocList, list)):
                for n in node:
                    x = getattr(n, curr_attr)
                    yield from AnyDocArray._traverse(x, path_attrs)
            else:
                x = getattr(node, curr_attr)
                yield from AnyDocArray._traverse(x, path_attrs)
        else:
            yield node

    @staticmethod
    def _flatten_one_level(sequence: List[Any]) -> List[Any]:
        from docarray import DocList

        if len(sequence) == 0 or not isinstance(sequence[0], (list, DocList)):
            return sequence
        else:
            return [item for sublist in sequence for item in sublist]

    def summary(self):
        """
        Print a summary of this [`DocList`][docarray.array.doc_list.doc_list.DocList] object and a summary of the schema of its
        Document type.
        """
        DocArraySummary(self).summary()

    def _batch(
        self: T,
        batch_size: int,
        shuffle: bool = False,
        show_progress: bool = False,
    ) -> Generator[T, None, None]:
        """
        Creates a `Generator` that yields [`DocList`][docarray.array.doc_list.doc_list.DocList] of size `batch_size`.
        Note, that the last batch might be smaller than `batch_size`.

        :param batch_size: Size of each generated batch.
        :param shuffle: If set, shuffle the Documents before dividing into minibatches.
        :param show_progress: if set, show a progress bar when batching documents.
        :yield: a Generator of [`DocList`][docarray.array.doc_list.doc_list.DocList], each in the length of `batch_size`
        """
        from rich.progress import track

        if not (isinstance(batch_size, int) and batch_size > 0):
            raise ValueError(
                f'`batch_size` should be a positive integer, received: {batch_size}'
            )

        N = len(self)
        indices = list(range(N))
        n_batches = int(np.ceil(N / batch_size))

        if shuffle:
            random.shuffle(indices)

        for i in track(
            range(n_batches),
            description='Batching documents',
            disable=not show_progress,
        ):
            yield self[indices[i * batch_size : (i + 1) * batch_size]]

    @abstractmethod
    def __iter__(self) -> Iterator[T_doc]:
        ...
