from typing import Generic, List, Sequence, Type, TypeVar, Union

from docarray import BaseDocument, DocumentArray
from docarray.storage.abstract_doc_store import BaseDocumentStore, FindResult
from docarray.typing import AnyTensor
from docarray.utils.find import find, find_batched

TSchema = TypeVar('TSchema', bound=BaseDocument)


class MemoryDocumentStore(BaseDocumentStore, Generic[TSchema]):
    """In-memory document store"""

    def __init__(self):
        super().__init__()
        # TODO(johannes) not sure why the line below doesn't type check
        self._docs = DocumentArray[self._schema]([])  # type: ignore

    def python_type_to_db_type(self, python_type: Type) -> Type:
        return python_type

    def index(self, docs: Union[TSchema, Sequence[TSchema]]):
        self._docs.extend(docs)

    def find(
        self,
        query: Union[AnyTensor, BaseDocument],
        embedding_field: str = 'embedding',
        metric: str = 'cosine_sim',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:
        return find(
            index=self._docs,
            query=query,
            embedding_field=embedding_field,
            metric=metric,
            limit=limit,
            **kwargs,
        )

    def find_batched(
        self,
        query: Union[AnyTensor, DocumentArray],
        embedding_field: str = 'embedding',
        metric: str = 'cosine_sim',
        limit: int = 10,
        **kwargs,
    ) -> List[FindResult]:
        return find_batched(
            index=self._docs,
            query=query,
            embedding_field=embedding_field,
            metric=metric,
            limit=limit,
            **kwargs,
        )
