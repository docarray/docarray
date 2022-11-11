from fake_jina import Executor, requests

from docarray import DocumentArray, Document, Text
from docarray.document import AnyDocument


class MyDoc(Document):
    text: str


class MyExecutor(Executor):
    @requests
    def index(self, docs: DocumentArray[MyDoc]):
        ## here this work even if the Executor does not know about Text. It just received a schema less ( AnySchema) Document
        for doc_ in docs:
            print(doc_.text)
            assert isinstance(doc_, AnyDocument)


exec = MyExecutor()

exec.index(DocumentArray([Text(text='hello') for _ in range(10)]))
