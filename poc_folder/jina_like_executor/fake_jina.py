from typing import Callable

from docarray import DocumentArray


class Executor:
    pass


def requests(f: Callable):
    def wrap(self, docs: DocumentArray, *args, **kwargs):
        docs = DocumentArray.from_protobuf(docs.to_protobuf())
        return f(self, docs, *args, **kwargs)

    return wrap
