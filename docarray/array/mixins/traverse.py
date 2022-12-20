from typing import TYPE_CHECKING, Any, List, Union

from docarray.array.abstract_array import AbstractDocumentArray

if TYPE_CHECKING:
    from docarray import Document, DocumentArray


class TraverseMixin(AbstractDocumentArray):
    """
    A mixin used for traversing :class:`DocumentArray`.
    """

    def traverse_flat(
        self: AbstractDocumentArray,
        access_path: str,
    ) -> List[Any]:
        """
        Return a List of the accessed objects when applying the access_path. If this res
        Return a List of the accessed objects when applying the access_path. If this
        results in a nested list or list of DocumentArrays, the list will be flattened
        on the first level. The access path is a string that consists of attribute
        names, concatenated and dot-seperated. It describes the path from the first
        level to an arbitrary one, e.g. 'doc_attr_x.sub_doc_attr_x.sub_sub_doc_attr_z'.

        :param access_path: a string that represents the access path.
        :return: list of the accessed objects, flattened if nested.

        EXAMPLE USAGE
        .. code-block:: python
            from docarray import Document, DocumentArray, Text


            class Author(Document):
                name: str


            class Book(Document):
                author: Author
                content: Text


            da = DocumentArray[Book](
                Book(author=Author(name='Ben'), content=Text(text=f'book_{i}')) for i in range(10)
            )

            books = da.traverse_flat(access_path='content')  # list of 10 Text objs
            authors = da.traverse_flat(access_path='author.name')  # list of 10 strings

        If the resulting list is a nested list, it will be flattened:

        EXAMPLE USAGE
        .. code-block:: python
            from docarray import Document, DocumentArray


            class Chapter(Document):
                content: str


            class Book(Document):
                chapters: DocumentArray[Chapter]


            da = DocumentArray[Book](
                Book(
                    chapters=DocumentArray[Chapter](
                        [Chapter(content='some_content') for _ in range(3)]
                    )
                )
                for _ in range(10)
            )

            chapters = da.traverse_flat(access_path='chapters')  # list of 30 strings

        """
        leaves = list(self._traverse(docs=self, access_path=access_path))
        return self._flatten(leaves)

    @staticmethod
    def _traverse(docs: Union['Document', 'DocumentArray'], access_path: str):
        if access_path:
            path_attrs = access_path.split('.')
            curr_attr = path_attrs[0]
            path_attrs.pop(0)

            from docarray import Document

            if isinstance(docs, Document):
                docs = [docs]

            for d in docs:
                x = getattr(d, curr_attr)
                yield from TraverseMixin._traverse(x, '.'.join(path_attrs))
        else:
            yield docs

    @staticmethod
    def _flatten(sequence) -> 'DocumentArray':
        from docarray import DocumentArray

        res = []
        for seq in sequence:
            if isinstance(seq, (list, DocumentArray)):
                res += seq
            else:
                res.append(seq)

        return res
