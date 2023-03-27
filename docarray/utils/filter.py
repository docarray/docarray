import json
from typing import Dict, List, Union

from docarray.array.abstract_array import AnyDocArray
from docarray.array.array.array import DocumentArray


def filter_docs(
    docs: AnyDocArray,
    query: Union[str, Dict, List[Dict]],
) -> AnyDocArray:
    """
    Filter the Documents in the index according to the given filter query.


    EXAMPLE USAGE

    .. code-block:: python

        from docarray import DocumentArray, BaseDoc
        from docarray.documents import Text, Image
        from docarray.util.filter import filter_docs


        class MyDocument(BaseDoc):
            caption: Text
            image: Image
            price: int


        docs = DocumentArray[MyDocument](
            [
                MyDocument(
                    caption='A tiger in the jungle',
                    image=Image(url='tigerphoto.png'),
                    price=100,
                ),
                MyDocument(
                    caption='A swimming turtle', image=Image(url='turtlepic.png'), price=50
                ),
                MyDocument(
                    caption='A couple birdwatching with binoculars',
                    image=Image(url='binocularsphoto.png'),
                    price=30,
                ),
            ]
        )
        query = {
            '$and': {
                'image__url': {'$regex': 'photo'},
                'price': {'$lte': 50},
            }
        }

        results = filter_docs(docs, query)
        assert len(results) == 1
        assert results[0].price == 30
        assert results[0].caption == 'A couple birdwatching with binoculars'
        assert results[0].image.url == 'binocularsphoto.png'

    :param docs: the DocumentArray where to apply the filter
    :param query: the query to filter by
    :return: A DocumentArray containing the Documents
    in `docs` that fulfill the filter conditions in the `query`
    """
    from docarray.utils.query_language.query_parser import QueryParser

    if query:
        query = query if not isinstance(query, str) else json.loads(query)
        parser = QueryParser(query)
        return DocumentArray.__class_getitem__(docs.document_type)(
            d for d in docs if parser.evaluate(d)
        )
    else:
        return docs
