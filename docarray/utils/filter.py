import json

from typing import Union, Dict, List


from docarray.array.abstract_array import AnyDocumentArray
from docarray.array.array import DocumentArray


def filter(
    docs: AnyDocumentArray,
    query: Union[str, Dict, List[Dict]],
) -> AnyDocumentArray:
    """
    Filter the Documents in the index according to the given filter query.


    EXAMPLE USAGE

    .. code-block:: python

        from docarray import DocumentArray, BaseDocument
        from docarray.documents import Text, Image
        from docarray.util.filter import filter


        class MyDocument(BaseDocument):
            caption: Text
            image: Image
            price: int


        docs = DocumentArray[MyDocument](
            [MyDocument(caption='A tiger in the jungle',
            image=Image(url='tigerphoto.png'), price=100),
            MyDocument(caption='A swimming turtle',
            image=Image(url='turtlepic.png'), price=50),
            MyDocument(caption='A couple birdwatching with binoculars',
            image=Image(url='binocularsphoto.png'), price=30)]
        )
        query = {
            '$and': {
                'image.url': {'$regex': 'photo'},
                'price': {'$lte': 50},
            }
        }

        results = filter(docs, query)
        assert len(results) == 1
        assert results[0].price == 30
        assert results[0].caption == 'A couple birdwatching with binoculars'
        assert results[0].image.url == 'binocularsphoto.png'

    :param docs: the DocumentArray where to apply the filter
    :param query: the query to filter by
    :return: A DocumentArray containing the Documents
    inside DocumentArray that fullfil the filter conditions
    """
    from docarray.utils.query_parser import QueryParser

    if query:
        query = query if not isinstance(query, str) else json.loads(query)
        parser = QueryParser(query)
        return DocumentArray(d for d in docs if parser.evaluate(d))
    else:
        return docs
