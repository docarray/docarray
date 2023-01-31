from docarray import BaseDocument, DocumentArray


def test_instance_and_equivalence():
    class MyDoc(BaseDocument):
        text: str

    docs = DocumentArray[MyDoc]([MyDoc(text='hello')])

    assert issubclass(DocumentArray[MyDoc], DocumentArray[MyDoc])
    assert issubclass(docs.__class__, DocumentArray[MyDoc])

    assert isinstance(docs, DocumentArray[MyDoc])


def test_subclassing():
    class MyDoc(BaseDocument):
        text: str

    class MyDocArray(DocumentArray[MyDoc]):
        pass

    docs = MyDocArray([MyDoc(text='hello')])

    assert issubclass(MyDocArray, DocumentArray[MyDoc])
    assert issubclass(docs.__class__, DocumentArray[MyDoc])

    assert isinstance(docs, MyDocArray)
    assert isinstance(docs, DocumentArray[MyDoc])

    assert issubclass(MyDoc, BaseDocument)
    assert not issubclass(DocumentArray[MyDoc], DocumentArray[BaseDocument])
    assert not issubclass(MyDocArray, DocumentArray[BaseDocument])
