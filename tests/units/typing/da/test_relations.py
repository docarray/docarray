from docarray import BaseDoc, DocArray


def test_instance_and_equivalence():
    class MyDoc(BaseDoc):
        text: str

    docs = DocArray[MyDoc]([MyDoc(text='hello')])

    assert issubclass(DocArray[MyDoc], DocArray[MyDoc])
    assert issubclass(docs.__class__, DocArray[MyDoc])

    assert isinstance(docs, DocArray[MyDoc])


def test_subclassing():
    class MyDoc(BaseDoc):
        text: str

    class MyDocArray(DocArray[MyDoc]):
        pass

    docs = MyDocArray([MyDoc(text='hello')])

    assert issubclass(MyDocArray, DocArray[MyDoc])
    assert issubclass(docs.__class__, DocArray[MyDoc])

    assert isinstance(docs, MyDocArray)
    assert isinstance(docs, DocArray[MyDoc])

    assert issubclass(MyDoc, BaseDoc)
    assert not issubclass(DocArray[MyDoc], DocArray[BaseDoc])
    assert not issubclass(MyDocArray, DocArray[BaseDoc])
