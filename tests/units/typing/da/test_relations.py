from docarray import BaseDoc, DocList


def test_instance_and_equivalence():
    class MyDoc(BaseDoc):
        text: str

    docs = DocList[MyDoc]([MyDoc(text='hello')])

    assert issubclass(DocList[MyDoc], DocList[MyDoc])
    assert issubclass(docs.__class__, DocList[MyDoc])

    assert isinstance(docs, DocList[MyDoc])


def test_subclassing():
    class MyDoc(BaseDoc):
        text: str

    class MyDocList(DocList[MyDoc]):
        pass

    docs = MyDocList([MyDoc(text='hello')])

    assert issubclass(MyDocList, DocList[MyDoc])
    assert issubclass(docs.__class__, DocList[MyDoc])

    assert isinstance(docs, MyDocList)
    assert isinstance(docs, DocList[MyDoc])

    assert issubclass(MyDoc, BaseDoc)
    assert not issubclass(DocList[MyDoc], DocList[BaseDoc])
    assert not issubclass(MyDocList, DocList[BaseDoc])
