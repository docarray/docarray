from docarray import DocumentArray, Document


def test_document_array():
    class Text(Document):
        text: str

    da = DocumentArray([Text(text='hello') for _ in range(10)])


def test_document_array_fixed_type():
    class Text(Document):
        text: str

    da = DocumentArray[Text]([Text(text='hello') for _ in range(10)])
