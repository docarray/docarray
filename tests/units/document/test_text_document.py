from docarray.documents import Text


def test_text_document_init():
    text = Text('hello world')
    assert text.text == 'hello world'
    assert text == 'hello world'

    text = Text(text='hello world')
    assert text.text == 'hello world'
    assert text == 'hello world'

    text = Text()
    assert text is not None
    assert text.text is None
