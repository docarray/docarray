from docarray import Document, DocumentArray, dataclass
from docarray.typing import Image, Text

@dataclass
class MyDocument:
    image: Image
    paragraph: Text

def test_database_subindices():
    # extend with Documents, including embeddings
    _docs = [(MyDocument( image='https://docarray.jina.ai/_images/apple.png', paragraph='hello world'))]

    da = DocumentArray(
        storage='sqlite',  # use SQLite as vector database
        config={'connection': 'jina4.db', 'table_name': 'test4'},
        subindex_configs={'@.[image]': {'connection': 'jina4.db', 'table_name': 'test5'},\
                          '@.[paragraph]': {'connection': 'jina4.db', 'table_name': 'test6'}},  \
        # set up subindices for image and description
    )
    da.summary()

    for item in _docs:
      d = Document(item)
      da.append(d)

    da = DocumentArray(
        storage='sqlite',  # use SQLite as vector database
        config={'connection': 'jina4.db', 'table_name': 'test4'},
        subindex_configs={'@.[image]': {'connection': 'jina4.db', 'table_name': 'test5'}, \
                          '@.[paragraph]': {'connection': 'jina4.db', 'table_name': 'test6'}},  \
        # set up subindices for image and description
    )
    da.summary()