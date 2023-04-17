(text-type)=
# {octicon}`typography` Text

Representing text in DocArray is as easy as:
```python
from docarray import Document

Document(text='hello, world.')
```

If your text data is larger and can't be written inline, or comes from a URI, then you can also define `uri` first and load the text into a Document later:

```python
from docarray import Document

d = Document(uri='https://www.w3.org/History/19921103-hypertext/hypertext/README.html')
d.load_uri_to_text()

d.summary()
```

```text
<Document ('id', 'mime_type', 'text', 'uri') at 3c128f326fbf11ec90821e008a366d49>
```

And of course, you can use characters from different languages:

```python
from docarray import Document

d = Document(text='👋	नमस्ते दुनिया!	你好世界！こんにちは世界！	Привет мир!')
```


## Segment long Documents

Often times when you index/search textual Documents, you don't want to consider thousands of words as one huge Document -- some finer granularity would be nice. You can do this by leveraging Document `chunks`. For example, let's split this simple Document at each `!` mark:

```python
from docarray import Document

d = Document(text='👋	नमस्ते दुनिया!	你好世界!こんにちは世界!	Привет мир!')

d.chunks.extend([Document(text=c) for c in d.text.split('!')])

d.summary()
```

```text
 <Document ('id', 'mime_type', 'text', 'chunks') at 5a12d7a86fbf11ec99a21e008a366d49>
    └─ chunks
          ├─ <Document ('id', 'mime_type', 'text') at 5a12e2346fbf11ec99a21e008a366d49>
          ├─ <Document ('id', 'mime_type', 'text') at 5a12e2f26fbf11ec99a21e008a366d49>
          ├─ <Document ('id', 'mime_type', 'text') at 5a12e3886fbf11ec99a21e008a366d49>
          ├─ <Document ('id', 'mime_type', 'text') at 5a12e41e6fbf11ec99a21e008a366d49>
          └─ <Document ('id',) at 5a12e4966fbf11ec99a21e008a366d49>
```

This creates five sub-Documents under the original Document and stores them under the original Document's `.chunks`.

## Convert text to `ndarray`

Sometimes you need to encode the text into a `numpy.ndarray` before further computation. We provide some helper functions in Document and DocumentArray that allow you to do that easily.

For example, we have a DocumentArray with three Documents:
```python
from docarray import DocumentArray, Document

da = DocumentArray(
    [
        Document(text='hello world'),
        Document(text='goodbye world'),
        Document(text='hello goodbye'),
    ]
)
```

To get the vocabulary, you can use:

```python
vocab = da.get_vocabulary()
```

```text
{'hello': 2, 'world': 3, 'goodbye': 4}
```

The vocabulary is 2-indexed as `0` is reserved for the padding symbol and `1` for the unknown symbol.

You can further use this vocabulary to convert `.text` field into `.tensor`:

```python
for d in da:
    d.convert_text_to_tensor(vocab)
    print(d.tensor)
```

```text
[2 3]
[4 3]
[2 4]
```

When you have text of different lengths and want output `.tensor`s to have the same length, you can define `max_length` during conversion:

```python
from docarray import Document, DocumentArray

da = DocumentArray(
    [
        Document(text='a short phrase'),
        Document(text='word'),
        Document(text='this is a much longer sentence'),
    ]
)
vocab = da.get_vocabulary()

for d in da:
    d.convert_text_to_tensor(vocab, max_length=10)
    print(d.tensor)
```

```text
[0 0 0 0 0 0 0 2 3 4]
[0 0 0 0 0 0 0 0 0 5]
[ 0  0  0  0  6  7  2  8  9 10]
```

You can get also use a DocumentArray's `.tensors` to get all tensors in one `ndarray`.

```python
print(da.tensors)
```

```text
[[ 0  0  0  0  0  0  0  2  3  4]
 [ 0  0  0  0  0  0  0  0  0  5]
 [ 0  0  0  0  6  7  2  8  9 10]]
```

## Convert `ndarray` back to text

As a bonus, you can also easily convert an integer `ndarray` back to text based on a given vocabulary. This is often termed "decoding". 

```python
from docarray import Document, DocumentArray

da = DocumentArray(
    [
        Document(text='a short phrase'),
        Document(text='word'),
        Document(text='this is a much longer sentence'),
    ]
)
vocab = da.get_vocabulary()

# encoding
for d in da:
    d.convert_text_to_tensor(vocab, max_length=10)

# decoding
for d in da:
    d.convert_tensor_to_text(vocab)
    print(d.text)
```

```text
a short phrase
word
this is a much longer sentence
```


## Simple text matching with feature hashing

Let's search for `"she entered the room"` in *Pride and Prejudice*:

```python
from docarray import Document, DocumentArray

d = Document(uri='https://www.gutenberg.org/files/1342/1342-0.txt').load_uri_to_text()
da = DocumentArray(Document(text=s.strip()) for s in d.text.split('\n') if s.strip())
da.apply(lambda d: d.embed_feature_hashing())

q = (
    Document(text='she entered the room')
    .embed_feature_hashing()
    .match(da, limit=5, exclude_self=True, metric='jaccard', use_scipy=True)
)

print(q.matches[:, ('text', 'scores__jaccard')])
```

```text
[['staircase, than she entered the breakfast-room, and congratulated', 
'of the room.', 
'She entered the room with an air more than usually ungracious,', 
'entered the breakfast-room, where Mrs. Bennet was alone, than she', 
'those in the room.'], 
[{'value': 0.6, 'ref_id': 'f47f7448709811ec960a1e008a366d49'}, 
{'value': 0.6666666666666666, 'ref_id': 'f47f7448709811ec960a1e008a366d49'}, 
{'value': 0.6666666666666666, 'ref_id': 'f47f7448709811ec960a1e008a366d49'}, 
{'value': 0.6666666666666666, 'ref_id': 'f47f7448709811ec960a1e008a366d49'}, 
{'value': 0.7142857142857143, 'ref_id': 'f47f7448709811ec960a1e008a366d49'}]]
```


## Searching at chunk level with subindex

You can create applications that search at chunk level using a subindex. 
Imagine you want an application that searches at a sentence granularity and returns the title of the Document
containing the closest sentence to the query. For example, you have a database of song lyrics and want to
search a title from which you remember a small part of the lyrics (like the chorus).

```{admonition} Multi-modal Documents
:class: seealso

Modelling nested Documents is often more convenient using DocArray's {ref}`dataclass API <dataclass>`, especially when multiple modalities are
involved.

You can find the corresponding example {ref}`here <multimodal-example>`.
```

```python
song1_title = 'Old MacDonald Had a Farm'

song1 = """
Old MacDonald had a farm, E-I-E-I-O
And on that farm he had some dogs, E-I-E-I-O
With a bow-wow here, and a bow-wow there,
Here a bow, there a bow, everywhere a bow-wow.
"""

song2_title = 'Ode an die Freude'

song2 = """
Freude, schöner Götterfunken,
Tochter aus Elisium,
Wir betreten feuertrunken
Himmlische, dein Heiligthum.
Deine Zauber binden wieder,
was der Mode Schwerd getheilt;
Bettler werden Fürstenbrüder,
wo dein sanfter Flügel weilt.
"""
```

We can create one Document for each song, containing the song's lines as chunks:

```python
from docarray import Document, DocumentArray

doc1 = Document(
    chunks=[Document(text=line) for line in song1.split('\n')], song_title=song1_title
)
doc2 = Document(
    chunks=[Document(text=line) for line in song2.split('\n')], song_title=song2_title
)
da = DocumentArray()
da.extend([doc1, doc2])
```

Now we can build a feature vector for each line of each song. Here we use a very simple Bag of Words descriptor as 
the feature vector.

```python
import re


def build_tokenizer(token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall


def bow_feature_vector(d, vocab, tokenizer):
    embedding = np.zeros(len(vocab) + 2)
    tokens = tokenizer(d.text)
    for token in tokens:
        if token in vocab:
            embedding[vocab.get(token)] += 1

    return embedding


tokenizer = build_tokenizer()
vocab = da['@c'].get_vocabulary()
for d in da['@c']:
    d.embedding = bow_feature_vector(d, vocab, tokenizer)
```

Once we've prepared the data, we can store it in a DocumentArray that supports a subindex:

```buildoutcfg
n_features = len(vocab)+2
n_dim = 3
da_backend=DocumentArray(
    storage='annlite',
    config={'data_path':'./annlite_data',
            'n_dim': n_dim, 
            'metric': 'Cosine'},
    subindex_configs={'@c': {'n_dim': n_features}},
)

with da_backend:
    da_backend.extend(da)
```

Given a query like `into death` we want to search songs that contain a similar sentence.

```python
def find_song_name_from_song_snippet(query: Document, da_backend) -> str:
    similar_items = da_backend.find(query=query, on='@c', limit=10)[0]
    most_similar_docs = similar_items[0]
    return da_backend[most_similar_docs.parent_id].tags


query = Document(text='farm')
query.embedding = bow_feature_vector(query, vocab, tokenizer)

similar_items = find_song_name_from_song_snippet(query, da_backend)
print(similar_items)
```
This prints:
```text
{'song_title': 'Old MacDonald Had a Farm'}
```
