(text-type)=
# {octicon}`typography` Text

Representing text in DocArray is easy. Simply do:
```python
from docarray import Document

Document(text='hello, world.')
```

If your text data is big and can not be written inline, or it comes from a URI, then you can also define `uri` first and load the text into Document later.

```python
from docarray import Document

d = Document(uri='https://www.w3.org/History/19921103-hypertext/hypertext/README.html')
d.load_uri_to_text()

d.summary()
```

```text
<Document ('id', 'mime_type', 'text', 'uri') at 3c128f326fbf11ec90821e008a366d49>
```

And of course, you can have characters from different languages.

```python
from docarray import Document

d = Document(text='👋	नमस्ते दुनिया!	你好世界！こんにちは世界！	Привет мир!')
```


## Segment long documents

Often times when you index/search textual document, you don't want to consider thousands of words as one document, some finer granularity would be nice. You can do these by leveraging `chunks` of Document. For example, let's segment this simple document by `!` mark:

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

Which creates five sub-documents under the original documents and stores them under `.chunks`.

## Convert text into `ndarray`

Sometimes you may need to encode the text into a `numpy.ndarray` before further computation. We provide some helper functions in Document and DocumentArray that allow you to convert easily.

For example, we have a DocumentArray with three Documents:
```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='hello world'), 
                    Document(text='goodbye world'),
                    Document(text='hello goodbye')])
```

To get the vocabulary, you can use:

```python
vocab = da.get_vocabulary()
```

```text
{'hello': 2, 'world': 3, 'goodbye': 4}
```

The vocabulary is 2-indexed as `0` is reserved for padding symbol and `1` is reserved for unknown symbol.

One can further use this vocabulary to convert `.text` field into `.tensor` via:

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

When you have text in different length and you want the output `.tensor` to have the same length, you can define `max_length` during converting:

```python
from docarray import Document, DocumentArray

da = DocumentArray([Document(text='a short phrase'), 
                    Document(text='word'), 
                    Document(text='this is a much longer sentence')])
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

You can get also use `.tensors` of DocumentArray to get all tensors in one `ndarray`.

```python
print(da.tensors)
````

```text
[[ 0  0  0  0  0  0  0  2  3  4]
 [ 0  0  0  0  0  0  0  0  0  5]
 [ 0  0  0  0  6  7  2  8  9 10]]
```

## Convert `ndarray` back to text

As a bonus, you can also easily convert an integer `ndarray` back to text based on some given vocabulary. This procedure is often termed as "decoding". 

```python
from docarray import Document, DocumentArray

da = DocumentArray([Document(text='a short phrase'), 
                    Document(text='word'), 
                    Document(text='this is a much longer sentence')])
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


## Simple text matching via feature hashing

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