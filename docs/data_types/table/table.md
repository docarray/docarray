# Table

## Load CSV table

You can easily load tabular data from `csv` file into a DocumentArray. For example, 

```text
title,author,year
Harry Potter and the Philosopher's Stone,J. K. Rowling,1997
Klara and the sun,Kazuo Ishiguro,2020
A little life,Hanya Yanagihara,2015
```

First you have to define the Document schema:
```python
from docarray import BaseDoc, DocArray


class Book(BaseDoc):
    title: str
    author: str
    year: int


da = DocArray[Book].from_csv(
    file_path='/Users/charlottegerhaher/Desktop/jina-ai/docarray_v2/docarray/docs/data_types/table/books.csv'
)
da.summary()
```
```text
╭───── DocArray Summary ──────╮
│                             │
│   Type     DocArray[Book]   │
│   Length   3                │
│                             │
╰─────────────────────────────╯
╭── Document Schema ──╮
│                     │
│   Book              │
│   ├── title: str    │
│   ├── author: str   │
│   └── year: int     │
│                     │
╰─────────────────────╯
```
Each row in the csv file corresponds to one Document.


## Save to CSV file

Saving a DocumentArray as a csv file is easy.
```python
da.to_csv(file_path='/path/to/my_file.csv')
```

Tabular data is often not the best choice to represent nested Documents. Hence, nested Document will be stored in flatten. and can be accessed by their "__"-separated access path:

```python
class BookReview(BaseDoc):
    book: Book
    n_ratings: int
    stars: float


da_reviews = DocArray[BookReview](
    [BookReview(book=book, n_ratings=12345, stars=5) for book in da]
)
da_reviews.summary()
```
```text
╭──────── DocArray Summary ─────────╮
│                                   │
│   Type     DocArray[BookReview]   │
│   Length   3                      │
│                                   │
╰───────────────────────────────────╯
╭──── Document Schema ────╮
│                         │
│   BookReview            │
│   ├── book: Book        │
│   │   ├── title: str    │
│   │   ├── author: str   │
│   │   └── year: int     │
│   ├── n_ratings: int    │
│   └── stars: float      │
│                         │
╰─────────────────────────╯
```
csv file content with nested access paths:
```text
id,book__id,book__title,book__author,book__year,n_ratings,stars
d6363aa3b78b4f4244fb976570a84ff7,8cd85fea52b3a3bc582cf56c9d612cbb,Harry Potter and the Philosopher's Stone,J. K. Rowling,1997,12345,5.0
5b53fff67e6b6cede5870f2ee09edb05,87b369b93593967226c525cf226e3325,Klara and the sun,Kazuo Ishiguro,2020,12345,5.0
addca0475756fc12cdec8faf8fb10d71,03194cec1b75927c2259b3c0fff1ab6f,A little life,Hanya Yanagihara,2015,12345,5.0

```
