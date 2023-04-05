# Table

DocArray supports many different modalities including tabular data.
This section will show you how to load and handle tabular data in DocArray.

## Load CSV table

A common way to store tabular data is via `.csv` files.
You can easily load such data from a given `.csv` file into a DocumentArray. 
Let's take a look at the following example file, which includes data about book and their authors and publishing year.

```text
title,author,year
Harry Potter and the Philosopher's Stone,J. K. Rowling,1997
Klara and the sun,Kazuo Ishiguro,2020
A little life,Hanya Yanagihara,2015
```

First we have to define the Document schema describing our data.
```python
from docarray import BaseDoc


class Book(BaseDoc):
    title: str
    author: str
    year: int
```
Next, we can load the content of the csv file to a DocList instance of `Book`s.
```python
from docarray import DocList


docs = DocList[Book].from_csv(file_path='books.csv')
docs.summary()
```
``` { .text .no-copy }
╭────── DocList Summary ──────╮
│                             │
│   Type     DocList[Book]    │
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
The resulting DocList contains three `Book`s, since each row of the csv file corresponds to one book and is assigned to one `Book` instance.


## Save to CSV file

Vice versa, you can also store your DocList data to a `.csv` file.
```python
docs.to_csv(file_path='/path/to/my_file.csv')
```

Tabular data is often not the best choice to represent nested Documents. Hence, nested Documents will be stored flattened and can be accessed by their "__"-separated access paths.

Let's look at an example. We now want to store not only the book data, but moreover book review data. Our `BookReview class` has a nested `book` attribute as well as the non-nested attributes `n_ratings` and `stars`.

```python
class BookReview(BaseDoc):
    book: Book
    n_ratings: int
    stars: float


review_docs = DocList[BookReview](
    [BookReview(book=book, n_ratings=12345, stars=5) for book in docs]
)
review_docs.summary()
```
``` { .text .no-copy}
╭───────── DocList Summary ─────────╮
│                                   │
│   Type     DocList[BookReview]    │
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
As expected all nested attributes will be stored by there access path.
```python
review_docs.to_csv(file_path='/path/to/nested_documents.csv')
```
``` { .text .no-copy}
id,book__id,book__title,book__author,book__year,n_ratings,stars
d6363aa3b78b4f4244fb976570a84ff7,8cd85fea52b3a3bc582cf56c9d612cbb,Harry Potter and the Philosopher's Stone,J. K. Rowling,1997,12345,5.0
5b53fff67e6b6cede5870f2ee09edb05,87b369b93593967226c525cf226e3325,Klara and the sun,Kazuo Ishiguro,2020,12345,5.0
addca0475756fc12cdec8faf8fb10d71,03194cec1b75927c2259b3c0fff1ab6f,A little life,Hanya Yanagihara,2015,12345,5.0

```
