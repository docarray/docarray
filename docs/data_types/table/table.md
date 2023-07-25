# 📊 Table

DocArray supports many different modalities including tabular data.
This section will show you how to load and handle tabular data using DocArray.

## Load CSV table

A common way to store tabular data is via `CSV` (comma-separated values) files.
You can load such data from a given `CSV` file into a [`DocList`][docarray.DocList]. 

Let's take a look at the following example file, which includes data about books and their authors and year of publication:

```text
title,author,year
Harry Potter and the Philosopher's Stone,J. K. Rowling,1997
Klara and the sun,Kazuo Ishiguro,2020
A little life,Hanya Yanagihara,2015
```

First, define the Document schema describing the data:

```python
from docarray import BaseDoc


class Book(BaseDoc):
    title: str
    author: str
    year: int
```
Next, load the content of the CSV file to a [`DocList`][docarray.DocList] instance of `Book`s via [`.from_csv()`][docarray.array.doc_list.io.IOMixinDocList.from_csv]:

```python
from docarray import DocList


docs = DocList[Book].from_csv(
    file_path='https://github.com/docarray/docarray/blob/main/tests/toydata/books.csv?raw=true'
)
docs.summary()
```

<details>
    <summary>Output</summary>
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
</details>

The resulting [`DocList`][docarray.DocList] object contains three `Book`s, since each row of the CSV file corresponds to one book and is assigned to one `Book` instance.

## Save to CSV file

Vice versa, you can also store your [`DocList`][docarray.DocList] data in a `.csv` file using [`.to_csv()`][docarray.array.doc_list.io.IOMixinDocList.to_csv]:

``` { .python }
docs.to_csv(file_path='/path/to/my_file.csv')
```

Tabular data is often not the best choice to represent nested Documents. Hence, nested Documents will be stored flattened and can be accessed by their `'__'`-separated access paths.

Let's take a look at an example. We now want to store not only the book data but moreover book review data. To do so, we define a `BookReview` class that has a nested `book` attribute as well as the non-nested attributes `n_ratings` and `stars`:

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

<details>
    <summary>Output</summary>
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
</details>

As expected all nested attributes will be stored by their access path:

``` { .python }
review_docs.to_csv(file_path='/path/to/nested_documents.csv')
```

``` { .text .no-copy hl_lines="1" }
id,book__id,book__title,book__author,book__year,n_ratings,stars
d6363aa3b78b4f4244fb976570a84ff7,8cd85fea52b3a3bc582cf56c9d612cbb,Harry Potter and the Philosopher's Stone,J. K. Rowling,1997,12345,5.0
5b53fff67e6b6cede5870f2ee09edb05,87b369b93593967226c525cf226e3325,Klara and the sun,Kazuo Ishiguro,2020,12345,5.0
addca0475756fc12cdec8faf8fb10d71,03194cec1b75927c2259b3c0fff1ab6f,A little life,Hanya Yanagihara,2015,12345,5.0
```

## Handle TSV tables

Not only can you load and save comma-separated values (`CSV`) data, but also tab-separated values (`TSV`), 
by adjusting the `dialect` parameter in [`.from_csv()`][docarray.array.doc_list.io.IOMixinDocList.from_csv] 
and [`.to_csv()`][docarray.array.doc_list.io.IOMixinDocList.to_csv].

The dialect defaults to `'excel'`, which refers to comma-separated values. For tab-separated values, you can use 
`'excel-tab'`.

Let's take a look at what this would look like with a tab-separated file:

```text
title	author	year
Title1	author1	2020
Title2	author2	1234
```

```python
docs = DocList[Book].from_csv(
    file_path='https://github.com/docarray/docarray/blob/main/tests/toydata/books.tsv?raw=true',
    dialect='excel-tab',
)
for doc in docs:
    doc.summary()
```

<details>
    <summary>Output</summary>
    ```text
    📄 Book : c1ac9d4 ...
    ╭──────────────────────┬───────────────╮
    │ Attribute            │ Value         │
    ├──────────────────────┼───────────────┤
    │ title: str           │ Title1        │
    │ author: str          │ author1       │
    │ year: int            │ 2020          │
    ╰──────────────────────┴───────────────╯
    📄 Book : c1ac9d4 ...
    ╭──────────────────────┬───────────────╮
    │ Attribute            │ Value         │
    ├──────────────────────┼───────────────┤
    │ title: str           │ Title1        │
    │ author: str          │ author1       │
    │ year: int            │ 2020          │
    ╰──────────────────────┴───────────────╯
    ```
</details>

Great! All the data is correctly read and stored in `Book` instances.

## Other separators

If your values are separated by yet another separator, you can create your own class that inherits from `csv.Dialect`.
Within this class, you can define your dialect's behavior by setting the provided [formatting parameters](https://docs.python.org/3/library/csv.html#dialects-and-formatting-parameters).

For instance, let's assume you have a semicolon-separated table:

```text
first_name;last_name;year
Jane;Austin;2020
John;Doe;1234
```

Now, let's define our `SemicolonSeparator` class. Next to the `delimiter` parameter, we have to set some more formatting parameters such as `doublequote` and `lineterminator`.

```python
import csv


class SemicolonSeparator(csv.Dialect):
    delimiter = ';'
    doublequote = True
    lineterminator = '\r\n'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL
```

Finally, you can load your data by setting the `dialect` parameter in [`.from_csv()`][docarray.array.doc_list.io.IOMixinDocList.from_csv] to an instance of your `SemicolonSeparator`.

```python
docs = DocList[Book].from_csv(
    file_path='https://github.com/docarray/docarray/blob/main/tests/toydata/books_semicolon_sep.csv?raw=true',
    dialect=SemicolonSeparator(),
)
for doc in docs:
    doc.summary()
```

<details>
    <summary>Output</summary>
    ```text
    📄 Book : 321e9fd ...
    ╭──────────────────────┬───────────────╮
    │ Attribute            │ Value         │
    ├──────────────────────┼───────────────┤
    │ title: str           │ Title1        │
    │ author: str          │ author1       │
    │ year: int            │ 2020          │
    ╰──────────────────────┴───────────────╯
    📄 Book : 16d2097 ...
    ╭──────────────────────┬───────────────╮
    │ Attribute            │ Value         │
    ├──────────────────────┼───────────────┤
    │ title: str           │ Title2        │
    │ author: str          │ author2       │
    │ year: int            │ 1234          │
    ╰──────────────────────┴───────────────╯
    ```
</details>
