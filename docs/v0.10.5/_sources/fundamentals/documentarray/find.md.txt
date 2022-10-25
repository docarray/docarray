(find-documentarray)=
# Query by Conditions

We can use {meth}`~docarray.array.mixins.find.FindMixin.find` to select Documents from a DocumentArray based the conditions specified in a `query` object. One can use `da.find(query)` to filter Documents and get nearest neighbours from `da`:

- To filter Documents, the `query` object is a Python dictionary object that defines the filtering conditions using a [MongoDB](https://docs.mongodb.com/manual/reference/operator/query/)-like query language.
- To find nearest neighbours, the `query` object needs to be a NdArray-like, a Document, or a DocumentArray object that defines embedding. One can also use `.match()` function for this purpose, and there is a minor interface difference between these two functions, which will be described {ref}`in the next chapter<match-documentarray>`.

Let's see some examples in action. First, let's prepare a DocumentArray we will use.

```python
from jina import Document, DocumentArray

da = DocumentArray([Document(text='journal', weight=25, tags={'h': 14, 'w': 21, 'uom': 'cm'}, modality='A'),
                    Document(text='notebook', weight=50, tags={'h': 8.5, 'w': 11, 'uom': 'in'}, modality='A'),
                    Document(text='paper', weight=100, tags={'h': 8.5, 'w': 11, 'uom': 'in'}, modality='D'),
                    Document(text='planner', weight=75, tags={'h': 22.85, 'w': 30, 'uom': 'cm'}, modality='D'),
                    Document(text='postcard', weight=45, tags={'h': 10, 'w': 15.25, 'uom': 'cm'}, modality='A')])

da.summary()
```

```text
                            Documents Summary                            
                                                                         
  Length                 5                                               
  Homogenous Documents   True                                            
  Common Attributes      ('id', 'text', 'tags', 'weight', 'modality')  
                                                                         
                     Attributes Summary                     
                                                            
  Attribute   Data type   #Unique values   Has empty value  
 ────────────────────────────────────────────────────────── 
  id          ('str',)    5                False            
  weight      ('int',)    5                False            
  modality    ('str',)    2                False            
  tags        ('dict',)   5                False            
  text        ('str',)    5                False            
```

## Filter with query operators

A query filter document can use the query operators to specify conditions in the following form:

```text
{ <field1>: { <operator1>: <value1> }, ... }
```

Here `field1` is {ref}`any field name<doc-fields>` of a Document object.  To access nested fields, one can use the dunder expression. For example, `tags__timestamp` is to access `doc.tags['timestamp']` field.

`value1` can be either a user given Python object, or a substitution field with curly bracket `{field}`   

Finally, `operator1` can be one of the following:

| Query Operator | Description                                                                                                |
|----------------|------------------------------------------------------------------------------------------------------------|
| `$eq`          | Equal to (number, string)                                                                                  |
| `$ne`          | Not equal to (number, string)                                                                              |
| `$gt`          | Greater than (number)                                                                                      |
| `$gte`         | Greater than or equal to (number)                                                                          |
| `$lt`          | Less than (number)                                                                                         |
| `$lte`         | Less than or equal to (number)                                                                             |
| `$in`          | Is in an array                                                                                             |
| `$nin`         | Not in an array                                                                                            |
| `$regex`       | Match the specified regular expression                                                                     |
| `$size`        | Match array/dict field that have the specified size. `$size` does not accept ranges of values.             |
| `$exists`      | Matches documents that have the specified field. And empty string content is also considered as not exists. |


For example, to select all `modality='D'` Documents,

```python
r = da.find({'modality': {'$eq': 'D'}})

pprint(r.to_dict(exclude_none=True))  # just for pretty print
```

```text
[{'id': '92aee5d665d0c4dd34db10d83642aded',
  'modality': 'D',
  'tags': {'h': 8.5, 'uom': 'in', 'w': 11.0},
  'text': 'paper',
  'weight': 100.0},
 {'id': '1a9d2139b02bc1c7842ecda94b347889',
  'modality': 'D',
  'tags': {'h': 22.85, 'uom': 'cm', 'w': 30.0},
  'text': 'planner',
  'weight': 75.0}]
```

To select all Documents whose `.tags['h']>10`,

```python
r = da.find({'tags__h': {'$gt': 10}})
```

```text
[{'id': '4045a9659875fd1299e482d710753de3',
  'modality': 'A',
  'tags': {'h': 14.0, 'uom': 'cm', 'w': 21.0},
  'text': 'journal',
  'weight': 25.0},
 {'id': 'cf7691c445220b94b88ff116911bad24',
  'modality': 'D',
  'tags': {'h': 22.85, 'uom': 'cm', 'w': 30.0},
  'text': 'planner',
  'weight': 75.0}]
```

Beside using a predefined value, one can also use a substitution with `{field}`, notice the curly brackets there. For example,

```python
r = da.find({'tags__h': {'$gt': '{tags__w}'}})
```

```text
[{'id': '44c6a4b18eaa005c6dbe15a28a32ebce',
  'modality': 'A',
  'tags': {'h': 14.0, 'uom': 'cm', 'w': 10.0},
  'text': 'journal',
  'weight': 25.0}]
```



## Combine multiple conditions


You can combine multiple conditions using the following operators

| Boolean Operator | Description                                        |
|------------------|----------------------------------------------------|
| `$and`           | Join query clauses with a logical AND              |
| `$or`            | Join query clauses with a logical OR               |
| `$not`           | Inverts the effect of a query expression           |



```python
r = da.find({'$or': [{'weight': {'$eq': 45}}, {'modality': {'$eq': 'D'}}]})
```

```text
[{'id': '22985b71b6d483c31cbe507ed4d02bd1',
  'modality': 'D',
  'tags': {'h': 8.5, 'uom': 'in', 'w': 11.0},
  'text': 'paper',
  'weight': 100.0},
 {'id': 'a071faf19feac5809642e3afcd3a5878',
  'modality': 'D',
  'tags': {'h': 22.85, 'uom': 'cm', 'w': 30.0},
  'text': 'planner',
  'weight': 75.0},
 {'id': '411ecc70a71a3f00fc3259bf08c239d1',
  'modality': 'A',
  'tags': {'h': 10.0, 'uom': 'cm', 'w': 15.25},
  'text': 'postcard',
  'weight': 45.0}]
```
