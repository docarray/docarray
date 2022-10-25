(table-type)=
# {octicon}`table` Table

One can freely convert between DocumentArray and `pandas.Dataframe`, read more details in {ref}`docarray-serialization`. Besides, one can load and write CSV file with DocumentArray. 

## Load CSV table

One can easily load tabular data from `csv` file into a DocumentArray. For example, 

```text
Username;Identifier;First name;Last name
booker12;9012;Rachel;Booker
grey07;2070;Laura;Grey
johnson81;4081;Craig;Johnson
jenkins46;9346;Mary;Jenkins
smith79;5079;Jamie;Smith
```

```python
from docarray import DocumentArray

da = DocumentArray.from_csv('toy.csv')
```

```text
            Documents Summary            
                                         
  Length                 5               
  Homogenous Documents   True            
  Common Attributes      ('id', 'tags')  
                                         
                     Attributes Summary                     
                                                            
  Attribute   Data type   #Unique values   Has empty value  
 ────────────────────────────────────────────────────────── 
  id          ('str',)    5                False            
  tags        ('dict',)   5                False            
```

One can observe that each row is loaded as a Document and the columns are loaded into `Document.tags`.


In general, `from_csv` will try its best to resolve the column names of the table and map them into the corresponding Document attributes. If such attempt fails, one can always resolve the field manually via:

```python
from docarray import DocumentArray

da = DocumentArray.from_csv('toy.csv', field_resolver={'Identifier': 'id'})
```

## Save as CSV file

Saving a DocumentArray as a `csv` file is easy.

```python
da.save_csv('tmp.csv')
```

One thing needs to be careful is that tabular data is often not good for representing nested Document. Hence, nested Document will be stored in flatten.

If your Documents contain tags, and you want to store each tag in a separate column, then you can do:

```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(tags={'english': 'hello', 'german': 'hallo'}),
                    Document(tags={'english': 'world', 'german': 'welt'})])

da.save_csv('toy.csv', flatten_tags=True)
```

````{tab} flatten_tags=True

```text
id,tag__english,tag__german
029388a4-3830-11ec-b6b2-1e008a366d48,hello,hallo
0293968c-3830-11ec-b6b2-1e008a366d48,world,welt
```
````
````{tab} flatten_tags=False

```text
id,tags
418de220-3830-11ec-aad8-1e008a366d48,"{'german': 'hallo', 'english': 'hello'}"
418dec52-3830-11ec-aad8-1e008a366d48,"{'english': 'world', 'german': 'welt'}"
```
````
