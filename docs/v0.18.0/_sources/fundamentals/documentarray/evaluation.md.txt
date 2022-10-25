# Evaluate Matches

After you get `.matches`, you can evaluate matches against the groundtruth via {meth}`~docarray.array.mixins.evaluation.EvaluationMixin.evaluate`.

```python
da_predict.evaluate(ground_truth=da_groundtruth, metrics=['...'], **kwargs)
```

Alternatively, you can add labels to your documents to evaluate them.
In this case, a match is considered as relevant to its root document, if it has the same label.

```python
import numpy as np
from docarray import Document, DocumentArray

example_da = DocumentArray([Document(tags={'label': (i % 2)}) for i in range(10)])
example_da.embeddings = np.random.random([10, 3])

example_da.match(example_da)

example_da.evaluate(metrics=['precision_at_k'])
```

The results are stored in `.evaluations` field of each Document.

DocArray provides some common metrics used in the information retrieval community that allows one to evaluate the nearest-neighbour matches. Different metric accepts different arguments as `kwargs`:

| Metric                                              | Accept `kwargs`  |
|-----------------------------------------------------|------------------|
| {meth}`~docarray.math.evaluation.r_precision`       | None             |
| {meth}`~docarray.math.evaluation.average_precision` | None             |            
| {meth}`~docarray.math.evaluation.reciprocal_rank`   | None             |
| {meth}`~docarray.math.evaluation.precision_at_k`    | `k`              |
| {meth}`~docarray.math.evaluation.hit_at_k`          | `k`              |
| {meth}`~docarray.math.evaluation.recall_at_k`       | `max_rel`, `k`   |
| {meth}`~docarray.math.evaluation.f1_score_at_k`     | `max_rel`, `k`   |
| {meth}`~docarray.math.evaluation.dcg_at_k`          | `method`, `k`    |
| {meth}`~docarray.math.evaluation.ndcg_at_k`         | `method`, `k`    |

```{danger}
This metric scores might change if the `limit` attribute of the match function is set differently.

**Note:** Not all of these metrics can be applied to a Top-K result, i.e., `ndcg_at_k` and `r_precision` are calculated correctly only if the limit is set equal or higher than the number of documents in the `DocumentArray` provided to the match function.
```


For example, let's create a DocumentArray with random embeddings and matching it to itself:

```python
import numpy as np
from docarray import DocumentArray

da = DocumentArray.empty(10)
da.embeddings = np.random.random([10, 3])
da.match(da, exclude_self=True)

da.summary()
```


```text
                     Documents Summary                      
                                                            
  Length                    10                              
  Homogenous Documents      True                            
  Has nested Documents in   ('matches',)                    
  Common Attributes         ('id', 'embedding', 'matches')  
                                                            
                        Attributes Summary                        
                                                                  
  Attribute   Data type         #Unique values   Has empty value  
 ──────────────────────────────────────────────────────────────── 
  embedding   ('ndarray',)      10               False            
  id          ('str',)          10               False            
  matches     ('MatchArray',)   10               False                
```

Now `da.matches` contains the nearest neighbours. Let's use it as the groundtruth. 

Let's create imperfect matches by mixing in ten "noise Documents" to every `d.matches`.

```python
da2 = DocumentArray(da, copy=True)

for d in da2:
    d.matches.extend(DocumentArray.empty(10))
    d.matches = d.matches.shuffle()

da2['@m'].summary()
```

```text
                              Documents Summary                               
                                                                              
  Length                          190                                         
  Homogenous Documents            False                                       
  100 Documents have attributes   ('id', 'adjacency')                         
  90 Documents have attributes    ('id', 'adjacency', 'embedding', 'scores')  
                                                                              
                            Attributes Summary                            
                                                                          
  Attribute   Data type                 #Unique values   Has empty value  
 ──────────────────────────────────────────────────────────────────────── 
  adjacency   ('int',)                  1                False            
  embedding   ('ndarray', 'NoneType')   190              True             
  id          ('str',)                  110              False            
  scores      ('defaultdict',)          190              False            
```



Now `da2` is our prediction, and `da` is our groundtruth. If we evaluate the average Precision@10, we should get something close to 0.47 (we have 9 real matches, we mixed in 10 fake matches and shuffle it, so top-10 would have approximate 9/19 real matches):

```python
da2.evaluate(ground_truth=da, metrics=['precision_at_k'], k=10)
```

```text
{'precision_at_k': 0.48}
```

Note that this value is an average number over all Documents of `da2`. If you want to look at the individual evaluation, you can check {attr}`~docarray.Document.evaluations` attribute, e.g.

```python
for d in da2:
    print(d.evaluations['precision_at_k'].value)
```

```text
0.5
0.4
0.3
0.6
0.5
0.3
0.4
0.6
0.5
0.7
```

If you want to evaluate your data with multiple metric functions, you can pass a list of metrics:

```python
da2.evaluate(ground_truth=da, metrics=['precision_at_k', 'reciprocal_rank'], k=10)
```

```text
{'precision_at_k': 0.48, 'reciprocal_rank': 0.6333333333333333}
```

In this case, the keyword attribute `k` is passed to all metric functions, even though it does not fulfill any specific function for the calculation of the reciprocal rank.

## Document identifier

Note that `.evaluate()` works only when two DocumentArray have the same length and their nested structure are same. It makes no sense to evaluate on two completely irrelevant DocumentArrays.

While evaluating, Document pairs are recognized as correct if they share the same identifier. By default, it simply uses {attr}`~docarray.Document.id`. One can customize this behavior by specifying `hash_fn`.

Let's see an example by creating two DocumentArrays with some matches with identical texts.

```python
from docarray import DocumentArray, Document

p_da = DocumentArray.empty(3)

for d in p_da:
    d.matches.append(Document(text='my predict'))

g_da = DocumentArray.empty(3)
for d in g_da:
    d.matches.append(Document(text='my groundtruth'))
```

Now when you do evaluate, you will receive an error: 

```python
p_da.evaluate('average_precision', ground_truth=g_da)
```

```text
ValueError: Document <Document ('id', 'matches') at 42dc84b26fab11ecbc181e008a366d49> from the left-hand side and <Document ('id', 'matches') at 42dc98086fab11ecbc181e008a366d49> from the right-hand are not hashed to the same value. This means your left and right DocumentArray may not be aligned; or it means your `hash_fn` is badly designed.
```

This basically saying that based on `.id` (default identifier), the given two DocumentArrays are so different that they can't be evaluated. It is a valid point because our two DocumentArrays have completely random `.id`.

If we override the hash function as following the evaluation can be conducted:

```python
p_da.evaluate('average_precision', ground_truth=g_da, hash_fn=lambda d: d.text[:2])
```

```text
{'average_precision': 1.0}
```

It is correct as we define the evaluation as checking if the first two characters in `.text` are the same.

