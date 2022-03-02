# Evaluate Matches

After you get `.matches` from the last chapter, you can easily evaluate matches against the groundtruth via {meth}`~docarray.array.mixins.evaluation.EvaluationMixin.evaluate`.

```python
da_predict.evaluate(da_groundtruth, metric='...', **kwargs)
```

The results are stored in `.evaluations` field of each Document.

DocArray provides some common metrics used in the information retrieval community that allows one to evaluate the nearest-neighbour matches. Different metric accepts different arguments as `kwargs`:

| Metric              | Accept `kwargs`  |
|---------------------|------------------|
| `r_precision`       | None             |
| `average_precision` | None             |            
| `reciprocal_rank`   | None             |
| `precision_at_k`    | `k`              |
| `hit_at_k`          | `k`              |
| `recall_at_k`       | `max_rel`, `k`   |
| `f1_score_at_k`     | `max_rel`, `k`   |
| `dcg_at_k`          | `method`, `k`    |
| `ndcg_at_k`         | `method`, `k`    |


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



Now `da2` is our prediction, and `da` is our groundtruth. If we evaluate the average Precision@10, we should get something close to 0.5 (we have 10 real matches, we mixed in 10 fake matches and shuffle it, so top-10 would have approximate 10/20 real matches):

```python
da2.evaluate(da, metric='precision_at_k', k=5)
```

```text
0.48
```

Note that this value is an average number over all Documents of `da2`. If you want to look at the individual evaluation, you can check {attr}`~docarray.Document.evaluations` attribute, e.g.

```python
for d in da2:
    print(d.evaluations['precision_at_k'].value)
```

```text
0.4
0.4
0.6
0.6
0.2
0.4
0.8
0.8
0.2
0.4
```

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
p_da.evaluate(g_da, 'average_precision')
```

```text
ValueError: Document <Document ('id', 'matches') at 42dc84b26fab11ecbc181e008a366d49> from the left-hand side and <Document ('id', 'matches') at 42dc98086fab11ecbc181e008a366d49> from the right-hand are not hashed to the same value. This means your left and right DocumentArray may not be aligned; or it means your `hash_fn` is badly designed.
```

This basically saying that based on `.id` (default identifier), the given two DocumentArrays are so different that they can't be evaluated. It is a valid point because our two DocumentArrays have completely random `.id`.

If we override the hash function as following the evaluation can be conducted:

```python
p_da.evaluate(g_da, 'average_precision', hash_fn=lambda d: d.text[:2])
```

```text
1.0
```

It is correct as we define the evaluation as checking if the first two characters in `.text` are the same.

