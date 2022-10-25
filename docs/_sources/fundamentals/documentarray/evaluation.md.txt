# Evaluate Matches

After the execution of {meth}`~docarray.array.mixins.match.MatchMixin.match`, your `DocumentArray` receives a `.matches` attribute.
You can evaluate those matches against the ground truth via {meth}`~docarray.array.mixins.evaluation.EvaluationMixin.evaluate`.
The ground truth describes which matches are relevant and non-relevant and can be provided in two formats: (1) a ground truth array or (2) in the form of labels.

To demonstrate this, let's create a DocumentArray with random embeddings and match it to itself:

```python
import numpy as np
from docarray import DocumentArray

da_original = DocumentArray.empty(10)
da_original.embeddings = np.random.random([10, 3])
da_original.match(da_original, exclude_self=True)

da_original.summary()
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
Now `da.matches` contains the nearest neighbours.
To make our scenario more interesting, we mix in ten "noise Documents" to every `d.matches`:

```python
da_prediction = DocumentArray(da_original, copy=True)

for d in da_prediction:
    d.matches.extend(DocumentArray.empty(10))
    d.matches = d.matches.shuffle()

da_prediction['@m'].summary()
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

## Evaluation against a ground truth array

To evaluate the matches against a ground truth array, you simply provide a DocumentArray to the evaluate function like `da_groundtruth` in the call below:

```python
da_predict.evaluate(ground_truth=da_groundtruth, metrics=['...'], **kwargs)
```

Thereby, `da_groundtruth` should contain the same documents as in `da_prediction` where each `matches` attribute contains exactly those documents which are relevant to the respective root document.
The `metrics` argument determines the metric you want to use for your evaluation, e.g., `precision_at_k`.

In the code cell below, we evaluate the array `da_prediction` with the noisy matches against the original one `da_original`:

```python
da_prediction.evaluate(ground_truth=da_original, metrics=['precision_at_k'], k=10)
```

```text
{'precision_at_k': 0.45}
```
It returns the average value for the `precision_at_k` metric.
The average is calculated over all Documents of `da_prediction`.
If you want to look at the individual evaluation values, you can check the {attr}`~docarray.Document.evaluations` attribute, e.g.:

```python
for d in da_prediction:
    print(d.evaluations['precision_at_k'].value)
```

```text
0.5
0.5
0.5
0.6
0.3
0.4
0.5
0.4
0.5
0.3
```

### Document identifier

Note that the evaluation against a ground truth DocumentArray only works if both DocumentArrays have the same length and their nested structure is the same.
It makes no sense to evaluate with a completely different DocumentArray.

While evaluating, Document pairs are recognized as correct if they share the same identifier. By default, it simply uses {attr}`~docarray.Document.id`. One can customize this behavior by specifying `hash_fn`.

Let's see an example by creating two DocumentArrays with some matches with identical texts.

```python
from docarray import DocumentArray, Document

p_da = DocumentArray.empty(3)

for d in p_da:
    d.matches.append(Document(text='my predict'))

g_da = DocumentArray.empty(3)
for d in g_da:
    d.matches.append(Document(text='my ground truth'))
```

Now when you do evaluate, you will receive an error: 

```python
p_da.evaluate('average_precision', ground_truth=g_da)
```

```text
ValueError: Document <Document ('id', 'matches') at 42dc84b26fab11ecbc181e008a366d49> from the left-hand side and <Document ('id', 'matches') at 42dc98086fab11ecbc181e008a366d49> from the right-hand are not hashed to the same value. This means your left and right DocumentArray may not be aligned; or it means your `hash_fn` is badly designed.
```

This says that based on `.id` (default identifier), the given two DocumentArrays are so different that they can't be evaluated.
It is a valid point because our two DocumentArrays have completely random `.id`.

If we override the hash function as follows, the evaluation can be conducted:

```python
p_da.evaluate('average_precision', ground_truth=g_da, hash_fn=lambda d: d.text[:2])
```

```text
{'average_precision': 1.0}
```

It is correct as we define the evaluation as checking if the first two characters in `.text` are the same.



## Evaluation via labels

Alternatively, you can add labels to your documents to evaluate them.
In this case, a match is considered relevant to its root document if it has the same label:

```python
import numpy as np
from docarray import Document, DocumentArray

example_da = DocumentArray([Document(tags={'label': (i % 2)}) for i in range(10)])
example_da.embeddings = np.random.random([10, 3])

example_da.match(example_da)

example_da.evaluate(metrics=['precision_at_k'])
```

```text
{'precision_at_k': 0.5}
```

Also here, the results are stored in the `.evaluations` field of each Document.

## Metric functions

DocArray provides common metrics used in the information retrieval community for evaluating the nearest-neighbour matches.
Some of those metrics accept additional arguments as `kwargs` which you can simply add to the call of the evaluate function:

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
These metric scores might change if the `limit` argument of the match function is set differently.

**Note:** Not all of these metrics can be applied to a Top-K result, i.e., `ndcg_at_k` and `r_precision` are calculated correctly only if the limit is set equal or higher than the number of documents in the `DocumentArray` provided to the match function.
```

You can evaluate multiple metric functions at once, as you can see below:

```python
da_prediction.evaluate(
    ground_truth=da_original, metrics=['precision_at_k', 'reciprocal_rank'], k=10
)
```

```text
{'precision_at_k': 0.45, 'reciprocal_rank': 0.8166666666666667}
```

In this case, the keyword argument `k` is passed to all metric functions, even though it does not fulfill any specific function for the calculation of the reciprocal rank.

### Custom metrics

If the pre-defined metrics do not fit your use-case, you can define a custom metric function.
It should take as input a list of binary relevance judgements of a query (`1` and `0` values).
The evaluate function already calculates this binary list from the `matches` attribute so that each number represents the relevancy of a match.

Let's write a custom metric function, which counts the number of relevant documents per query:

```python
def count_relevant(binary_relevance):
    return sum(binary_relevance)

da_prediction.evaluate(ground_truth=da_original, metrics=[count_relevant])
```

```text
{'count_relevant': 9.0}
```

For an inspiration for writing your own metric function, you can take a look at DocArray's {mod}`~docarray.math.evaluation` module, which contains the implementations of the custom metric functions.

### Custom names

By default, the metrics are stored with the name of the metric function.
Alternatively, you can customize those names via the `metric_names` argument of the `evaluate` function:

```python
da_prediction.evaluate(
    ground_truth=da_original,
    metrics=[count_relevant, 'precision_at_k'],
    metric_names=['#Relevant', 'Precision@K'],
)
```

```text
{'#Relevant': 9.0, 'Precision@K': 0.47368421052631576}
```
