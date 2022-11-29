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
da_prediction.evaluate(ground_truth=da_groundtruth, metrics=['...'], **kwargs)
```

Thereby, `da_groundtruth` should contain the same Documents as in `da_prediction` where each `matches` attribute contains exactly those Documents which are relevant to the respective root Document.
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

While evaluating, Document pairs are recognized as correct if they share the same identifier. By default, it simply uses {attr}`~docarray.Document.id`.
You can customize this behavior by specifying `hash_fn`.

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

Alternatively, you can add labels to your Documents to evaluate them.
In this case, a match is considered relevant to its root Document if it has the same label:

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

**Note:** Not all of these metrics can be applied to a Top-K result, i.e., `ndcg_at_k` and `r_precision` are calculated correctly only if the limit is set equal or higher than the number of Documents in the `DocumentArray` provided to the match function.
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

### The max_rel parameter

Some metric functions shown in the table above require a `max_rel` parameter.
This parameter should be set to the number of relevant Documents in the Document collection.
Without the knowledge of this number, metrics like `recall_at_k` and `f1_score_at_k` cannot be calculated.

In the `evaluate` function, you can provide a keyword argument `max_rel`, which is then used for all queries.
In the example below, we can use the datasets `da_prediction` and `da_original` from the beginning, where each query has nine relevant Documents.
Therefore, we set `max_rel=9`.

```python
da_prediction.evaluate(ground_truth=da_original, metrics=['recall_at_k'], max_rel=9)
```

```text
{'recall_at_k': 1.0}
```

Since all relevant Documents are in the matches, the recall is one.
However, this only makes sense if the number of relevant Documents is equal for each query.
If you provide a `ground_truth` parameter to the `evaluate` function, `max_rel` is set to the number of matches of the query Document.

```python
da_prediction.evaluate(ground_truth=da_original, metrics=['recall_at_k'])
```
```text
{'recall_at_k': 1.0}
```

For labeled datasets, this is not possible.
Here, you can set the `num_relevant_documents_per_label` parameter of `evaluate`.
It accepts a dictionary that contains the number of relevant Documents for each label.
In this way, the function can set `max_rel` to the correct value for each query Document.

```python
example_da.evaluate(
    metrics=['recall_at_k'], num_relevant_documents_per_label={0: 5, 1: 5}
)
```

```text
{'recall_at_k': 1.0}
```

### Custom metrics

If the pre-defined metrics do not fit your use-case, you can define a custom metric function.
It should take as input a list of binary relevance judgements of a query (`1` and `0` values).
The evaluate function already calculates this binary list from the `matches` attribute so that each number represents the relevancy of a match.

Let's write a custom metric function, which counts the number of relevant Documents per query:

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

## Embed, match & evaluate at once

Instead of executing the functions {meth}`~docarray.array.mixins.embed.EmbedMixin.embed`, {meth}`~docarray.array.mixins.match.MatchMixin.match`, and {meth}`~docarray.array.mixins.evaluation.EvaluationMixin.evaluate` separately from each other, you can also execute them all at once by using {meth}`~docarray.array.mixins.evaluation.EvaluationMixin.embed_and_evaluate`.
To demonstrate this, we constuct two labeled DocumentArrays `example_queries` and `example_index`.
The second one `example_index` should be matched with `example_queries` and afterwards, we want to evaluate the reciprocal rank based on the labels of the matches in `example_queries`.

```python
import numpy as np
from docarray import Document, DocumentArray

example_queries = DocumentArray([Document(tags={'label': (i % 2)}) for i in range(10)])
example_index = DocumentArray([Document(tags={'label': (i % 2)}) for i in range(10)])


def embedding_function(da):
    da[:, 'embedding'] = np.random.random((len(da), 5))


result = example_queries.embed_and_evaluate(
    'reciprocal_rank', index_da=example_index, embed_funcs=embedding_function
)
print(result)
```

```text
{'reciprocal_rank': 0.7583333333333333}
```

For metric functions which require a `max_rel` parameter, the `embed_and_evaluate` function (described later in this section) automatically constructs the dictionary for `num_relevant_documents_per_label` based on the `index_data` argument.

### Batch-wise matching

The ``embed_and_evaluate`` function is especially useful, when you need to evaluate the queries on a very large Document collection (`example_index` in the code snippet above), which is too large to store the embeddings of all Documents in main-memory.
In this case, ``embed_and_evaluate`` matches the queries to batches of the Document collection.
After the batch is processed all embeddings are deleted.
By default, the batch size for the matching (`match_batch_size`) is set to `100_000`.
If you want to reduce the memory footprint, you can set it to a lower value.

### Sampling Queries

If you want to evaluate a large dataset, it might be useful to sample query Documents.
Since the metric values returned by the `embed_and_evaluate` are mean values, sampling should not change the result significantly if the sample is large enough.
By default, sampling is applied for `DocumentArray` objects with more than 1,000 Documents.
However, it is only applied on the `DocumentArray` itself and not on the Documents provided in `index_data`.
If you want to change the number of samples, you can ajust the `query_sample_size` argument.
In the following code block an evaluation is done with 100 samples:

```python
import numpy as np
from docarray import Document, DocumentArray


def emb_func(da):
    for d in da:
        np.random.seed(int(d.text))
        d.embedding = np.random.random(5)


da = DocumentArray(
    [Document(text=str(i), tags={'label': i % 10}) for i in range(1_000)]
)

da.embed_and_evaluate(
    metrics=['precision_at_k'], embed_funcs=emb_func, query_sample_size=100
)
```

```text
{'precision_at_k': 0.13649999999999998}
```

Please note that in this way only Documents which are actually evaluated obtain an `.evaluations` attribute.

To test how close it is to the exact result, we execute the function again with `query_sample_size` set to 1,000:

```python
da.embed_and_evaluate(
    metrics=['precision_at_k'], embed_funcs=emb_func, query_sample_size=1_000
)
```

```text
{'precision_at_k': 0.14245}
```
