# Evaluate Matches

After executing {meth}`~docarray.array.mixins.match.MatchMixin.match`, your `DocumentArray` receives a `.matches` attribute.
You can evaluate these matches against the ground truth via {meth}`~docarray.array.mixins.evaluation.EvaluationMixin.evaluate`.
The ground truth describes which matches are relevant and non-relevant, and can be provided in two formats: as a ground truth array or as labels.

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
Now `da.matches` contains the nearest neighbors.
To make this more interesting, let's mix in ten "noise Documents" in every `d.matches`:

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

To evaluate the matches against a ground-truth array, you pass a DocumentArray (like `da_groundtruth`) to the `evaluate()` method:

```python
da_prediction.evaluate(ground_truth=da_groundtruth, metrics=['...'], **kwargs)
```

Thereby, `da_groundtruth` should contain the same Documents as in `da_prediction`. Each `matches` attribute contains exactly those Documents which are relevant to the respective root document.

You define the metrics you want to use for your evaluation (e.g. `precision_at_k`) with the `metrics` parameter.

Let's evaluate the `da_prediction` DocumentArray (with the noisy matches) against `da_original`:

```python
da_prediction.evaluate(ground_truth=da_original, metrics=['precision_at_k'], k=10)
```

```text
{'precision_at_k': 0.45}
```
This returns the average value for the `precision_at_k` metric, calculated over all Documents of `da_prediction`.
To see the individual evaluation values, check the {attr}`~docarray.Document.evaluations` attribute:

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

Note that evaluating a DocumentArray against a ground truth DocumentArray only works if both have the same length and nested structure.
It makes no sense to evaluate with a completely different DocumentArray.

While evaluating, Document pairs are recognized as correct if they share the same identifier. By default, this is just {attr}`~docarray.Document.id`. You can customize this by specifying `hash_fn`.

Let's see an example by creating two DocumentArrays. Each DocumentArray has matches that are identical to each other, but differ from the matches of the other DocumentArray:

```python
from docarray import DocumentArray, Document

p_da = DocumentArray.empty(3)

for d in p_da:
    d.matches.append(Document(text='my predict'))

g_da = DocumentArray.empty(3)
for d in g_da:
    d.matches.append(Document(text='my ground truth'))
```

Now when you evaluate, you'll receive an error: 

```python
p_da.evaluate('average_precision', ground_truth=g_da)
```

```text
ValueError: Document <Document ('id', 'matches') at 42dc84b26fab11ecbc181e008a366d49> from the left-hand side and <Document ('id', 'matches') at 42dc98086fab11ecbc181e008a366d49> from the right-hand are not hashed to the same value. This means your left and right DocumentArray may not be aligned; or it means your `hash_fn` is badly designed.
```

This says that based on `.id` (the default identifier), the two DocumentArrays are so different that they can't be evaluated.
It is a valid point because our two DocumentArrays have completely random `.id`s.

If we override the hash function, the evaluation can proceed:

```python
p_da.evaluate('average_precision', ground_truth=g_da, hash_fn=lambda d: d.text[:2])
```

```text
{'average_precision': 1.0}
```

This is correct, as we define evaluation as checking if the first two characters in `.text` (in this case, `my`) are the same.

## Evaluation via labels

Alternatively, you can evaluate your Documents by adding labels.
A match is considered relevant to its root Document if it has the same label:

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

Also here, results are stored in the `.evaluations` attribute of each Document.

## Metric functions

DocArray provides common metrics used in the information retrieval community to evaluate nearest-neighbor matches.
Some of those metrics accept additional arguments as `kwargs` which you can add to the call of the `evaluate()` method:

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
These metric scores might change if you set the `limit` argument of the match method differently.

**Note:** Not all of these metrics can be applied to a top-K result, i.e., `ndcg_at_k` and `r_precision` are calculated correctly only if the limit is set equal to or higher than the number of Documents in the DocumentArray provided to the match method.
```

You can evaluate multiple metric functions at once:

```python
da_prediction.evaluate(
    ground_truth=da_original, metrics=['precision_at_k', 'reciprocal_rank'], k=10
)
```

```text
{'precision_at_k': 0.45, 'reciprocal_rank': 0.8166666666666667}
```

In this case, the keyword argument `k` is passed to all metric functions, even though it fulfills no specific function for calculating the reciprocal rank.

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

If pre-defined metrics don't fit your use case, you can define a custom metric function, taking as input a list of binary relevance judgements of a query (`1` and `0` values).
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

As inspiration for writing your own metric function, see DocArray's {mod}`~docarray.math.evaluation` module, which contains the implementations of the custom metric functions.

### Custom names

By default, metrics are stored with the name of the metric function.
Alternatively, you can customize those names with the `metric_names` argument of the `evaluate` method:

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

## Embed, match and evaluate at once

Instead of executing the methods {meth}`~docarray.array.mixins.embed.EmbedMixin.embed`, {meth}`~docarray.array.mixins.match.MatchMixin.match`, and {meth}`~docarray.array.mixins.evaluation.EvaluationMixin.evaluate` separately, you can execute them all at once with {meth}`~docarray.array.mixins.evaluation.EvaluationMixin.embed_and_evaluate`.

To demonstrate this, let's construct two labeled DocumentArrays `example_queries` and `example_index`. `example_index` should be matched with `example_queries` and then we want to evaluate the reciprocal rank based on the matches' labels in `example_queries`.

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

``embed_and_evaluate`` is especially useful to evaluate queries on a Document collection (like `example_index`) which is too large to fit the embeddings of all Documents in main memory. In this case, the method matches queries to batches of the Document collection, then deletes embeddings after processing each batch.

By default, the batch size for the matching (`match_batch_size`) is set to `100_000`. To reduce the memory footprint, you can set it to a lower value.

### Sampling Queries

To evaluate a large dataset, it might be useful to sample query Documents.
Since the metric values returned by `embed_and_evaluate` are mean values, sampling shouldn't significantly change the result if the sample is large enough.
By default, sampling is applied for DocumentArrays with over 1,000 Documents. However, it's only applied on the `DocumentArray` itself and not on the Document provided in `index_data`.

To change the number of samples, you can adjust the `query_sample_size` argument. In the following code block an evaluation is performed with 100 samples:

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

Note that in this way, only Documents which are actually evaluated obtain an `.evaluations` attribute.

To test how close it is to the exact result, you can execute the function again with `query_sample_size` set to `1_000`:

```python
da.embed_and_evaluate(
    metrics=['precision_at_k'], embed_funcs=emb_func, query_sample_size=1_000
)
```

```text
{'precision_at_k': 0.14245}
```
