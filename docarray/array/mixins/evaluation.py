import warnings
from typing import Optional, Union, TYPE_CHECKING, Callable, List, Dict, Tuple, Any

from functools import wraps

import numpy as np
from collections import defaultdict, Counter

from docarray.score import NamedScore

if TYPE_CHECKING:  # pragma: no cover
    from docarray import Document, DocumentArray
    from docarray.array.mixins.embed import CollateFnType
    from docarray.typing import ArrayType, AnyDNN


def _evaluate_deprecation(f):
    """Raises a deprecation warning if the user executes the evaluate function with
    the old interface and adjust the input to fit the new interface."""

    @wraps(f)
    def func(*args, **kwargs):
        if len(args) > 1:
            if not (
                isinstance(args[1], Callable)
                or isinstance(args[1], str)
                or isinstance(args[1], list)
            ):
                kwargs['ground_truth'] = args[1]
                args = [args[0]] + list(args[2:])
                warnings.warn(
                    'The `other` attribute in `evaluate()` is transfered from a '
                    'positional attribute into the keyword attribute `ground_truth`.'
                    'Using it as a positional attribute is deprecated and will be removed '
                    'soon.',
                    DeprecationWarning,
                )
        for old_key, new_key in zip(
            ['other', 'metric', 'metric_name'],
            ['ground_truth', 'metrics', 'metric_names'],
        ):
            if old_key in kwargs:
                kwargs[new_key] = kwargs[old_key]
                warnings.warn(
                    f'`{old_key}` is renamed to `{new_key}` in `evaluate()`, the '
                    f'usage of `{old_key}` is deprecated and will be removed soon.',
                    DeprecationWarning,
                )

        # transfer metrics and metric_names into lists
        list_warning_msg = (
            'The attribute `%s` now accepts a list instead of a '
            'single element. Passing a single element is deprecated and will soon not '
            'be supported anymore.'
        )
        if len(args) > 1:
            if type(args[1]) is str:
                args = list(args)
                args[1] = [args[1]]
                warnings.warn(list_warning_msg % 'metrics', DeprecationWarning)
        for key in ['metrics', 'metric_names']:
            if key in kwargs and type(kwargs[key]) is str:
                kwargs[key] = [kwargs[key]]
                warnings.warn(list_warning_msg % key, DeprecationWarning)
        return f(*args, **kwargs)

    return func


class EvaluationMixin:
    """A mixin that provides ranking evaluation functionality to DocumentArrayLike objects"""

    @_evaluate_deprecation
    def evaluate(
        self,
        metrics: List[Union[str, Callable[..., float]]],
        ground_truth: Optional['DocumentArray'] = None,
        hash_fn: Optional[Callable[['Document'], str]] = None,
        metric_names: Optional[List[str]] = None,
        strict: bool = True,
        label_tag: str = 'label',
        num_relevant_documents_per_label: Optional[Dict[Any, int]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute ranking evaluation metrics for a given `DocumentArray` when compared
        with a ground truth.

        If one provides a `ground_truth` DocumentArray that is structurally identical
        to `self`, this function compares the `matches` of `documents` inside the
        `DocumentArray` to this `ground_truth`.
        Alternatively, one can directly annotate the documents by adding labels in the
        form of tags with the key specified in the `label_tag` attribute.
        Those tags need to be added to `self` as well as to the documents in the
        matches properties.

        This method will fill the `evaluations` field of Documents inside this
        `DocumentArray` and will return the average of the computations

        :param metrics: List of metric names or metric functions to be computed
        :param ground_truth: The ground_truth `DocumentArray` that the `DocumentArray`
            compares to.
        :param hash_fn: For the evaluation against a `ground_truth` DocumentArray,
            this function is used for generating hashes which are used to compare the
            documents. If not given, ``Document.id`` is used.
        :param metric_names: If provided, the results of the metrics computation will be
            stored in the `evaluations` field of each Document with this names. If not
            provided, the names will be derived from the metric function names.
        :param strict: If set, then left and right sides are required to be fully
            aligned: on the length, and on the semantic of length. These are preventing
            you to evaluate on irrelevant matches accidentally.
        :param label_tag: Specifies the tag which contains the labels.
        :param num_relevant_documents_per_label: Some metrics, e.g., recall@k, require
            the number of relevant documents. To apply those to a labeled dataset, one
            can provide a dictionary which maps labels to the total number of documents
            with this label.
        :param kwargs: Additional keyword arguments to be passed to the metric
            functions.
        :return: A dictionary which stores for each metric name the average evaluation
            score.
        """
        if len(self) == 0:
            raise ValueError('It is not possible to evaluate an empty DocumentArray')
        if ground_truth and len(ground_truth) > 0 and ground_truth[0].matches:
            ground_truth_type = 'matches'
        elif label_tag in self[0].tags:
            if ground_truth:
                warnings.warn(
                    'A ground_truth attribute is provided but does not '
                    'contain matches. The labels are used instead and '
                    'ground_truth is ignored.'
                )
            ground_truth = self
            ground_truth_type = 'labels'

        else:
            raise RuntimeError(
                'Could not find proper ground truth data. Either labels or the '
                'ground_truth attribute with matches is required'
            )

        if strict:
            self._check_length(len(ground_truth))

        if hash_fn is None:
            hash_fn = lambda d: d.id

        metric_fns = []
        for metric in metrics:
            if callable(metric):
                metric_fns.append(metric)
            elif isinstance(metric, str):
                from docarray.math import evaluation

                metric_fns.append(getattr(evaluation, metric))

        if not metric_names:
            metric_names = [metric_fn.__name__ for metric_fn in metric_fns]

        if len(metric_names) != len(metrics):
            raise ValueError(
                'Could not match metric names to the metrics since the number of '
                'metric names does not match the number of metrics'
            )

        results = defaultdict(list)
        caller_max_rel = kwargs.pop('max_rel', None)
        for d, gd in zip(self, ground_truth):
            if caller_max_rel:
                max_rel = caller_max_rel
            elif ground_truth_type == 'labels':
                if num_relevant_documents_per_label:
                    max_rel = num_relevant_documents_per_label.get(
                        d.tags[label_tag], None
                    )
                    if max_rel is None:
                        raise ValueError(
                            '`num_relevant_documents_per_label` misses the label '
                            + str(d.tags[label_tag])
                        )
                else:
                    max_rel = None
            else:
                max_rel = len(gd.matches)
            if strict and hash_fn(d) != hash_fn(gd):
                raise ValueError(
                    f'Document {d} from the left-hand side and '
                    f'{gd} from the right-hand are not hashed to the same value. '
                    f'This means your left and right DocumentArray may not be aligned; or it means your '
                    f'`hash_fn` is badly designed.'
                )
            if not d.matches or not gd.matches:
                raise ValueError(
                    f'Document {d!r} or {gd!r} has no matches, please check your Document'
                )

            targets = gd.matches

            if ground_truth_type == 'matches':
                desired = {hash_fn(m) for m in targets}
                if len(desired) != len(targets):
                    warnings.warn(
                        f'{hash_fn!r} may not be valid, as it maps multiple Documents into the same hash. '
                        f'Evaluation results may be affected'
                    )
                binary_relevance = [
                    1 if hash_fn(m) in desired else 0 for m in d.matches
                ]
            elif ground_truth_type == 'labels':
                binary_relevance = [
                    1 if m.tags[label_tag] == d.tags[label_tag] else 0 for m in targets
                ]
            else:
                raise RuntimeError(
                    'Could not identify which kind of ground truth'
                    'information is provided to evaluate the matches.'
                )
            for metric_name, metric_fn in zip(metric_names, metric_fns):
                if 'max_rel' in metric_fn.__code__.co_varnames:
                    kwargs['max_rel'] = max_rel
                r = metric_fn(binary_relevance, **kwargs)
                d.evaluations[metric_name] = NamedScore(
                    value=r, op_name=str(metric_fn), ref_id=d.id
                )
                results[metric_name].append(r)
        return {
            metric_name: float(np.mean(values))
            for metric_name, values in results.items()
        }

    def embed_and_evaluate(
        self,
        metrics: List[Union[str, Callable[..., float]]],
        index_data: Optional['DocumentArray'] = None,
        ground_truth: Optional['DocumentArray'] = None,
        metric_names: Optional[str] = None,
        strict: bool = True,
        label_tag: str = 'label',
        embed_models: Optional[Union['AnyDNN', Tuple['AnyDNN', 'AnyDNN']]] = None,
        embed_funcs: Optional[Union[Callable, Tuple[Callable, Callable]]] = None,
        device: str = 'cpu',
        batch_size: Union[int, Tuple[int, int]] = 256,
        collate_fns: Union[
            Optional['CollateFnType'],
            Tuple[Optional['CollateFnType'], Optional['CollateFnType']],
        ] = None,
        distance: Union[
            str, Callable[['ArrayType', 'ArrayType'], 'np.ndarray']
        ] = 'cosine',
        limit: Optional[Union[int, float]] = 20,
        normalization: Optional[Tuple[float, float]] = None,
        exclude_self: bool = False,
        use_scipy: bool = False,
        num_worker: int = 1,
        match_batch_size: int = 100_000,
        query_sample_size: int = 1_000,
        **kwargs,
    ) -> Optional[Union[float, List[float]]]:  # average for each metric
        """
        Computes ranking evaluation metrics for a given `DocumentArray`. This
        function does embedding and matching in the same turn. Thus, you don't need to
        call ``embed`` and ``match`` before it. Instead, it embeds the documents in
        `self` (and `index_data` when provided`) and compute the nearest neighbour
        itself. This might be done in batches for the `index_data` object to reduce
        the memory consumption of the evlauation process. The evaluation itself can be
        done against a `ground_truth` DocumentArray or on the basis of labels like it
        is possible with the :func:``evaluate`` function.

        :param metrics: List of metric names or metric functions to be computed
        :param index_data: The other DocumentArray  to match against, if not given,
            `self` will be matched against itself. This means that every document in
            will be compared to all other documents in `self` to determine the nearest
            neighbors.
        :param ground_truth: The ground_truth `DocumentArray` that the `DocumentArray`
            compares to.
        :param metric_names: If provided, the results of the metrics computation will be
            stored in the `evaluations` field of each Document with this names. If not
            provided, the names will be derived from the metric function names.
        :param strict: If set, then left and right sides are required to be fully
            aligned: on the length, and on the semantic of length. These are preventing
            you to evaluate on irrelevant matches accidentally.
        :param label_tag: Specifies the tag which contains the labels.
        :param embed_models: One or two embedding model written in Keras / Pytorch /
            Paddle for embedding `self` and `index_data`.
        :param embed_funcs: As an alternative to embedding models, custom embedding
            functions can be provided.
        :param device: the computational device for `embed_models`, and the matching
            can be either `cpu` or `cuda`.
        :param batch_size: Number of documents in a batch for embedding.
        :param collate_fns: For each embedding function the respective collate
            function creates a mini-batch of input(s) from the given `DocumentArray`.
            If not provided a default built-in collate_fn uses the `tensors` of the
            documents to create input batches.
        :param distance: The distance metric.
        :param limit: The maximum number of matches, when not given defaults to 20.
        :param normalization: A tuple [a, b] to be used with min-max normalization,
            the min distance will be rescaled to `a`, the max distance will be
            rescaled to `b` all values will be rescaled into range `[a, b]`.
        :param exclude_self: If set, Documents in ``index_data`` with same ``id``
            as the left-hand values will not be considered as matches.
        :param use_scipy: if set, use ``scipy`` as the computation backend. Note,
            ``scipy`` does not support distance on sparse matrix.
        :param num_worker: Specifies the number of workers for the execution of the
            match function.
        :parma match_batch_size: The number of documents which are embedded and
            matched at once. Set this value to a lower value, if you experience high
            memory consumption.
        :param kwargs: Additional keyword arguments to be passed to the metric
            functions.
        :param query_sample_size: For a large number of documents in `self` the
            evaluation becomes infeasible, especially, if `index_data` is large.
            Therefore, queries are sampled if the number of documents in `self` exceeds
            `query_sample_size`. Usually, this has only small impact on the mean metric
            values returned by this function. To prevent sampling, you can set
            `query_sample_size` to None.
        :return: A dictionary which stores for each metric name the average evaluation
            score.
        """

        from docarray import Document, DocumentArray

        if not query_sample_size:
            query_sample_size = len(self)

        query_data = self
        only_one_dataset = not index_data
        apply_sampling = len(self) > query_sample_size

        if only_one_dataset:
            # if the user does not provide a separate set of documents for indexing,
            # the matching is done on the documents itself
            copy_flag = (
                apply_sampling
                or (type(embed_funcs) is tuple)
                or ((embed_funcs is None) and (type(embed_models) is tuple))
            )
            index_data = DocumentArray(self, copy=True) if copy_flag else self

        if apply_sampling:
            rng = np.random.default_rng()
            query_data = DocumentArray(
                rng.choice(self, size=query_sample_size, replace=False)
            )

        if ground_truth and apply_sampling:
            ground_truth = DocumentArray(
                [ground_truth[d.id] for d in query_data if d.id in ground_truth]
            )
            if len(ground_truth) != len(query_data):
                raise ValueError(
                    'The DocumentArray provided in the ground_truth attribute does '
                    'not contain all the documents in self.'
                )

        index_data_labels = None
        if not ground_truth:
            if not label_tag in query_data[0].tags:
                raise ValueError(
                    'Either a ground_truth `DocumentArray` or labels are '
                    'required for the evaluation.'
                )
            if not label_tag in index_data[0].tags:
                raise ValueError(
                    'The `DocumentArray` provided in `index_data` misses ' 'labels.'
                )
            index_data_labels = dict()
            for id_value, tags in zip(index_data[:, 'id'], index_data[:, 'tags']):
                index_data_labels[id_value] = tags[label_tag]

        if embed_funcs is None:
            # derive embed function from embed model
            if embed_models is None:
                raise RuntimeError(
                    'For embedding the documents you need to provide either embedding '
                    'model(s) or embedding function(s)'
                )
            else:
                if type(embed_models) is not tuple:
                    embed_models = (embed_models, embed_models)
                embed_args = [
                    {
                        'embed_model': model,
                        'device': device,
                        'batch_size': batch_size,
                        'collate_fn': collate_fns[i]
                        if type(collate_fns) is tuple
                        else collate_fns,
                    }
                    for i, (model, docs) in enumerate(
                        zip(embed_models, (query_data, index_data))
                    )
                ]
        else:
            if type(embed_funcs) is not tuple:
                embed_funcs = (
                    embed_funcs,
                    embed_funcs,
                )  # use the same embedding function for queries and index

        # embed queries:
        if embed_funcs:
            embed_funcs[0](query_data)
        else:
            query_data.embed(**embed_args[0])

        for doc in query_data:
            doc.matches.clear()

        local_queries = DocumentArray(
            [Document(id=doc.id, embedding=doc.embedding) for doc in query_data]
        )

        def fuse_matches(global_matches: DocumentArray, local_matches: DocumentArray):
            global_matches.extend(local_matches)
            global_matches = sorted(
                global_matches,
                key=lambda x: x.scores[distance].value,
            )[:limit]
            return DocumentArray(global_matches)

        for batch in index_data.batch(match_batch_size):
            if (
                apply_sampling
                or (batch.embeddings is None)
                or (batch[0].embedding[0] == 0)
            ):
                if embed_funcs:
                    embed_funcs[1](batch)
                else:
                    batch.embed(**embed_args[1])

            local_queries.match(
                batch,
                limit=limit,
                metric=distance,
                normalization=normalization,
                exclude_self=exclude_self,
                use_scipy=use_scipy,
                num_worker=num_worker,
                device=device,
                batch_size=int(len(batch) / num_worker) if num_worker > 1 else None,
                only_id=True,
            )

            for doc in local_queries:
                query_data[doc.id, 'matches'] = fuse_matches(
                    query_data[doc.id].matches,
                    doc.matches,
                )

            batch.embeddings = None
        # set labels if necessary
        if not ground_truth:
            for i, doc in enumerate(query_data):
                new_matches = DocumentArray()
                for m in doc.matches:
                    m.tags = {label_tag: index_data_labels[m.id]}
                    new_matches.append(m)
                query_data[doc.id, 'matches'] = new_matches

        if ground_truth and label_tag in ground_truth[0].tags:
            num_relevant_documents_per_label = dict(
                Counter([d.tags[label_tag] for d in ground_truth])
            )
        elif not ground_truth and label_tag in query_data[0].tags:
            num_relevant_documents_per_label = dict(
                Counter([d.tags[label_tag] for d in query_data])
            )
        else:
            num_relevant_documents_per_label = None

        metrics_resp = query_data.evaluate(
            ground_truth=ground_truth,
            metrics=metrics,
            metric_names=metric_names,
            strict=strict,
            label_tag=label_tag,
            num_relevant_documents_per_label=num_relevant_documents_per_label,
            **kwargs,
        )

        return metrics_resp
