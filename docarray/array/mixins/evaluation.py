import warnings
from typing import Optional, Union, TYPE_CHECKING, Callable, List, Dict

from functools import wraps

import numpy as np
from collections import defaultdict

from docarray.score import NamedScore

if TYPE_CHECKING:  # pragma: no cover
    from docarray import Document, DocumentArray


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

        :param metrics: list of metric names or metric functions to be computed
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
            max_rel = caller_max_rel or len(gd.matches)
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

            targets = gd.matches[:max_rel]

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
