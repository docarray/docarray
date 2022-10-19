import warnings
from typing import Optional, Union, TYPE_CHECKING, Callable

import numpy as np

from docarray.score import NamedScore

if TYPE_CHECKING:
    from docarray import Document, DocumentArray


def _evaluate_deprecation(f):
    """Raises a deprecation warning if the user executes the evaluate function with
    the old interface and adjust the input to fit the new interface."""

    def func(*args, **kwargs):
        if len(args) > 1:
            if not (isinstance(args[1], Callable) or isinstance(args[1], str)):
                kwargs['ground_truth'] = args[1]
                args = [args[0]] + list(args[2:])
                warnings.warn(
                    'The `other` attribute in `evaluate()` is transfered from a '
                    'positional attribute into the keyword attribute `ground_truth`.'
                    'Using it as a positional attribute is deprecated and will be removed '
                    'soon.',
                    DeprecationWarning,
                )
        if 'other' in kwargs:
            kwargs['ground_truth'] = kwargs['other']
            warnings.warn(
                '`other` is renamed to `groundtruth` in `evaluate()`, the usage of `other` is '
                'deprecated and will be removed soon.',
                DeprecationWarning,
            )
        return f(*args, **kwargs)

    return func


class EvaluationMixin:
    """A mixin that provides ranking evaluation functionality to DocumentArrayLike objects"""

    @_evaluate_deprecation
    def evaluate(
        self,
        metric: Union[str, Callable[..., float]],
        ground_truth: Optional['DocumentArray'] = None,
        hash_fn: Optional[Callable[['Document'], str]] = None,
        metric_name: Optional[str] = None,
        strict: bool = True,
        label_tag: str = 'label',
        **kwargs,
    ) -> Optional[float]:
        """
        Compute ranking evaluation metrics for a given `DocumentArray` when compared
        with a groundtruth.

        If one provides a `ground_truth` DocumentArray that is structurally identical
        to `self`, this function compares the `matches` of `documents` inside the
        `DocumentArray` to this `ground_truth`.
        Alternatively, one can directly annotate the documents by adding labels in the
        form of tags with the key specified in the `label_tag` attribute.
        Those tags need to be added to `self` as well as to the documents in the
        matches properties.

        This method will fill the `evaluations` field of Documents inside this
        `DocumentArray` and will return the average of the computations

        :param metric: The name of the metric, or multiple metrics to be computed
        :param ground_truth: The ground_truth `DocumentArray` that the `DocumentArray`
            compares to.
        :param hash_fn: The function used for identifying the uniqueness of Documents.
            If not given, then ``Document.id`` is used.
        :param metric_name: If provided, the results of the metrics computation will be
            stored in the `evaluations` field of each Document. If not provided, the
            name will be computed based on the metrics name.
        :param strict: If set, then left and right sides are required to be fully
            aligned: on the length, and on the semantic of length. These are preventing
            you to evaluate on irrelevant matches accidentally.
        :param label_tag: Specifies the tag which contains the labels.
        :param kwargs: Additional keyword arguments to be passed to `metric_fn`
        :return: The average evaluation computed or a list of them if multiple metrics
            are required
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

        if callable(metric):
            metric_fn = metric
        elif isinstance(metric, str):
            from docarray.math import evaluation

            metric_fn = getattr(evaluation, metric)

        metric_name = metric_name or metric_fn.__name__
        results = []
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

            r = metric_fn(binary_relevance, max_rel=max_rel, **kwargs)
            d.evaluations[metric_name] = NamedScore(
                value=r, op_name=str(metric_fn), ref_id=d.id
            )
            results.append(r)
        if results:
            return float(np.mean(results))
