import random
from collections import defaultdict
from typing import Dict, Any, TYPE_CHECKING, Generator, List

import numpy as np

from docarray.helper import dunder_get

if TYPE_CHECKING:
    from docarray import DocumentArray


class GroupMixin:
    """These helpers yield groups of :class:`DocumentArray` from
    a source :class:`DocumentArray`."""

    def split_by_tag(self, tag: str) -> Dict[Any, 'DocumentArray']:
        """Split the `DocumentArray` into multiple DocumentArray according to the tag value of each `Document`.

        :param tag: the tag name to split stored in tags.
        :return: a dict where Documents with the same value on `tag` are grouped together, their orders
            are preserved from the original :class:`DocumentArray`.

        .. note::
            If the :attr:`tags` of :class:`Document` do not contains the specified :attr:`tag`,
            return an empty dict.
        """
        from docarray import DocumentArray

        rv = defaultdict(DocumentArray)
        for doc in self:
            if '__' in tag:
                value = dunder_get(doc.tags, tag)
            elif tag in doc.tags:
                value = doc.tags[tag]
            else:
                continue
            rv[value].append(doc)
        return dict(rv)

    def batch(
        self,
        batch_size: int,
        shuffle: bool = False,
        show_progress: bool = False,
    ) -> Generator['DocumentArray', None, None]:
        """
        Creates a `Generator` that yields `DocumentArray` of size `batch_size` until `docs` is fully traversed along
        the `traversal_path`. The None `docs` are filtered out and optionally the `docs` can be filtered by checking for
        the existence of a `Document` attribute.
        Note, that the last batch might be smaller than `batch_size`.

        :param batch_size: Size of each generated batch (except the last one, which might be smaller, default: 32)
        :param shuffle: If set, shuffle the Documents before dividing into minibatches.
        :param show_progress: if set, show a progress bar when batching documents.
        :yield: a Generator of `DocumentArray`, each in the length of `batch_size`
        """
        from rich.progress import track

        if not (isinstance(batch_size, int) and batch_size > 0):
            raise ValueError('`batch_size` should be a positive integer')

        N = len(self)
        ix = list(range(N))
        n_batches = int(np.ceil(N / batch_size))

        if shuffle:
            random.shuffle(ix)

        for i in track(
            range(n_batches),
            description='Batching documents',
            disable=not show_progress,
        ):
            yield self[ix[i * batch_size : (i + 1) * batch_size]]

    def batch_ids(
        self,
        batch_size: int,
        shuffle: bool = False,
    ) -> Generator[List[str], None, None]:
        """
        Creates a `Generator` that yields `lists of ids` of size `batch_size` until `self` is fully traversed.
        Note, that the last batch might be smaller than `batch_size`.

        :param batch_size: Size of each generated batch (except the last one, which might be smaller)
        :param shuffle: If set, shuffle the Documents before dividing into minibatches.
        :yield: a Generator of `list` of IDs, each in the length of `batch_size`
        """

        if not (isinstance(batch_size, int) and batch_size > 0):
            raise ValueError('`batch_size` should be a positive integer')

        N = len(self)
        ix = self[:, 'id']
        n_batches = int(np.ceil(N / batch_size))

        if shuffle:
            random.shuffle(ix)

        for i in range(n_batches):
            yield ix[i * batch_size : (i + 1) * batch_size]
