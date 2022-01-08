import hashlib
import json
from typing import Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...types import T


class FeatureHashMixin:
    """Provide helper functions for feature hashing."""

    def embed_feature_hashing(
        self: 'T',
        n_dim: int = 256,
        sparse: bool = False,
        fields: Tuple[str, ...] = ('text', 'tags'),
        max_value: int = 1_000_000,
    ) -> 'T':
        """Convert an arbitrary set of attributes into a fixed-dimensional matrix using the hashing trick.

        :param n_dim: the dimensionality of each document in the output embedding.
            Small numbers of features are likely to cause hash collisions,
            but large numbers will cause larger overall parameter dimensions.
        :param sparse: whether the resulting feature matrix should be a sparse csr_matrix or dense ndarray.
            Note that this feature requires ``scipy``
        :param fields: which attributes to be considered as for feature hashing.
        """
        if sparse:
            from scipy.sparse import csr_matrix

        idxs, data = [], []  # sparse
        table = np.zeros(n_dim)  # dense

        for f in fields:
            if 'text' in fields:
                all_tokens = self.get_vocabulary(('text',))
                for f_id, val in all_tokens.items():
                    _hash_column(f_id, val, n_dim, max_value, idxs, data, table)

            if 'tags' in fields:
                for k, v in self.tags.items():
                    _hash_column(k, v, n_dim, max_value, idxs, data, table)

            v = getattr(self, f, None)
            if v:
                _hash_column(f, v, n_dim, max_value, idxs, data, table)

        if sparse:
            self.embedding = csr_matrix((data, zip(*idxs)), shape=(1, n_dim))
        else:
            self.embedding = table
        return self


def _hash_column(col_name, col_val, n_dim, max_value, idxs, data, table):
    h = _any_hash(col_name)
    col_val = _any_hash(col_val) % max_value
    col = h % n_dim
    idxs.append((0, col))
    data.append(np.sign(h) * col_val)
    table[col] += np.sign(h) * col_val


def _any_hash(v):
    try:
        return int(v)  # parse int parameter
    except ValueError:
        try:
            return float(v)  # parse float parameter
        except ValueError:
            if not v:
                # ignore it when the parameter is empty
                return 0
            if isinstance(v, str):
                v = v.strip()
                if v.lower() in {'true', 'yes'}:  # parse boolean parameter
                    return 1
                if v.lower() in {'false', 'no'}:
                    return 0
            if isinstance(v, (tuple, dict, list)):
                v = json.dumps(v, sort_keys=True)

    return int(hashlib.md5(str(v).encode('utf-8')).hexdigest(), base=16)
