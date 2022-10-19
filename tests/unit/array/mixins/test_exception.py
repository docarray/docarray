import numpy as np
import pytest

from docarray import DocumentArray


def test_embedding_ops_error():
    da = DocumentArray.empty(100)
    db = DocumentArray.empty(100)
    da.embeddings = np.random.random([100, 256])

    da[2].embedding = None
    da[3].embedding = None

    with pytest.raises(ValueError, match='[2, 3]'):
        da.embeddings

    db.embeddings = np.random.random([100, 256])
    with pytest.raises(ValueError, match='[2, 3]'):
        da.match(db)
    with pytest.raises(ValueError, match='[2, 3]'):
        db.match(da)
    with pytest.raises(ValueError, match='[2, 3]'):
        db.find(da)
    with pytest.raises(ValueError, match='[2, 3]'):
        da.find(db)

    da.embeddings = None
    with pytest.raises(ValueError, match='Did you forget to set'):
        da.find(db)
    db.embeddings = None
    with pytest.raises(ValueError, match='Did you forget to set'):
        da.find(db)
    with pytest.raises(ValueError, match='Did you forget to set'):
        db.find(da)
    da.embeddings = np.random.random([100, 256])
    with pytest.raises(
        ValueError, match='filter must be dict or str when query is None'
    ):
        da.find(None)
