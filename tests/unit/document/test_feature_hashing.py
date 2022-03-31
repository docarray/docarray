import pytest

from docarray import DocumentArray, Document
from docarray.math.ndarray import to_numpy_array


@pytest.mark.parametrize('n_dim', [2, 4, 100])
@pytest.mark.parametrize('sparse', [True, False])
@pytest.mark.parametrize('metric', ['jaccard', 'cosine'])
def test_feature_hashing(n_dim, sparse, metric):
    da = DocumentArray.empty(6)
    da.texts = [
        'hello world',
        'world, bye',
        'hello bye',
        'infinity test',
        'nan test',
        '2.3 test',
    ]
    da.apply(lambda d: d.embed_feature_hashing(n_dim=n_dim, sparse=sparse))
    assert da.embeddings.shape == (6, n_dim)
    da.embeddings = to_numpy_array(da.embeddings)
    da.match(da, metric=metric, use_scipy=True)
    result = da['@m', ('id', f'scores__{metric}__value')]
    assert len(result) == 2
    assert result[1][0] == 0.0
    assert result[1][1] > 0.0
