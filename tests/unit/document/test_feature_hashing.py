import pytest

from docarray import DocumentArray, Document
from docarray.math.ndarray import to_numpy_array


@pytest.mark.parametrize('n_dim', [4])
@pytest.mark.parametrize('sparse', [True])
@pytest.mark.parametrize('metric', ['cosine'])
def test_feature_hashing(n_dim, sparse, metric):
    da = DocumentArray([Document(id=str(i)) for i in range(6)])
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
    for doc in da:
        assert doc.matches[0].scores[metric].value == pytest.approx(0.0)
        assert doc.matches[1].scores[metric].value > 0.0
