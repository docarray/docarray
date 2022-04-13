from qdrant_client.http.models.models import Distance

DISTANCES = {
    'cosine': Distance.COSINE,
    'euclidean': Distance.EUCLID,
    'dot': Distance.DOT,
}
