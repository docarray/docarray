from qdrant_openapi_client.models.models import Distance

DISTANCES = {
    'cosine': Distance.COSINE,
    'euclidean': Distance.EUCLID,
    'dot': Distance.DOT,
}
