from docarray import Document
from docarray.typing import PointCloudUrl


def test_set_point_cloud_url():
    class MyDocument(Document):
        point_cloud_url: PointCloudUrl

    d = MyDocument(point_cloud_url="https://jina.ai/mesh.obj")

    assert isinstance(d.point_cloud_url, PointCloudUrl)
    assert d.point_cloud_url == "https://jina.ai/mesh.obj"
