from typing import Optional

from docarray.document import BaseDocument
from docarray.typing import Embedding, PointCloudUrl, Tensor


class PointCloud(BaseDocument):
    """
    Document for handling point clouds for 3D data representation.
    It can contain an PointCloudUrl (`PointCloud.url`), a Tensor (`PointCloud.tensor`),
    and an Embedding (`PointCloud.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray import PointCloud

        # use it directly
        pc = PointCloud(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        pc.tensor = pc.url.load(samples=100)
        model = MyEmbeddingModel()
        pc.embedding = model(pc.tensor)

    You can extend this Document:

    .. code-block:: python

        from docarray import PointCloud
        from docarray.typing import Embedding
        from typing import Optional

        # extend it
        class MyPointCloud(PointCloud):
            second_embedding: Optional[Embedding]


        pc = MyPointCloud(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        pc.tensor = pc.url.load(samples=100)
        model = MyEmbeddingModel()
        pc.embedding = model(pc.tensor)
        pc.second_embedding = model(pc.tensor)


    You can use this Document for composition:

    .. code-block:: python

        from docarray import Document, PointCloud, Text

        # compose it
        class MultiModalDoc(Document):
            point_cloud: PointCloud
            text: Text


        mmdoc = MultiModalDoc(
            point_cloud=PointCloud(url="https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj"),
            text=Text(text="hello world, how are you doing?"),
        )
        mmdoc.point_cloud.tensor = mmdoc.point_cloud.url.load(samples=100)
    """

    url: Optional[PointCloudUrl]
    tensor: Optional[Tensor]
    embedding: Optional[Embedding]
