from typing import Optional

from docarray.document import BaseDocument
from docarray.typing import AnyTensor, Embedding, PointCloud3DUrl


class PointCloud3D(BaseDocument):
    """
    Document for handling point clouds for 3D data representation.

    Point cloud is a representation of a 3D mesh. It is made by repeatedly and uniformly
    sampling points within the surface of the 3D body. Compared to the mesh
    representation, the point cloud is a fixed size ndarray (shape=(n_samples, 3)) and
    hence easier for deep learning algorithms to handle.

    A PointCloud3D Document can contain an PointCloud3DUrl (`PointCloud3D.url`), a
    Tensor (`PointCloud3D.tensor`), and an Embedding (`PointCloud3D.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray import PointCloud3D

        # use it directly
        pc = PointCloud3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        pc.tensor = pc.url.load(samples=100)
        model = MyEmbeddingModel()
        pc.embedding = model(pc.tensor)

    You can extend this Document:

    .. code-block:: python

        from docarray import PointCloud3D
        from docarray.typing import Embedding
        from typing import Optional

        # extend it
        class MyPointCloud3D(PointCloud3D):
            second_embedding: Optional[Embedding]


        pc = MyPointCloud3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        pc.tensor = pc.url.load(samples=100)
        model = MyEmbeddingModel()
        pc.embedding = model(pc.tensor)
        pc.second_embedding = model(pc.tensor)


    You can use this Document for composition:

    .. code-block:: python

        from docarray import Document, PointCloud3D, Text

        # compose it
        class MultiModalDoc(Document):
            point_cloud: PointCloud3D
            text: Text


        mmdoc = MultiModalDoc(
            point_cloud=PointCloud3D(
                url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'
            ),
            text=Text(text='hello world, how are you doing?'),
        )
        mmdoc.point_cloud.tensor = mmdoc.point_cloud.url.load(samples=100)
    """

    url: Optional[PointCloud3DUrl]
    tensor: Optional[AnyTensor]
    embedding: Optional[Embedding]
