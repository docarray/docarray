import numpy as np
import torch

from docarray import Document
from docarray.document import AnyDocument
from docarray.typing import (
    AnyUrl,
    Embedding,
    ImageUrl,
    MeshUrl,
    NdArray,
    PointCloudUrl,
    TextUrl,
    TorchTensor,
)


def test_proto_all_types():
    class Mymmdoc(Document):
        tensor: NdArray
        torch_tensor: TorchTensor
        embedding: Embedding
        any_url: AnyUrl
        image_url: ImageUrl
        text_url: TextUrl
        mesh_url: MeshUrl
        point_cloud_url: PointCloudUrl

    doc = Mymmdoc(
        tensor=np.zeros((3, 224, 224)),
        torch_tensor=torch.zeros((3, 224, 224)),
        embedding=np.zeros((100, 1)),
        any_url='http://jina.ai',
        image_url='http://jina.ai/bla.jpg',
        text_url='http://jina.ai',
        mesh_url='http://jina.ai/mesh.obj',
        point_cloud_url='http://jina.ai/mesh.obj',
    )

    new_doc = AnyDocument.from_protobuf(doc.to_protobuf())

    for field, value in new_doc:
        if field == 'embedding':
            # embedding is a Union type, not supported by isinstance
            assert isinstance(value, np.ndarray) or isinstance(value, torch.Tensor)
        else:
            assert isinstance(value, doc._get_nested_document_class(field))
