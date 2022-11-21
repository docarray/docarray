import numpy as np
import torch

from docarray import Document
from docarray.document import AnyDocument
from docarray.typing import AnyUrl, Embedding, ImageUrl, Tensor, TorchTensor


def test_proto_all_types():
    class Mymmdoc(Document):
        tensor: Tensor
        torch_tensor: TorchTensor
        embedding: Embedding
        any_url: AnyUrl
        image_url: ImageUrl

    doc = Mymmdoc(
        tensor=np.zeros((3, 224, 224)),
        torch_tensor=torch.zeros((3, 224, 224)),
        embedding=np.zeros((100, 1)),
        any_url='http://jina.ai',
        image_url='http://jina.ai/bla.jpg',
    )

    new_doc = AnyDocument.from_protobuf(doc.to_protobuf())

    for field, value in new_doc:
        assert isinstance(value, doc._get_nested_document_class(field))
