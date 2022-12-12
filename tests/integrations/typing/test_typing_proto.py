import numpy as np
import torch

from docarray import Document
from docarray.document import AnyDocument
from docarray.typing import AnyUrl, Embedding, ImageUrl, NdArray, TextUrl, TorchTensor


def test_proto_all_types():
    class Mymmdoc(Document):
        tensor: NdArray
        torch_tensor: TorchTensor
        embedding: Embedding
        any_url: AnyUrl
        image_url: ImageUrl
        text_url: TextUrl

    doc = Mymmdoc(
        tensor=np.zeros((3, 224, 224)),
        torch_tensor=torch.zeros((3, 224, 224)),
        embedding=np.zeros((100, 1)),
        any_url='http://jina.ai',
        image_url='http://jina.ai/bla.jpg',
        text_url='http://jina.ai',
    )

    new_doc = AnyDocument.from_protobuf(doc.__columns__())

    for field, value in new_doc:
        if field == 'embedding':
            # embedding is a Union type, not supported by isinstance
            assert isinstance(value, np.ndarray) or isinstance(value, torch.Tensor)
        else:
            assert isinstance(value, doc._get_nested_document_class(field))
