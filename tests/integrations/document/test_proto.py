import numpy as np
import torch

from docarray import Document, Image, Text
from docarray.typing import (
    AnyUrl,
    Embedding,
    ImageUrl,
    NdArray,
    Tensor,
    TextUrl,
    TorchTensor,
)


def test_multi_modal_doc_proto():
    class MyMultiModalDoc(Document):
        image: Image
        text: Text

    class MySUperDoc(Document):
        doc: MyMultiModalDoc
        description: str

    doc = MyMultiModalDoc(
        image=Image(tensor=np.zeros((3, 224, 224))), text=Text(text='hello')
    )

    MyMultiModalDoc.from_protobuf(doc.to_protobuf())


def test_all_types():
    class MyDoc(Document):
        img_url: ImageUrl
        txt_url: TextUrl
        any_url: AnyUrl
        torch_tensor: TorchTensor
        torch_tensor_param: TorchTensor[224, 224, 3]
        np_array: NdArray
        np_array_param: NdArray[224, 224, 3]
        generic_nd_array: Tensor
        generic_torch_tensor: Tensor
        embedding: Embedding

    doc = MyDoc(
        img_url='test.png',
        txt_url='test.txt',
        any_url='www.jina.ai',
        torch_tensor=torch.zeros((3, 224, 224)),
        torch_tensor_param=torch.zeros((3, 224, 224)),
        np_array=np.zeros((3, 224, 224)),
        np_array_param=np.zeros((3, 224, 224)),
        generic_nd_array=np.zeros((3, 224, 224)),
        generic_torch_tensor=torch.zeros((3, 224, 224)),
        embedding=np.zeros((3, 224, 224)),
    )
    doc = MyDoc.from_protobuf(doc.to_protobuf())

    assert doc.img_url == 'test.png'
    assert doc.txt_url == 'test.txt'
    assert doc.any_url == 'www.jina.ai'

    assert (doc.torch_tensor == torch.zeros((3, 224, 224))).all()
    assert isinstance(doc.torch_tensor, torch.Tensor)

    assert (doc.torch_tensor_param == torch.zeros((224, 224, 3))).all()
    assert isinstance(doc.torch_tensor_param, torch.Tensor)

    assert (doc.np_array == np.zeros((3, 224, 224))).all()
    assert isinstance(doc.np_array, np.ndarray)

    assert (doc.np_array_param == np.zeros((224, 224, 3))).all()
    assert isinstance(doc.np_array_param, np.ndarray)

    assert (doc.generic_nd_array == np.zeros((3, 224, 224))).all()
    assert isinstance(doc.generic_nd_array, np.ndarray)

    assert (doc.generic_torch_tensor == torch.zeros((3, 224, 224))).all()
    assert isinstance(doc.generic_torch_tensor, torch.Tensor)

    assert (doc.embedding == np.zeros((3, 224, 224))).all()
