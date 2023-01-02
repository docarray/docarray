import numpy as np
import torch

from docarray import Document, DocumentArray, Image, Text
from docarray.typing import (
    AnyTensor,
    AnyUrl,
    Embedding,
    ImageUrl,
    Mesh3DUrl,
    NdArray,
    PointCloud3DUrl,
    TextUrl,
    TorchEmbedding,
    TorchTensor,
)
from docarray.typing.tensor import NdArrayEmbedding


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
    class NestedDoc(Document):
        tensor: NdArray

    class MyDoc(Document):
        img_url: ImageUrl
        txt_url: TextUrl
        mesh_url: Mesh3DUrl
        point_cloud_url: PointCloud3DUrl
        any_url: AnyUrl
        torch_tensor: TorchTensor
        torch_tensor_param: TorchTensor[224, 224, 3]
        np_array: NdArray
        np_array_param: NdArray[224, 224, 3]
        generic_nd_array: AnyTensor
        generic_torch_tensor: AnyTensor
        embedding: Embedding
        torch_embedding: TorchEmbedding[128]
        np_embedding: NdArrayEmbedding[128]
        nested_docs: DocumentArray[NestedDoc]

    doc = MyDoc(
        img_url='test.png',
        txt_url='test.txt',
        mesh_url='test.obj',
        point_cloud_url='test.obj',
        any_url='www.jina.ai',
        torch_tensor=torch.zeros((3, 224, 224)),
        torch_tensor_param=torch.zeros((3, 224, 224)),
        np_array=np.zeros((3, 224, 224)),
        np_array_param=np.zeros((3, 224, 224)),
        generic_nd_array=np.zeros((3, 224, 224)),
        generic_torch_tensor=torch.zeros((3, 224, 224)),
        embedding=np.zeros((3, 224, 224)),
        torch_embedding=torch.zeros((128,)),
        np_embedding=np.zeros((128,)),
        nested_docs=DocumentArray[NestedDoc]([NestedDoc(tensor=np.zeros((128,)))]),
    )
    doc = MyDoc.from_protobuf(doc.to_protobuf())

    assert doc.img_url == 'test.png'
    assert doc.txt_url == 'test.txt'
    assert doc.mesh_url == 'test.obj'
    assert doc.point_cloud_url == 'test.obj'
    assert doc.any_url == 'www.jina.ai'

    assert (doc.torch_tensor == torch.zeros((3, 224, 224))).all()
    assert isinstance(doc.torch_tensor, torch.Tensor)

    assert (doc.torch_tensor_param == torch.zeros((224, 224, 3))).all()
    assert isinstance(doc.torch_tensor_param, torch.Tensor)

    assert (doc.np_array == np.zeros((3, 224, 224))).all()
    assert isinstance(doc.np_array, np.ndarray)
    assert doc.np_array.flags.writeable

    assert (doc.np_array_param == np.zeros((224, 224, 3))).all()
    assert isinstance(doc.np_array_param, np.ndarray)

    assert (doc.generic_nd_array == np.zeros((3, 224, 224))).all()
    assert isinstance(doc.generic_nd_array, np.ndarray)

    assert (doc.generic_torch_tensor == torch.zeros((3, 224, 224))).all()
    assert isinstance(doc.generic_torch_tensor, torch.Tensor)

    assert (doc.torch_embedding == torch.zeros((128,))).all()
    assert isinstance(doc.torch_embedding, torch.Tensor)

    assert (doc.np_embedding == np.zeros((128,))).all()
    assert isinstance(doc.np_embedding, np.ndarray)

    assert (doc.embedding == np.zeros((3, 224, 224))).all()
