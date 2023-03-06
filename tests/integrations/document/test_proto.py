import numpy as np
import pytest
import torch

from docarray import BaseDocument, DocumentArray
from docarray.documents import ImageDoc, Text
from docarray.typing import (
    AnyEmbedding,
    AnyTensor,
    AnyUrl,
    ImageBytes,
    ImageUrl,
    Mesh3DUrl,
    NdArray,
    PointCloud3DUrl,
    TextUrl,
    TorchEmbedding,
    TorchTensor,
)
from docarray.typing.tensor import NdArrayEmbedding
from docarray.utils.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp

    from docarray.typing import TensorFlowEmbedding, TensorFlowTensor


@pytest.mark.proto
def test_multi_modal_doc_proto():
    class MyMultiModalDoc(BaseDocument):
        image: ImageDoc
        text: Text

    doc = MyMultiModalDoc(
        image=ImageDoc(tensor=np.zeros((3, 224, 224))), text=Text(text='hello')
    )

    MyMultiModalDoc.from_protobuf(doc.to_protobuf())


@pytest.mark.proto
def test_all_types():
    class NestedDoc(BaseDocument):
        tensor: NdArray

    class MyDoc(BaseDocument):
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
        embedding: AnyEmbedding
        torch_embedding: TorchEmbedding[128]
        np_embedding: NdArrayEmbedding[128]
        nested_docs: DocumentArray[NestedDoc]
        bytes_: bytes
        img_bytes: ImageBytes

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
        bytes_=b'hello',
        img_bytes=b'img',
    )
    doc = doc.to_protobuf()
    doc = MyDoc.from_protobuf(doc)

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

    assert doc.bytes_ == b'hello'
    assert doc.img_bytes == b'img'


@pytest.mark.tensorflow
def test_tensorflow_types():
    class NestedDoc(BaseDocument):
        tensor: TensorFlowTensor

    class MyDoc(BaseDocument):
        tf_tensor: TensorFlowTensor
        tf_tensor_param: TensorFlowTensor[224, 224, 3]
        generic_tf_tensor: AnyTensor
        embedding: AnyEmbedding
        tf_embedding: TensorFlowEmbedding[128]
        nested_docs: DocumentArray[NestedDoc]

    doc = MyDoc(
        tf_tensor=tf.zeros((3, 224, 224)),
        tf_tensor_param=tf.zeros((3, 224, 224)),
        generic_tf_tensor=tf.zeros((3, 224, 224)),
        embedding=tf.zeros((3, 224, 224)),
        tf_embedding=tf.zeros((128,)),
        nested_docs=DocumentArray[NestedDoc]([NestedDoc(tensor=tf.zeros((128,)))]),
    )
    doc = doc.to_protobuf()
    doc = MyDoc.from_protobuf(doc)

    assert tnp.allclose(doc.tf_tensor.tensor, tf.zeros((3, 224, 224)))
    assert isinstance(doc.tf_tensor.tensor, tf.Tensor)
    assert isinstance(doc.tf_tensor, TensorFlowTensor)

    assert tnp.allclose(doc.tf_tensor_param.tensor, tf.zeros((224, 224, 3)))
    assert isinstance(doc.tf_tensor_param.tensor, tf.Tensor)
    assert isinstance(doc.tf_tensor_param, TensorFlowTensor)

    assert tnp.allclose(doc.generic_tf_tensor.tensor, tf.zeros((3, 224, 224)))
    assert isinstance(doc.generic_tf_tensor.tensor, tf.Tensor)
    assert isinstance(doc.generic_tf_tensor, TensorFlowTensor)

    assert tnp.allclose(doc.tf_embedding.tensor, tf.zeros((128,)))
    assert isinstance(doc.tf_embedding.tensor, tf.Tensor)
    assert isinstance(doc.tf_embedding, TensorFlowTensor)

    assert tnp.allclose(doc.embedding.tensor, tf.zeros((3, 224, 224)))
    assert isinstance(doc.embedding.tensor, tf.Tensor)
    assert isinstance(doc.embedding, TensorFlowTensor)
