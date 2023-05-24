from typing import Optional, Union

import pytest

from docarray import BaseDoc, DocList
from docarray.array import DocVec
from docarray.typing import (
    AnyEmbedding,
    AnyTensor,
    AudioTensor,
    ImageTensor,
    NdArray,
    VideoTensor,
)
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp

    from docarray.typing import TensorFlowTensor


@pytest.fixture()
def batch():
    class Image(BaseDoc):
        tensor: TensorFlowTensor[3, 224, 224]

    import tensorflow as tf

    batch = DocList[Image]([Image(tensor=tf.zeros((3, 224, 224))) for _ in range(10)])

    return batch.to_doc_vec()


@pytest.fixture()
def nested_batch():
    class Image(BaseDoc):
        tensor: TensorFlowTensor[3, 224, 224]

    class MMdoc(BaseDoc):
        img: DocList[Image]

    import tensorflow as tf

    batch = DocVec[MMdoc](
        [
            MMdoc(
                img=DocList[Image](
                    [Image(tensor=tf.zeros((3, 224, 224))) for _ in range(10)]
                )
            )
            for _ in range(10)
        ]
    )

    return batch


@pytest.mark.tensorflow
def test_len(batch):
    assert len(batch) == 10


@pytest.mark.tensorflow
def test_getitem(batch):
    for i in range(len(batch)):
        item = batch[i]
        assert isinstance(item.tensor, TensorFlowTensor)
        assert tnp.allclose(item.tensor.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_get_slice(batch):
    sliced = batch[0:2]
    assert isinstance(sliced, DocVec)
    assert len(sliced) == 2


@pytest.mark.tensorflow
def test_iterator(batch):
    for doc in batch:
        assert tnp.allclose(doc.tensor.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_set_after_stacking():
    class Image(BaseDoc):
        tensor: TensorFlowTensor[3, 224, 224]

    batch = DocVec[Image]([Image(tensor=tf.zeros((3, 224, 224))) for _ in range(10)])

    batch.tensor = tf.ones((10, 3, 224, 224))
    assert tnp.allclose(batch.tensor.tensor, tf.ones((10, 3, 224, 224)))
    for i, doc in enumerate(batch):
        assert tnp.allclose(doc.tensor.tensor, batch.tensor.tensor[i])


@pytest.mark.tensorflow
def test_stack_optional(batch):
    assert tnp.allclose(
        batch._storage.tensor_columns['tensor'].tensor, tf.zeros((10, 3, 224, 224))
    )
    assert tnp.allclose(batch.tensor.tensor, tf.zeros((10, 3, 224, 224)))


@pytest.mark.tensorflow
def test_stack_mod_nested_document():
    class Image(BaseDoc):
        tensor: TensorFlowTensor[3, 224, 224]

    class MMdoc(BaseDoc):
        img: Image

    batch = DocList[MMdoc](
        [MMdoc(img=Image(tensor=tf.zeros((3, 224, 224)))) for _ in range(10)]
    ).to_doc_vec()

    assert tnp.allclose(
        batch._storage.doc_columns['img']._storage.tensor_columns['tensor'].tensor,
        tf.zeros((10, 3, 224, 224)),
    )

    assert tnp.allclose(batch.img.tensor.tensor, tf.zeros((10, 3, 224, 224)))


@pytest.mark.tensorflow
def test_stack_nested_DocArray(nested_batch):
    for i in range(len(nested_batch)):
        assert tnp.allclose(
            nested_batch[i].img._storage.tensor_columns['tensor'].tensor,
            tf.zeros((10, 3, 224, 224)),
        )

        assert tnp.allclose(
            nested_batch[i].img.tensor.tensor, tf.zeros((10, 3, 224, 224))
        )


@pytest.mark.tensorflow
def test_convert_to_da(batch):
    da = batch.to_doc_list()

    for doc in da:
        assert tnp.allclose(doc.tensor.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_unstack_nested_document():
    class Image(BaseDoc):
        tensor: TensorFlowTensor[3, 224, 224]

    class MMdoc(BaseDoc):
        img: Image

    batch = DocVec[MMdoc](
        [MMdoc(img=Image(tensor=tf.zeros((3, 224, 224)))) for _ in range(10)]
    )
    assert isinstance(batch.img._storage.tensor_columns['tensor'], TensorFlowTensor)
    da = batch.to_doc_list()

    for doc in da:
        assert tnp.allclose(doc.img.tensor.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_unstack_nested_DocArray(nested_batch):
    batch = nested_batch.to_doc_list()
    for i in range(len(batch)):
        assert isinstance(batch[i].img, DocList)
        for doc in batch[i].img:
            assert tnp.allclose(doc.tensor.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_stack_call():
    class Image(BaseDoc):
        tensor: TensorFlowTensor[3, 224, 224]

    da = DocList[Image]([Image(tensor=tf.zeros((3, 224, 224))) for _ in range(10)])

    da = da.to_doc_vec()

    assert len(da) == 10

    assert da.tensor.tensor.shape == (10, 3, 224, 224)


@pytest.mark.tensorflow
def test_stack_union():
    class Image(BaseDoc):
        tensor: Union[TensorFlowTensor[3, 224, 224], NdArray[3, 224, 224]]

    DocVec[Image](
        [Image(tensor=tf.zeros((3, 224, 224))) for _ in range(10)],
        tensor_type=TensorFlowTensor,
    )

    # union fields aren't actually doc_vec
    # just checking that there is no error


@pytest.mark.tensorflow
def test_setitem_tensor(batch):
    batch[3].tensor.tensor = tf.zeros((3, 224, 224))


@pytest.mark.skip('not working yet')
@pytest.mark.tensorflow
def test_setitem_tensor_direct(batch):
    batch[3].tensor = tf.zeros((3, 224, 224))


@pytest.mark.parametrize(
    'cls_tensor', [ImageTensor, AudioTensor, VideoTensor, AnyEmbedding, AnyTensor]
)
@pytest.mark.tensorflow
def test_generic_tensors_with_tf(cls_tensor):
    tensor = tf.zeros((3, 224, 224))

    class Image(BaseDoc):
        tensor: cls_tensor

    da = DocVec[Image](
        [Image(tensor=tensor) for _ in range(10)],
        tensor_type=TensorFlowTensor,
    )

    for i in range(len(da)):
        assert tnp.allclose(da[i].tensor.tensor, tensor)

    assert 'tensor' in da._storage.tensor_columns.keys()
    assert isinstance(da._storage.tensor_columns['tensor'], TensorFlowTensor)


@pytest.mark.parametrize(
    'cls_tensor', [ImageTensor, AudioTensor, VideoTensor, AnyEmbedding, AnyTensor]
)
@pytest.mark.tensorflow
def test_generic_tensors_with_optional(cls_tensor):
    tensor = tf.zeros((3, 224, 224))

    class Image(BaseDoc):
        tensor: Optional[cls_tensor]

    class TopDoc(BaseDoc):
        img: Image

    da = DocVec[TopDoc](
        [TopDoc(img=Image(tensor=tensor)) for _ in range(10)],
        tensor_type=TensorFlowTensor,
    )

    for i in range(len(da)):
        assert tnp.allclose(da.img[i].tensor.tensor, tensor)

    assert 'tensor' in da.img._storage.tensor_columns.keys()
    assert isinstance(da.img._storage.tensor_columns['tensor'], TensorFlowTensor)
    assert isinstance(da.img._storage.tensor_columns['tensor'].tensor, tf.Tensor)


@pytest.mark.tensorflow
def test_get_from_slice_stacked():
    class Doc(BaseDoc):
        text: str
        tensor: TensorFlowTensor

    da = DocVec[Doc](
        [Doc(text=f'hello{i}', tensor=tf.zeros((3, 224, 224))) for i in range(10)]
    )

    da_sliced = da[0:10:2]
    assert isinstance(da_sliced, DocVec)

    tensors = da_sliced.tensor.tensor
    assert tensors.shape == (5, 3, 224, 224)


@pytest.mark.tensorflow
def test_stack_none():
    class MyDoc(BaseDoc):
        tensor: Optional[AnyTensor]

    da = DocVec[MyDoc](
        [MyDoc(tensor=None) for _ in range(10)], tensor_type=TensorFlowTensor
    )
    assert 'tensor' in da._storage.tensor_columns.keys()


@pytest.mark.tensorflow
def test_keep_dtype_tf():
    class MyDoc(BaseDoc):
        tensor: TensorFlowTensor

    da = DocList[MyDoc](
        [MyDoc(tensor=tf.zeros([2, 4], dtype=tf.int32)) for _ in range(3)]
    )
    assert da[0].tensor.tensor.dtype == tf.int32

    da = da.to_doc_vec()
    assert da[0].tensor.tensor.dtype == tf.int32
    assert da.tensor.tensor.dtype == tf.int32
