import sys
from typing import Optional, Union

print(f"getattr(sys, 'base_prefix', None) = {getattr(sys, 'base_prefix', None)}")
print(f"getattr(sys, 'real_prefix', None) = {getattr(sys, 'real_prefix', None)}")
print(f"sys.prefix = {sys.prefix}")

import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow._api.v2.experimental.numpy as tnp  # noqa: E402

from docarray import BaseDocument, DocumentArray  # noqa: E402
from docarray.array import DocumentArrayStacked  # noqa: E402
from docarray.typing import AnyTensor, NdArray, TensorFlowTensor  # noqa: E402


@pytest.fixture()
def batch():
    class Image(BaseDocument):
        tensor: TensorFlowTensor[3, 224, 224]

    import tensorflow as tf

    batch = DocumentArray[Image](
        [Image(tensor=tf.zeros((3, 224, 224))) for _ in range(10)]
    )

    return batch.stack()


@pytest.mark.tensorflow
def test_len(batch):
    assert len(batch) == 10


@pytest.mark.tensorflow
def test_getitem(batch):
    for i in range(len(batch)):
        assert tnp.allclose(batch[i].tensor.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_iterator(batch):
    for doc in batch:
        assert tnp.allclose(doc.tensor.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_stack_setter(batch):

    batch.tensor = tf.ones((10, 3, 224, 224))

    assert tnp.allclose(batch.tensor, tf.ones((10, 3, 224, 224)))


@pytest.mark.tensorflow
def test_set_after_stacking(batch):
    class Image(BaseDocument):
        tensor: TensorFlowTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [Image(tensor=tf.zeros((3, 224, 224))) for _ in range(10)]
    )

    batch = batch.stack()
    batch.tensor.tensor = tf.ones((10, 3, 224, 224))
    for i, doc in enumerate(batch):
        assert tnp.allclose(doc.tensor.tensor, batch.tensor.tensor[i])


@pytest.mark.tensorflow
def test_stack_optional(batch):

    assert tnp.allclose(batch._columns['tensor'].tensor, tf.zeros((10, 3, 224, 224)))
    assert tnp.allclose(batch.tensor.tensor, tf.zeros((10, 3, 224, 224)))


@pytest.mark.tensorflow
def test_stack_mod_nested_document():
    class Image(BaseDocument):
        tensor: TensorFlowTensor[3, 224, 224]

    class MMdoc(BaseDocument):
        img: Image

    batch = DocumentArray[MMdoc](
        [MMdoc(img=Image(tensor=tf.zeros((3, 224, 224)))) for _ in range(10)]
    )

    batch = batch.stack()

    assert tnp.allclose(
        batch._columns['img']._columns['tensor'].tensor, tf.zeros((10, 3, 224, 224))
    )

    assert tnp.allclose(batch.img.tensor.tensor, tf.zeros((10, 3, 224, 224)))


@pytest.mark.tensorflow
def test_convert_to_da(batch):
    da = batch.unstack()

    for doc in da:
        assert tnp.allclose(doc.tensor.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_unstack_nested_document():
    class Image(BaseDocument):
        tensor: TensorFlowTensor[3, 224, 224]

    class MMdoc(BaseDocument):
        img: Image

    batch = DocumentArray[MMdoc](
        [MMdoc(img=Image(tensor=tf.zeros((3, 224, 224)))) for _ in range(10)]
    )

    batch = batch.stack()
    da = batch.unstack()

    for doc in da:
        assert tnp.allclose(doc.img.tensor.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_stack_call():
    class Image(BaseDocument):
        tensor: TensorFlowTensor[3, 224, 224]

    da = DocumentArray[Image](
        [Image(tensor=tf.zeros((3, 224, 224))) for _ in range(10)]
    )

    da = da.stack()

    assert len(da) == 10

    assert da.tensor.tensor.shape == (10, 3, 224, 224)


@pytest.mark.tensorflow
def test_context_manager():
    class Image(BaseDocument):
        tensor: TensorFlowTensor[3, 224, 224]

    da = DocumentArray[Image](
        [Image(tensor=tf.zeros((3, 224, 224))) for _ in range(10)]
    )

    with da.stacked_mode() as da:
        assert len(da) == 10

        assert da.tensor.tensor.shape == ((10, 3, 224, 224))

        da.tensor = tf.ones((10, 3, 224, 224))

    tensor = da.tensor

    assert isinstance(tensor, list)
    for doc in da:
        assert tnp.allclose(doc.tensor.tensor, tf.ones((3, 224, 224)))


@pytest.mark.tensorflow
def test_stack_union():
    class Image(BaseDocument):
        tensor: Union[NdArray[3, 224, 224], TensorFlowTensor[3, 224, 224]]

    batch = DocumentArray[Image](
        [Image(tensor=tf.zeros((3, 224, 224))) for _ in range(10)]
    )
    batch[3].tensor = tf.zeros((3, 224, 224))

    # union fields aren't actually stacked
    # just checking that there is no error
    batch.stack()


@pytest.mark.tensorflow
def test_any_tensor_with_tf():
    tensor = tf.zeros((3, 224, 224))

    class Image(BaseDocument):
        tensor: AnyTensor

    da = DocumentArray[Image](
        [Image(tensor=tensor) for _ in range(10)],
        tensor_type=TensorFlowTensor,
    ).stack()

    for i in range(len(da)):
        assert tnp.allclose(da[i].tensor.tensor, tensor)

    assert 'tensor' in da._columns.keys()
    assert isinstance(da._columns['tensor'], TensorFlowTensor)


@pytest.mark.tensorflow
def test_any_tensor_with_optional():
    tensor = tf.zeros((3, 224, 224))

    class Image(BaseDocument):
        tensor: Optional[AnyTensor]

    class TopDoc(BaseDocument):
        img: Image

    da = DocumentArray[TopDoc](
        [TopDoc(img=Image(tensor=tensor)) for _ in range(10)],
        tensor_type=TensorFlowTensor,
    ).stack()

    for i in range(len(da)):
        assert tnp.allclose(da.img[i].tensor.tensor, tensor)

    assert 'tensor' in da.img._columns.keys()
    assert isinstance(da.img._columns['tensor'], TensorFlowTensor)
    assert isinstance(da.img._columns['tensor'].tensor, tf.Tensor)


@pytest.mark.tensorflow
def test_get_from_slice_stacked():
    class Doc(BaseDocument):
        text: str
        tensor: TensorFlowTensor

    da = DocumentArray[Doc](
        [Doc(text=f'hello{i}', tensor=tf.zeros((3, 224, 224))) for i in range(10)]
    ).stack()

    da_sliced = da[0:10:2]
    assert isinstance(da_sliced, DocumentArrayStacked)

    tensors = da_sliced.tensor.tensor
    assert tensors.shape == (5, 3, 224, 224)


@pytest.mark.tensorflow
def test_stack_none():
    class MyDoc(BaseDocument):
        tensor: Optional[AnyTensor]

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=None) for _ in range(10)], tensor_type=TensorFlowTensor
    ).stack()

    assert 'tensor' in da._columns.keys()


@pytest.mark.tensorflow
def test_keep_dtype_tf():
    class MyDoc(BaseDocument):
        tensor: TensorFlowTensor

    da = DocumentArray[MyDoc](
        [MyDoc(tensor=tf.zeros([2, 4], dtype=tf.int32)) for _ in range(3)]
    )
    assert da[0].tensor.tensor.dtype == tf.int32

    da = da.stack()
    assert da[0].tensor.tensor.dtype == tf.int32
    assert da.tensor.tensor.dtype == tf.int32
