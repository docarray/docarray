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
from docarray.utils._internal.misc import is_jax_available

jax_available = is_jax_available()
if jax_available:
    import jax.numpy as jnp

    from docarray.typing import JaxArray


@pytest.fixture()
@pytest.mark.jax
def batch():

    import jax.numpy as jnp

    class Image(BaseDoc):
        tensor: JaxArray[3, 224, 224]

    batch = DocList[Image]([Image(tensor=jnp.zeros((3, 224, 224))) for _ in range(10)])

    return batch.to_doc_vec()


@pytest.fixture()
@pytest.mark.jax
def nested_batch():
    class Image(BaseDoc):
        tensor: JaxArray[3, 224, 224]

    class MMdoc(BaseDoc):
        img: DocList[Image]

    batch = DocVec[MMdoc](
        [
            MMdoc(
                img=DocList[Image](
                    [Image(tensor=jnp.zeros((3, 224, 224))) for _ in range(10)]
                )
            )
            for _ in range(10)
        ]
    )

    return batch


@pytest.mark.jax
def test_len(batch):
    assert len(batch) == 10


@pytest.mark.jax
def test_getitem(batch):
    for i in range(len(batch)):
        item = batch[i]
        assert isinstance(item.tensor, JaxArray)
        assert jnp.allclose(item.tensor.tensor, jnp.zeros((3, 224, 224)))


@pytest.mark.jax
def test_get_slice(batch):
    sliced = batch[0:2]
    assert isinstance(sliced, DocVec)
    assert len(sliced) == 2


@pytest.mark.jax
def test_iterator(batch):
    for doc in batch:
        assert jnp.allclose(doc.tensor.tensor, jnp.zeros((3, 224, 224)))


@pytest.mark.jax
def test_set_after_stacking():
    class Image(BaseDoc):
        tensor: JaxArray[3, 224, 224]

    batch = DocVec[Image]([Image(tensor=jnp.zeros((3, 224, 224))) for _ in range(10)])

    batch.tensor = jnp.ones((10, 3, 224, 224))
    assert jnp.allclose(batch.tensor.tensor, jnp.ones((10, 3, 224, 224)))
    for i, doc in enumerate(batch):
        assert jnp.allclose(doc.tensor.tensor, batch.tensor.tensor[i])


@pytest.mark.jax
def test_stack_optional(batch):
    assert jnp.allclose(
        batch._storage.tensor_columns['tensor'].tensor, jnp.zeros((10, 3, 224, 224))
    )
    assert jnp.allclose(batch.tensor.tensor, jnp.zeros((10, 3, 224, 224)))


@pytest.mark.jax
def test_stack_mod_nested_document():
    class Image(BaseDoc):
        tensor: JaxArray[3, 224, 224]

    class MMdoc(BaseDoc):
        img: Image

    batch = DocList[MMdoc](
        [MMdoc(img=Image(tensor=jnp.zeros((3, 224, 224)))) for _ in range(10)]
    ).to_doc_vec()

    assert jnp.allclose(
        batch._storage.doc_columns['img']._storage.tensor_columns['tensor'].tensor,
        jnp.zeros((10, 3, 224, 224)),
    )

    assert jnp.allclose(batch.img.tensor.tensor, jnp.zeros((10, 3, 224, 224)))


@pytest.mark.jax
def test_stack_nested_DocArray(nested_batch):
    for i in range(len(nested_batch)):
        assert jnp.allclose(
            nested_batch[i].img._storage.tensor_columns['tensor'].tensor,
            jnp.zeros((10, 3, 224, 224)),
        )

        assert jnp.allclose(
            nested_batch[i].img.tensor.tensor, jnp.zeros((10, 3, 224, 224))
        )


@pytest.mark.jax
def test_convert_to_da(batch):
    da = batch.to_doc_list()

    for doc in da:
        assert jnp.allclose(doc.tensor.tensor, jnp.zeros((3, 224, 224)))


@pytest.mark.jax
def test_unstack_nested_document():
    class Image(BaseDoc):
        tensor: JaxArray[3, 224, 224]

    class MMdoc(BaseDoc):
        img: Image

    batch = DocVec[MMdoc](
        [MMdoc(img=Image(tensor=jnp.zeros((3, 224, 224)))) for _ in range(10)]
    )
    assert isinstance(batch.img._storage.tensor_columns['tensor'], JaxArray)
    da = batch.to_doc_list()

    for doc in da:
        assert jnp.allclose(doc.img.tensor.tensor, jnp.zeros((3, 224, 224)))


@pytest.mark.jax
def test_unstack_nested_DocArray(nested_batch):
    batch = nested_batch.to_doc_list()
    for i in range(len(batch)):
        assert isinstance(batch[i].img, DocList)
        for doc in batch[i].img:
            assert jnp.allclose(doc.tensor.tensor, jnp.zeros((3, 224, 224)))


@pytest.mark.jax
def test_stack_call():
    class Image(BaseDoc):
        tensor: JaxArray[3, 224, 224]

    da = DocList[Image]([Image(tensor=jnp.zeros((3, 224, 224))) for _ in range(10)])

    da = da.to_doc_vec()

    assert len(da) == 10

    assert da.tensor.tensor.shape == (10, 3, 224, 224)


@pytest.mark.jax
def test_stack_union():
    class Image(BaseDoc):
        tensor: Union[JaxArray[3, 224, 224], NdArray[3, 224, 224]]

    DocVec[Image](
        [Image(tensor=jnp.zeros((3, 224, 224))) for _ in range(10)],
        tensor_type=JaxArray,
    )

    # union fields aren't actually doc_vec
    # just checking that there is no error


@pytest.mark.jax
def test_setitem_tensor(batch):
    batch[3].tensor.tensor = jnp.zeros((3, 224, 224))


@pytest.mark.jax
@pytest.mark.skip('not working yet')
def test_setitem_tensor_direct(batch):
    batch[3].tensor = jnp.zeros((3, 224, 224))


@pytest.mark.jax
@pytest.mark.parametrize(
    'cls_tensor', [ImageTensor, AudioTensor, VideoTensor, AnyEmbedding, AnyTensor]
)
def test_generic_tensors_with_jnp(cls_tensor):
    tensor = jnp.zeros((3, 224, 224))

    class Image(BaseDoc):
        tensor: cls_tensor

    da = DocVec[Image](
        [Image(tensor=tensor) for _ in range(10)],
        tensor_type=JaxArray,
    )

    for i in range(len(da)):
        assert jnp.allclose(da[i].tensor.tensor, tensor)

    assert 'tensor' in da._storage.tensor_columns.keys()
    assert isinstance(da._storage.tensor_columns['tensor'], JaxArray)


@pytest.mark.jax
@pytest.mark.parametrize(
    'cls_tensor', [ImageTensor, AudioTensor, VideoTensor, AnyEmbedding, AnyTensor]
)
def test_generic_tensors_with_optional(cls_tensor):
    tensor = jnp.zeros((3, 224, 224))

    class Image(BaseDoc):
        tensor: Optional[cls_tensor] = None

    class TopDoc(BaseDoc):
        img: Image

    da = DocVec[TopDoc](
        [TopDoc(img=Image(tensor=tensor)) for _ in range(10)],
        tensor_type=JaxArray,
    )

    for i in range(len(da)):
        assert jnp.allclose(da.img[i].tensor.tensor, tensor)

    assert 'tensor' in da.img._storage.tensor_columns.keys()
    assert isinstance(da.img._storage.tensor_columns['tensor'], JaxArray)
    assert isinstance(da.img._storage.tensor_columns['tensor'].tensor, jnp.ndarray)


@pytest.mark.jax
def test_get_from_slice_stacked():
    class Doc(BaseDoc):
        text: str
        tensor: JaxArray

    da = DocVec[Doc](
        [Doc(text=f'hello{i}', tensor=jnp.zeros((3, 224, 224))) for i in range(10)]
    )

    da_sliced = da[0:10:2]
    assert isinstance(da_sliced, DocVec)

    tensors = da_sliced.tensor.tensor
    assert tensors.shape == (5, 3, 224, 224)


@pytest.mark.jax
def test_stack_none():
    class MyDoc(BaseDoc):
        tensor: Optional[AnyTensor] = None

    da = DocVec[MyDoc]([MyDoc(tensor=None) for _ in range(10)], tensor_type=JaxArray)
    assert 'tensor' in da._storage.tensor_columns.keys()


@pytest.mark.jax
def test_keep_dtype_jnp():
    class MyDoc(BaseDoc):
        tensor: JaxArray

    da = DocList[MyDoc](
        [MyDoc(tensor=jnp.zeros([2, 4], dtype=jnp.int32)) for _ in range(3)]
    )
    assert da[0].tensor.tensor.dtype == jnp.int32

    da = da.to_doc_vec()
    assert da[0].tensor.tensor.dtype == jnp.int32
    assert da.tensor.tensor.dtype == jnp.int32
