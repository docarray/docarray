from typing import Dict, Optional, Union

import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDoc, DocList
from docarray.array import DocVec
from docarray.documents import ImageDoc
from docarray.typing import AnyEmbedding, AnyTensor, NdArray, TorchTensor


@pytest.fixture()
def batch():
    class ImageDoc(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    batch = DocVec[ImageDoc](
        [ImageDoc(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    return batch


@pytest.fixture()
def nested_batch():
    class ImageDoc(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    class MMdoc(BaseDoc):
        img: DocList[ImageDoc]

    batch = DocList[MMdoc](
        [
            MMdoc(
                img=DocList[ImageDoc](
                    [ImageDoc(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
                )
            )
            for _ in range(10)
        ]
    )

    return batch.to_doc_vec()


def test_create_from_list_docs():
    list_ = [ImageDoc(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    da_stacked = DocVec[ImageDoc](docs=list_, tensor_type=TorchTensor)
    assert len(da_stacked) == 10
    assert da_stacked.tensor.shape == tuple([10, 3, 224, 224])


def test_len(batch):
    assert len(batch) == 10


def test_create_from_None():
    with pytest.raises(ValueError):
        DocVec[ImageDoc]([])


def test_getitem(batch):
    for i in range(len(batch)):
        assert (batch[i].tensor == torch.zeros(3, 224, 224)).all()


def test_iterator(batch):
    for doc in batch:
        assert (doc.tensor == torch.zeros(3, 224, 224)).all()


def test_stack_setter():
    class ImageDoc(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    batch = DocList[ImageDoc](
        [ImageDoc(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    batch = batch.to_doc_vec()
    batch.tensor = torch.ones(10, 3, 224, 224)

    assert (batch.tensor == torch.ones(10, 3, 224, 224)).all()

    for i, doc in enumerate(batch):
        assert (doc.tensor == batch.tensor[i]).all()


def test_stack_setter_np():
    class ImageDoc(BaseDoc):
        tensor: NdArray[3, 224, 224]

    batch = DocList[ImageDoc](
        [ImageDoc(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    batch = batch.to_doc_vec()
    batch.tensor = np.ones((10, 3, 224, 224))

    assert (batch.tensor == np.ones((10, 3, 224, 224))).all()

    for i, doc in enumerate(batch):
        assert (doc.tensor == batch.tensor[i]).all()


def test_stack_optional(batch):
    assert (
        batch._storage.tensor_columns['tensor'] == torch.zeros(10, 3, 224, 224)
    ).all()
    assert (batch.tensor == torch.zeros(10, 3, 224, 224)).all()


def test_stack_numpy():
    class ImageDoc(BaseDoc):
        tensor: NdArray[3, 224, 224]

    batch = DocList[ImageDoc](
        [ImageDoc(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    batch = batch.to_doc_vec()

    assert (
        batch._storage.tensor_columns['tensor'] == np.zeros((10, 3, 224, 224))
    ).all()
    assert (batch.tensor == np.zeros((10, 3, 224, 224))).all()
    assert (
        batch.tensor.ctypes.data == batch._storage.tensor_columns['tensor'].ctypes.data
    )


def test_stack(batch):
    assert (
        batch._storage.tensor_columns['tensor'] == torch.zeros(10, 3, 224, 224)
    ).all()
    assert (batch.tensor == torch.zeros(10, 3, 224, 224)).all()
    assert batch._storage.tensor_columns['tensor'].data_ptr() == batch.tensor.data_ptr()

    for doc, tensor in zip(batch, batch.tensor):
        assert doc.tensor.data_ptr() == tensor.data_ptr()

    for i in range(len(batch)):
        assert batch[i].tensor.data_ptr() == batch.tensor[i].data_ptr()


def test_stack_mod_nested_document():
    class ImageDoc(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    class MMdoc(BaseDoc):
        img: ImageDoc

    batch = DocList[MMdoc](
        [MMdoc(img=ImageDoc(tensor=torch.zeros(3, 224, 224))) for _ in range(10)]
    )

    batch = batch.to_doc_vec()

    assert (
        batch._storage.doc_columns['img']._storage.tensor_columns['tensor']
        == torch.zeros(10, 3, 224, 224)
    ).all()

    assert (batch.img.tensor == torch.zeros(10, 3, 224, 224)).all()

    assert (
        batch._storage.doc_columns['img']._storage.tensor_columns['tensor'].data_ptr()
        == batch.img.tensor.data_ptr()
    )


def test_stack_nested_DocArray(nested_batch):
    for i in range(len(nested_batch)):
        assert (
            nested_batch[i].img._storage.tensor_columns['tensor']
            == torch.zeros(10, 3, 224, 224)
        ).all()
        assert (nested_batch[i].img.tensor == torch.zeros(10, 3, 224, 224)).all()
        assert (
            nested_batch[i].img._storage.tensor_columns['tensor'].data_ptr()
            == nested_batch[i].img.tensor.data_ptr()
        )


def test_convert_to_da(batch):
    class ImageDoc(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    batch = DocList[ImageDoc](
        [ImageDoc(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    batch = batch.to_doc_vec()
    da = batch.to_doc_list()

    for doc in da:
        assert (doc.tensor == torch.zeros(3, 224, 224)).all()


def test_unstack_nested_document():
    class ImageDoc(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    class MMdoc(BaseDoc):
        img: ImageDoc

    batch = DocList[MMdoc](
        [MMdoc(img=ImageDoc(tensor=torch.zeros(3, 224, 224))) for _ in range(10)]
    )

    batch = batch.to_doc_vec()

    da = batch.to_doc_list()

    for doc in da:
        assert (doc.img.tensor == torch.zeros(3, 224, 224)).all()


def test_unstack_nested_DocArray(nested_batch):
    batch = nested_batch.to_doc_list()
    for i in range(len(batch)):
        assert isinstance(batch[i].img, DocList)
        for doc in batch[i].img:
            assert (doc.tensor == torch.zeros(3, 224, 224)).all()


def test_stack_call():
    class ImageDoc(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    da = DocList[ImageDoc](
        [ImageDoc(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    da = da.to_doc_vec()

    assert len(da) == 10

    assert da.tensor.shape == (10, 3, 224, 224)


def test_stack_union():
    class ImageDoc(BaseDoc):
        tensor: Union[NdArray[3, 224, 224], TorchTensor[3, 224, 224]]

    batch = DocList[ImageDoc](
        [ImageDoc(tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )
    batch[3].tensor = np.zeros((3, 224, 224))

    # union fields aren't actually doc_vec
    # just checking that there is no error
    batch.to_doc_vec()


@pytest.mark.parametrize(
    'tensor_type,tensor',
    [(TorchTensor, torch.zeros(3, 224, 224)), (NdArray, np.zeros((3, 224, 224)))],
)
def test_any_tensor_with_torch(tensor_type, tensor):
    class ImageDoc(BaseDoc):
        tensor: AnyTensor

    da = DocVec[ImageDoc](
        [ImageDoc(tensor=tensor) for _ in range(10)],
        tensor_type=tensor_type,
    )

    for i in range(len(da)):
        assert (da[i].tensor == tensor).all()

    assert 'tensor' in da._storage.tensor_columns.keys()
    assert isinstance(da._storage.tensor_columns['tensor'], tensor_type)


def test_any_tensor_with_optional():
    tensor = torch.zeros(3, 224, 224)

    class ImageDoc(BaseDoc):
        tensor: Optional[AnyTensor]

    class TopDoc(BaseDoc):
        img: ImageDoc

    da = DocVec[TopDoc](
        [TopDoc(img=ImageDoc(tensor=tensor)) for _ in range(10)],
        tensor_type=TorchTensor,
    )

    for i in range(len(da)):
        assert (da.img[i].tensor == tensor).all()

    assert 'tensor' in da.img._storage.tensor_columns.keys()
    assert isinstance(da.img._storage.tensor_columns['tensor'], TorchTensor)


def test_dict_stack():
    class MyDoc(BaseDoc):
        my_dict: Dict[str, int]

    da = DocVec[MyDoc]([MyDoc(my_dict={'a': 1, 'b': 2}) for _ in range(10)])

    da.my_dict


def test_get_from_slice_stacked():
    class Doc(BaseDoc):
        text: str
        tensor: NdArray

    N = 10

    da = DocVec[Doc](
        [Doc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N)]
    )

    da_sliced = da[0:10:2]
    assert isinstance(da_sliced, DocVec)

    tensors = da_sliced.tensor
    assert tensors.shape == (5, 3, 224, 224)

    texts = da_sliced.text
    assert len(texts) == 5
    for i, text in enumerate(texts):
        assert text == f'hello{i * 2}'


def test_stack_embedding():
    class MyDoc(BaseDoc):
        embedding: AnyEmbedding

    da = DocVec[MyDoc]([MyDoc(embedding=np.zeros(10)) for _ in range(10)])

    assert 'embedding' in da._storage.tensor_columns.keys()
    assert (da.embedding == np.zeros((10, 10))).all()


@pytest.mark.parametrize('tensor_backend', [TorchTensor, NdArray])
def test_stack_none(tensor_backend):
    class MyDoc(BaseDoc):
        tensor: Optional[AnyTensor]

    da = DocVec[MyDoc](
        [MyDoc(tensor=None) for _ in range(10)], tensor_type=tensor_backend
    )

    assert 'tensor' in da._storage.tensor_columns.keys()


def test_to_device():
    da = DocVec[ImageDoc]([ImageDoc(tensor=torch.zeros(3, 5))], tensor_type=TorchTensor)
    assert da.tensor.device == torch.device('cpu')
    da.to('meta')
    assert da.tensor.device == torch.device('meta')


def test_to_device_with_nested_da():
    class Video(BaseDoc):
        images: DocVec[ImageDoc]

    da_image = DocVec[ImageDoc](
        [ImageDoc(tensor=torch.zeros(3, 5))], tensor_type=TorchTensor
    )

    da = DocVec[Video]([Video(images=da_image)])
    assert da.images[0].tensor.device == torch.device('cpu')
    da.to('meta')
    assert da.images[0].tensor.device == torch.device('meta')


def test_to_device_nested():
    class MyDoc(BaseDoc):
        tensor: TorchTensor
        docs: ImageDoc

    da = DocVec[MyDoc](
        [MyDoc(tensor=torch.zeros(3, 5), docs=ImageDoc(tensor=torch.zeros(3, 5)))],
        tensor_type=TorchTensor,
    )
    assert da.tensor.device == torch.device('cpu')
    assert da.docs.tensor.device == torch.device('cpu')
    da.to('meta')
    assert da.tensor.device == torch.device('meta')
    assert da.docs.tensor.device == torch.device('meta')


def test_to_device_numpy():
    da = DocVec[ImageDoc]([ImageDoc(tensor=np.zeros((3, 5)))], tensor_type=NdArray)
    with pytest.raises(NotImplementedError):
        da.to('meta')


def test_keep_dtype_torch():
    class MyDoc(BaseDoc):
        tensor: TorchTensor

    da = DocList[MyDoc](
        [MyDoc(tensor=torch.zeros([2, 4], dtype=torch.int32)) for _ in range(3)]
    )
    assert da[0].tensor.dtype == torch.int32

    da = da.to_doc_vec()
    assert da[0].tensor.dtype == torch.int32
    assert da.tensor.dtype == torch.int32


def test_keep_dtype_np():
    class MyDoc(BaseDoc):
        tensor: NdArray

    da = DocList[MyDoc](
        [MyDoc(tensor=np.zeros([2, 4], dtype=np.int32)) for _ in range(3)]
    )
    assert da[0].tensor.dtype == np.int32

    da = da.to_doc_vec()
    assert da[0].tensor.dtype == np.int32
    assert da.tensor.dtype == np.int32


def test_del_item(batch):
    assert len(batch) == 10
    assert batch.tensor.shape[0] == 10
    with pytest.raises(NotImplementedError):
        del batch[2]


def test_np_scalar():
    class MyDoc(BaseDoc):
        scalar: NdArray

    da = DocList[MyDoc]([MyDoc(scalar=np.array(2.0)) for _ in range(3)])
    assert all(doc.scalar.ndim == 0 for doc in da)
    assert all(doc.scalar == 2.0 for doc in da)

    stacked_da = da.to_doc_vec()
    assert type(stacked_da.scalar) == NdArray

    assert all(type(doc.scalar) == NdArray for doc in stacked_da)
    assert all(doc.scalar.ndim == 1 for doc in stacked_da)
    assert all(doc.scalar == 2.0 for doc in stacked_da)

    # Make sure they share memory
    stacked_da.scalar[0] = 3.0
    assert stacked_da[0].scalar == 3.0


def test_torch_scalar():
    class MyDoc(BaseDoc):
        scalar: TorchTensor

    da = DocList[MyDoc](
        [MyDoc(scalar=torch.tensor(2.0)) for _ in range(3)],
    )
    assert all(doc.scalar.ndim == 0 for doc in da)
    assert all(doc.scalar == 2.0 for doc in da)
    stacked_da = da.to_doc_vec(tensor_type=TorchTensor)
    assert type(stacked_da.scalar) == TorchTensor

    assert all(type(doc.scalar) == TorchTensor for doc in stacked_da)
    assert all(doc.scalar.ndim == 1 for doc in stacked_da)  # TODO failing here
    assert all(doc.scalar == 2.0 for doc in stacked_da)

    stacked_da.scalar[0] = 3.0
    assert stacked_da[0].scalar == 3.0


def test_np_nan():
    class MyDoc(BaseDoc):
        scalar: Optional[NdArray]

    da = DocList[MyDoc]([MyDoc() for _ in range(3)])
    assert all(doc.scalar is None for doc in da)
    assert all(doc.scalar == doc.scalar for doc in da)
    stacked_da = da.to_doc_vec()
    assert stacked_da.scalar is None

    assert all(doc.scalar is None for doc in stacked_da)
    # Stacking them turns them into np.nan


def test_from_storage():
    class ImageDoc(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    batch = DocVec[ImageDoc](
        [ImageDoc(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    DocVec[ImageDoc].from_columns_storage(batch._storage)


def test_validate_from_da():
    class ImageDoc(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    batch = DocList[ImageDoc](
        [ImageDoc(tensor=torch.zeros(3, 224, 224)) for _ in range(10)]
    )

    da = parse_obj_as(DocVec[ImageDoc], batch)

    assert isinstance(da, DocVec[ImageDoc])


def test_validation_column_tensor(batch):
    batch.tensor = torch.zeros(10, 3, 224, 244)
    assert isinstance(batch.tensor, TorchTensor)


def test_validation_column_tensor_fail(batch):
    with pytest.raises(ValueError):
        batch.tensor = ['hello'] * 10

    with pytest.raises(ValueError):
        batch.tensor = torch.zeros(11, 3, 224, 244)


@pytest.fixture()
def batch_nested_doc():
    class Inner(BaseDoc):
        hello: str

    class Doc(BaseDoc):
        inner: Inner

    batch = DocVec[Doc]([Doc(inner=Inner(hello='hello')) for _ in range(10)])
    return batch, Doc, Inner


def test_validation_column_doc(batch_nested_doc):
    batch, Doc, Inner = batch_nested_doc

    batch.inner = DocList[Inner]([Inner(hello='hello') for _ in range(10)])
    assert isinstance(batch.inner, DocVec[Inner])


def test_validation_list_doc(batch_nested_doc):
    batch, Doc, Inner = batch_nested_doc

    batch.inner = [Inner(hello='hello') for _ in range(10)]
    assert isinstance(batch.inner, DocVec[Inner])


def test_validation_col_doc_fail(batch_nested_doc):
    batch, Doc, Inner = batch_nested_doc

    with pytest.raises(ValueError):
        batch.inner = ['hello'] * 10

    with pytest.raises(ValueError):
        batch.inner = DocList[Inner]([Inner(hello='hello') for _ in range(11)])


def test_doc_view_update(batch):
    batch[0].tensor = 12 * torch.ones(3, 224, 224)
    assert (batch.tensor[0] == 12 * torch.ones(3, 224, 224)).all()


def test_doc_view_nested(batch_nested_doc):
    batch, Doc, Inner = batch_nested_doc
    # batch[0].__fields_set__
    batch[0].inner = Inner(hello='world')
    assert batch.inner[0].hello == 'world'


def test_type_error_no_doc_type():

    with pytest.raises(TypeError):
        DocVec([BaseDoc() for _ in range(10)])


def test_doc_view_dict(batch):
    doc_view = batch[0]
    assert doc_view.is_view()
    d = doc_view.dict()
    assert d['tensor'].shape == (3, 224, 224)
    assert d['id'] == doc_view.id

    doc_view_two = batch[1]
    assert doc_view_two.is_view()
    d = doc_view_two.dict()
    assert d['tensor'].shape == (3, 224, 224)
    assert d['id'] == doc_view_two.id


def test_doc_vec_equality():
    class Text(BaseDoc):
        text: str

    da = DocVec[Text]([Text(text='hello') for _ in range(10)])
    da2 = DocList[Text]([Text(text='hello') for _ in range(10)])

    assert da != da2
    assert da == da2.to_doc_vec()


@pytest.mark.parametrize('tensor_type', [TorchTensor, NdArray])
def test_doc_vec_equality_tensor(tensor_type):
    class Text(BaseDoc):
        tens: tensor_type

    da = DocVec[Text](
        [Text(tens=[1, 2, 3, 4]) for _ in range(10)], tensor_type=tensor_type
    )
    da2 = DocVec[Text](
        [Text(tens=[1, 2, 3, 4]) for _ in range(10)], tensor_type=tensor_type
    )
    assert da == da2

    da2 = DocVec[Text](
        [Text(tens=[1, 2, 3, 4, 5]) for _ in range(10)], tensor_type=tensor_type
    )
    assert da != da2


@pytest.mark.tensorflow
def test_doc_vec_equality_tf():
    from docarray.typing import TensorflowTensor

    class Text(BaseDoc):
        tens: TensorflowTensor

    da = DocVec[Text](
        [Text(tens=[1, 2, 3, 4]) for _ in range(10)], tensor_type=TensorflowTensor
    )
    da2 = DocVec[Text](
        [Text(tens=[1, 2, 3, 4]) for _ in range(10)], tensor_type=TensorflowTensor
    )
    assert da == da2

    da2 = DocVec[Text](
        [Text(tens=[1, 2, 3, 4, 5]) for _ in range(10)], tensor_type=TensorflowTensor
    )
    assert da != da2


def test_doc_vec_nested(batch_nested_doc):
    batch, Doc, Inner = batch_nested_doc
    batch2 = DocVec[Doc]([Doc(inner=Inner(hello='hello')) for _ in range(10)])

    assert batch == batch2


def test_doc_vec_tensor_type():
    class ImageDoc(BaseDoc):
        tensor: AnyTensor

    da = DocVec[ImageDoc]([ImageDoc(tensor=np.zeros((3, 224, 224))) for _ in range(10)])

    da2 = DocVec[ImageDoc](
        [ImageDoc(tensor=torch.zeros(3, 224, 224)) for _ in range(10)],
        tensor_type=TorchTensor,
    )

    assert da != da2
