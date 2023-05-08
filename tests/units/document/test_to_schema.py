import numpy as np

from docarray import BaseDoc
from docarray.base_doc.io.json import orjson_dumps
from docarray.typing import NdArray, TensorFlowTensor, TorchTensor


class NpDoc(BaseDoc):
    embedding: NdArray[3, 4]
    embedding_no_shape: NdArray


class TorchDoc(BaseDoc):
    embedding: TorchTensor[3, 4]
    embedding_no_shape: TorchTensor


class TensorflowDoc(BaseDoc):
    embedding: TensorFlowTensor[3, 4]
    embedding_no_shape: TensorFlowTensor


def test_np_schema():
    schema = NpDoc.schema()
    assert schema['properties']['embedding']['tensor/array shape'] == '[3, 4]'
    assert schema['properties']['embedding']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'
    assert schema['properties']['embedding']['example'] == orjson_dumps(
        np.zeros([3, 4])
    )

    assert (
        schema['properties']['embedding_no_shape']['tensor/array shape']
        == 'not specified'
    )
    assert schema['properties']['embedding_no_shape']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'


def test_torch_schema():
    schema = TorchDoc.schema()
    assert schema['properties']['embedding']['tensor/array shape'] == '[3, 4]'
    assert schema['properties']['embedding']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'
    assert schema['properties']['embedding']['example'] == orjson_dumps(
        np.zeros([3, 4])
    )

    assert (
        schema['properties']['embedding_no_shape']['tensor/array shape']
        == 'not specified'
    )
    assert schema['properties']['embedding_no_shape']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'


def test_tensorflow_schema():
    schema = TensorflowDoc.schema()
    assert schema['properties']['embedding']['tensor/array shape'] == '[3, 4]'
    assert schema['properties']['embedding']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'
    assert schema['properties']['embedding']['example'] == orjson_dumps(
        np.zeros([3, 4])
    )

    assert (
        schema['properties']['embedding_no_shape']['tensor/array shape']
        == 'not specified'
    )
    assert schema['properties']['embedding_no_shape']['type'] == 'array'
    assert schema['properties']['embedding']['items']['type'] == 'number'
