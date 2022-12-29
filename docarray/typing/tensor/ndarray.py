import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.computation.numpy_backend import NumpyCompBackend
    from docarray.proto import NdArrayProto, NodeProto

T = TypeVar('T', bound='NdArray')
ShapeT = TypeVar('ShapeT')


class NdArray(AbstractTensor, np.ndarray, Generic[ShapeT]):
    """
    Subclass of np.ndarray, intended for use in a Document.
    This enables (de)serialization from/to protobuf and json, data validation,
    and coersion from compatible types like torch.Tensor.

    This type can also be used in a parametrized way, specifying the shape of the array.

    EXAMPLE USAGE

    .. code-block:: python

        from docarray import Document
        from docarray.typing import NdArray
        import numpy as np


        class MyDoc(Document):
            arr: NdArray
            image_arr: NdArray[3, 224, 224]


        # create a document with tensors
        doc = MyDoc(
            arr=np.zeros((128,)),
            image_arr=np.zeros((3, 224, 224)),
        )
        assert doc.image_arr.shape == (3, 224, 224)

        # automatic shape conversion
        doc = MyDoc(
            arr=np.zeros((128,)),
            image_arr=np.zeros((224, 224, 3)),  # will reshape to (3, 224, 224)
        )
        assert doc.image_arr.shape == (3, 224, 224)

        # !! The following will raise an error due to shape mismatch !!
        doc = MyDoc(
            arr=np.zeros((128,)),
            image_arr=np.zeros((224, 224)),  # this will fail validation
        )
    """

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __docarray_validate_shape__(cls, t: T, shape: Tuple[int]) -> T:  # type: ignore
        if t.shape == shape:
            return t
        else:
            warnings.warn(
                f'Tensor shape mismatch. Reshaping array '
                f'of shape {t.shape} to shape {shape}'
            )
            try:
                value = cls.__docarray_from_native__(np.reshape(t, shape))
                return cast(T, value)
            except RuntimeError:
                raise ValueError(
                    f'Cannot reshape array of shape {t.shape} to shape {shape}'
                )

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, List[Any], Tuple[Any], Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, np.ndarray):
            return cls.__docarray_from_native__(value)
        elif isinstance(value, NdArray):
            return cast(T, value)
        elif isinstance(value, list) or isinstance(value, tuple):
            try:
                arr_from_list: np.ndarray = np.asarray(value)
                return cls.__docarray_from_native__(arr_from_list)
            except Exception:
                pass  # handled below
        else:
            try:
                arr: np.ndarray = np.ndarray(value)
                return cls.__docarray_from_native__(arr)
            except Exception:
                pass  # handled below
        raise ValueError(f'Expected a numpy.ndarray compatible type, got {type(value)}')

    @classmethod
    def __docarray_from_native__(cls: Type[T], value: np.ndarray) -> T:
        return value.view(cls)

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        # this is needed to dump to json
        field_schema.update(type='string', format='tensor')

    def _to_json_compatible(self) -> np.ndarray:
        """
        Convert tensor into a json compatible object
        :return: a list representation of the tensor
        """
        return self.unwrap()

    def unwrap(self) -> np.ndarray:
        """
        Return the original ndarray without any memory copy.

        The original view rest intact and is still a Document NdArray
        but the return object is a pure np.ndarray but both object share
        the same memory layout.

        EXAMPLE USAGE
        .. code-block:: python
            from docarray.typing import NdArray
            import numpy as np

            t1 = NdArray.validate(np.zeros((3, 224, 224)), None, None)
            # here t is a docarray TenNdArray
            t2 = t.unwrap()
            # here t2 is a pure np.ndarray but t1 is still a Docarray NdArray
            # But both share the same underlying memory


        :return: a numpy ndarray
        """
        return self.view(np.ndarray)

    def _to_node_protobuf(self: T, field: str = 'ndarray') -> 'NodeProto':
        """Convert itself into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to be
        converted into a protobuf
        :param field: field in which to store the content in the node proto
        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        nd_proto = self.to_protobuf()
        return NodeProto(**{field: nd_proto})

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
        """
        read ndarray from a proto msg
        :param pb_msg:
        :return: a numpy array
        """
        source = pb_msg.dense
        if source.buffer:
            x = np.frombuffer(bytearray(source.buffer), dtype=source.dtype)
            return cls.__docarray_from_native__(x.reshape(source.shape))
        elif len(source.shape) > 0:
            return cls.__docarray_from_native__(np.zeros(source.shape))
        else:
            raise ValueError(f'proto message {pb_msg} cannot be cast to a NdArray')

    def to_protobuf(self) -> 'NdArrayProto':
        """
        transform self into a NdArrayProto protobuf message
        """
        from docarray.proto import NdArrayProto

        nd_proto = NdArrayProto()

        nd_proto.dense.buffer = self.tobytes()
        nd_proto.dense.ClearField('shape')
        nd_proto.dense.shape.extend(list(self.shape))
        nd_proto.dense.dtype = self.dtype.str

        return nd_proto

    @staticmethod
    def get_comp_backend() -> Type['NumpyCompBackend']:
        """Return the computational backend of the tensor"""
        from docarray.computation.numpy_backend import NumpyCompBackend

        return NumpyCompBackend

    def ndim(self) -> int:
        return self.ndim()
