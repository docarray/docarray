from typing import TYPE_CHECKING, Any, Generic, List, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


from docarray.base_doc.base_node import BaseNode

T = TypeVar('T')
ShapeT = TypeVar('ShapeT')

tensor_base: type = type(BaseNode)


# the mypy error suppression below should not be necessary anymore once the following
# is released in mypy: https://github.com/python/mypy/pull/14135
class metaNumpy(AbstractTensor.__parametrized_meta__, tensor_base):  # type: ignore
    pass


@_register_proto(proto_type_name='jaxarray')
class JaxArray(np.ndarray, AbstractTensor, Generic[ShapeT]):
    """
    Subclass of `np.ndarray`, intended for use in a Document.
    This enables (de)serialization from/to protobuf and json, data validation,
    and coersion from compatible types like `torch.Tensor`.

    This type can also be used in a parametrized way, specifying the shape of the array.

    ---

    ```python
    from docarray import BaseDoc
    from docarray.typing import NdArray
    import numpy as np


    class MyDoc(BaseDoc):
        arr: NdArray
        image_arr: NdArray[3, 224, 224]
        square_crop: NdArray[3, 'x', 'x']
        random_image: NdArray[3, ...]  # first dimension is fixed, can have arbitrary shape


    # create a document with tensors
    doc = MyDoc(
        arr=np.zeros((128,)),
        image_arr=np.zeros((3, 224, 224)),
        square_crop=np.zeros((3, 64, 64)),
        random_image=np.zeros((3, 128, 256)),
    )
    assert doc.image_arr.shape == (3, 224, 224)

    # automatic shape conversion
    doc = MyDoc(
        arr=np.zeros((128,)),
        image_arr=np.zeros((224, 224, 3)),  # will reshape to (3, 224, 224)
        square_crop=np.zeros((3, 128, 128)),
        random_image=np.zeros((3, 64, 128)),
    )
    assert doc.image_arr.shape == (3, 224, 224)

    # !! The following will raise an error due to shape mismatch !!
    from pydantic import ValidationError

    try:
        doc = MyDoc(
            arr=np.zeros((128,)),
            image_arr=np.zeros((224, 224)),  # this will fail validation
            square_crop=np.zeros((3, 128, 64)),  # this will also fail validation
            random_image=np.zeros((4, 64, 128)),  # this will also fail validation
        )
    except ValidationError as e:
        pass
    ```

    ---
    """

    __parametrized_meta__ = metaNumpy

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        pass

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, List[Any], Tuple[Any], Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        pass

    @classmethod
    def _docarray_from_native(cls: Type[T], value: np.ndarray) -> T:
        pass
