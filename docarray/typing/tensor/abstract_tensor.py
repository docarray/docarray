import abc
from abc import ABC
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from jaxtyping import (
    AbstractDtype as JaxTypingDType,  # type: ignore  # TODO(johannes) add all the types
)

from docarray.computation import AbstractComputationalBackend
from docarray.typing.abstract_type import AbstractType

if TYPE_CHECKING:
    from docarray.proto import NdArrayProto, NodeProto

T = TypeVar('T', bound='AbstractTensor')
TTensor = TypeVar('TTensor')
ShapeT = TypeVar('ShapeT')


class AbstractTensor(Generic[TTensor, T], AbstractType, ABC):
    _proto_type_name: str
    _dtype_classes: DefaultDict[str, List[Type]] = defaultdict(list)

    def __init_subclass__(cls, **kwargs):
        if issubclass(cls, JaxTypingDType):
            dtype: str = cls.dtypes[
                0
            ]  # assumes only one dtype per subclass. change later
            cls._dtype_classes[dtype].append(cls)

    def _to_node_protobuf(self: T) -> 'NodeProto':
        """Convert itself into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to be
        converted into a protobuf
        :param field: field in which to store the content in the node proto
        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        nd_proto = self.to_protobuf()
        return NodeProto(ndarray=nd_proto, type=self._proto_type_name)

    @staticmethod
    def _parse_item(item: Any) -> Tuple[Optional[str], Optional[str]]:
        # put into tuple form
        if isinstance(item, (int, str)):
            item = (item,)
        try:
            item = tuple(item)
        except TypeError:
            raise TypeError(f'{item} is not a valid tensor shape.')
        # parse shape
        shape_item = (
            i for i in item if not isinstance(i, tuple)
        )  # remove the dtype tuple
        shape = ' '.join([str(s) for s in shape_item]) if shape_item else None
        # parse dtype
        dtype_item = [i for i in item if isinstance(i, tuple)]
        dtype = (
            None
            if (not dtype_item or dtype_item[0][0] != 'dtype')
            else dtype_item[0][1]
        )
        return shape, dtype

    def __class_getitem__(cls, item: Any):
        native_tensor_type = cls.get_comp_backend().native_tensor_type()
        shape, dtype = cls._parse_item(item)
        candidate_classes = cls._dtype_classes[dtype]
        for JaxtypingDtypeClass in candidate_classes:
            if issubclass(JaxtypingDtypeClass, cls):
                return JaxtypingDtypeClass[native_tensor_type, shape]
        raise ValueError(f'{cls} does not support dtype {dtype}')

    @classmethod
    def _docarray_stack(cls: Type[T], seq: Union[List[T], Tuple[T]]) -> T:
        """Stack a sequence of tensors into a single tensor."""
        comp_backend = cls.get_comp_backend()
        # at runtime, 'T' is always the correct input type for .stack()
        # but mypy doesn't know that, so we ignore it here
        return cls._docarray_from_native(comp_backend.stack(seq))  # type: ignore

    @classmethod
    @abc.abstractmethod
    def _docarray_from_native(cls: Type[T], value: Any) -> T:
        """
        Create a DocArray tensor from a tensor that is native to the given framework,
        e.g. from numpy.ndarray or torch.Tensor.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def get_comp_backend() -> AbstractComputationalBackend:
        """The computational backend compatible with this tensor type."""
        ...

    def __getitem__(self, item):
        """Get a slice of this tensor."""
        ...

    def __setitem__(self, index, value):
        """Set a slice of this tensor."""
        ...

    def __iter__(self):
        """Iterate over the elements of this tensor."""
        ...

    @abc.abstractmethod
    def to_protobuf(self) -> 'NdArrayProto':
        """Convert DocumentArray into a Protobuf message"""
        ...

    def unwrap(self):
        """Return the native tensor object that this DocArray tensor wraps."""

    @abc.abstractmethod
    def _docarray_to_json_compatible(self):
        """
        Convert tensor into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        ...
