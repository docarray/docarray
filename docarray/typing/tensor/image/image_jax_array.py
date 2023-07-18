from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.image.abstract_image_tensor import AbstractImageTensor
from docarray.typing.tensor.jaxarray import JaxArray, metaJax

MAX_INT_16 = 2**15


@_register_proto(proto_type_name='image_jaxarray')
class ImageJaxArray(JaxArray, AbstractImageTensor, metaclass=metaJax):
    ...
