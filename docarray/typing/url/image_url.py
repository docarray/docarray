import numpy as np

from docarray.proto import NodeProto
from docarray.typing.url.any_url import AnyUrl


class ImageUrl(AnyUrl):
    def _to_node_protobuf(self) -> NodeProto:
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(image_url=str(self))

    def load(self) -> np.ndarray:
        """
        transform the url in a image Tensor

        this is just a patch we will move the function from old docarray
        :return: tensor image
        """

        return np.zeros((3, 224, 224))
