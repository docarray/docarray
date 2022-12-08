from typing import TYPE_CHECKING, Type

from docarray.array.abstract_array import AbstractDocumentArray

if TYPE_CHECKING:
    from docarray.proto import DocumentArrayProto, NodeProto


class ProtoArrayMixin(AbstractDocumentArray):
    @classmethod
    def from_protobuf(
        cls: Type[AbstractDocumentArray], pb_msg: 'DocumentArrayProto'
    ) -> AbstractDocumentArray:
        """create a Document from a protobuf message"""

        return cls(cls.document_type.from_protobuf(od) for od in pb_msg.docs)

    def to_protobuf(self) -> 'DocumentArrayProto':
        """Convert DocumentArray into a Protobuf message.

        :param ndarray_type: can be ``list`` or ``numpy``,
            if set it will force all ndarray-like object from all
            Documents to ``List`` or ``numpy.ndarray``.
        :return: the protobuf message
        """

        from docarray.proto import DocumentArrayProto

        dap = DocumentArrayProto()
        for doc in self:
            dap.docs.append(doc.to_protobuf())
        return dap

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert a DocumentArray into a NodeProto protobuf message.
         This function should be called when a DocumentArray
        is nested into another Document that need to be converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(chunks=self.to_protobuf())
