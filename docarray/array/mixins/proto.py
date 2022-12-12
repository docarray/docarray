from typing import TYPE_CHECKING, Dict, Type

from docarray.array.abstract_array import AbstractDocumentArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from docarray.proto import DocumentArrayProto, NodeProto


class ProtoArrayMixin(AbstractDocumentArray):
    @classmethod
    def from_protobuf(
        cls: Type[AbstractDocumentArray], pb_msg: 'DocumentArrayProto'
    ) -> AbstractDocumentArray:
        """create a Document from a protobuf message"""

        content_type = pb_msg.WhichOneof('content')

        if content_type == 'list_':
            return cls(cls.document_type.from_protobuf(od) for od in pb_msg.list_.docs)
        elif content_type == 'stack':
            da = cls(
                cls.document_type.from_protobuf(od) for od in pb_msg.stack.list_.docs
            )
            da._columns = pb_msg.stack.columns
            return da
        else:
            raise ValueError(f'proto message content {content_type} is not supported')

    def to_protobuf(self) -> 'DocumentArrayProto':
        """Convert DocumentArray into a Protobuf message.

        :param ndarray_type: can be ``list`` or ``numpy``,
            if set it will force all ndarray-like object from all
            Documents to ``List`` or ``numpy.ndarray``.
        :return: the protobuf message
        """

        from docarray.proto import (
            DocumentArrayListProto,
            DocumentArrayProto,
            DocumentArrayStackedProto,
            UnionArrayProto,
        )

        if self.is_stacked() and self._columns is not None:
            da_proto = DocumentArrayListProto()
            for doc in self.__iter_over_stacked_documents__():
                da_proto.docs.append(doc.to_protobuf())

            columns_proto: Dict[str, UnionArrayProto] = dict()
            for field, column in self._columns.items():
                if isinstance(column, ProtoArrayMixin):
                    columns_proto[field] = UnionArrayProto(
                        document_array=DocumentArrayProto(stack=column.to_protobuf())
                    )
                elif isinstance(column, AbstractTensor):
                    columns_proto[field] = UnionArrayProto(ndarray=column.to_protobuf())

            da_proto = DocumentArrayStackedProto(list_=da_proto, columns=columns_proto)
            return DocumentArrayProto(stack=da_proto)
        else:
            da_proto = DocumentArrayListProto()
            for doc in self:
                da_proto.docs.append(doc.to_protobuf())

            return DocumentArrayProto(list_=da_proto)

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert a DocumentArray into a NodeProto protobuf message.
         This function should be called when a DocumentArray
        is nested into another Document that need to be converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(chunks=self.to_protobuf())
