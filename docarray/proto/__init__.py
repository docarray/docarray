from google.protobuf import __version__ as __pb__version__

if __pb__version__.startswith('4'):
    from docarray.proto.pb.docarray_pb2 import (
        DocumentArrayProto,
        DocumentArrayStackedProto,
        DocumentProto,
        ListOfAnyProto,
        ListOfDocumentArrayProto,
        NdArrayProto,
        NodeProto,
    )
else:
    from docarray.proto.pb2.docarray_pb2 import (
        DocumentArrayProto,
        DocumentArrayStackedProto,
        DocumentProto,
        ListOfAnyProto,
        ListOfDocumentArrayProto,
        NdArrayProto,
        NodeProto,
    )

__all__ = [
    'DocumentArrayProto',
    'DocumentProto',
    'NdArrayProto',
    'NodeProto',
    'DocumentArrayStackedProto',
    'DocumentArrayProto',
    'ListOfDocumentArrayProto',
    'ListOfAnyProto',
]
