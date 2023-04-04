from typing import TYPE_CHECKING

from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from google.protobuf import __version__ as __pb__version__
else:
    protobuf = import_library('google.protobuf', raise_error=True)
    __pb__version__ = protobuf.__version__


if __pb__version__.startswith('4'):
    from docarray.proto.pb.docarray_pb2 import (
        DictOfAnyProto,
        DocArrayStackedProto,
        DocumentArrayProto,
        DocumentProto,
        ListOfAnyProto,
        ListOfDocArrayProto,
        NdArrayProto,
        NodeProto,
    )
else:
    from docarray.proto.pb2.docarray_pb2 import (
        DictOfAnyProto,
        DocArrayStackedProto,
        DocumentArrayProto,
        DocumentProto,
        ListOfAnyProto,
        ListOfDocArrayProto,
        NdArrayProto,
        NodeProto,
    )

__all__ = [
    'DocumentArrayProto',
    'DocumentProto',
    'NdArrayProto',
    'NodeProto',
    'DocArrayStackedProto',
    'DocumentArrayProto',
    'ListOfDocArrayProto',
    'ListOfAnyProto',
    'DictOfAnyProto',
]
