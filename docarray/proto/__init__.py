from google.protobuf import __version__ as __pb__version__

if __pb__version__.startswith('4'):
    from .pb.docarray_pb2 import DocumentProto, DocumentArrayProto, NdArrayProto

    # compatibility with any dependent that imports docarray_pb2 (for example jina)
    from .pb import docarray_pb2
else:
    from .pb2.docarray_pb2 import DocumentProto, DocumentArrayProto, NdArrayProto

    # compatibility with any dependent that imports docarray_pb2 (for example jina)
    from .pb2 import docarray_pb2
