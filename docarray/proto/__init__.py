from google.protobuf import __version__ as __pb__version__

if __pb__version__.startswith('4'):
    print('importing pb4')
    from .pb.docarray_pb2 import DocumentProto, DocumentArrayProto, NdArrayProto
else:
    print('importing pb3')
    from .pb2.docarray_pb2 import DocumentProto, DocumentArrayProto, NdArrayProto
