from google.protobuf import __version__ as __pb__version__

if __pb__version__.startswith('4'):
    from .pb.docarray_pb2 import *
else:
    from .pb2.docarray_pb2 import *
