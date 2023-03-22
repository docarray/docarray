from docarray.remote.file import FileDocStore
from docarray.remote.jinaai import JACDocStore
from docarray.remote.s3 import S3DocStore

__all__ = ['JACDocStore', 'FileDocStore', 'S3DocStore']
