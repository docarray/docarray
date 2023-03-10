from typing import Dict, Iterator, List, Optional, Type

import boto3
import botocore
from smart_open import open
from typing_extensions import TYPE_CHECKING

from docarray.array.array.pushpull.helpers import _from_binary_stream, _to_binary_stream

if TYPE_CHECKING:  # pragma: no cover
    from docarray import BaseDocument, DocumentArray


class PushPullS3:
    """Class to push and pull DocumentArray to and from S3."""

    @staticmethod
    def list(namespace: str, show_table: bool = False) -> List[str]:
        bucket, namespace = namespace.split('/', 1)
        s3 = boto3.resource('s3')
        s3_bucket = s3.Bucket(bucket)
        da_files = [
            obj.key
            for obj in s3_bucket.objects.all()
            if obj.key.startswith(namespace) and obj.key.endswith('.da')
        ]
        da_names = [f.split('/')[-1].split('.')[0] for f in da_files]
        return da_names

    @staticmethod
    def delete(name: str, missing_ok: bool = True) -> bool:
        bucket, name = name.split('/', 1)
        s3 = boto3.resource('s3')
        object = s3.Object(bucket, name + '.da')
        try:
            object.load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                if missing_ok:
                    return False
                else:
                    raise ValueError(f'Object {name} does not exist')
            else:
                raise
        object.delete()
        return True

    @staticmethod
    def push(
        da: 'DocumentArray',
        name: str,
        public: bool = False,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        return PushPullS3.push_stream(iter(da), name, public, show_progress, branding)

    @staticmethod
    def push_stream(
        docs: Iterator['BaseDocument'],
        name: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        bucket, name = name.split('/', 1)
        binary_stream = _to_binary_stream(
            docs, protocol='protobuf', compress='gzip', show_progress=show_progress
        )
        # Upload to S3
        with open(f"s3://{bucket}/{name}.da", 'wb') as fout:
            while True:
                try:
                    fout.write(next(binary_stream))
                except StopIteration:
                    break

        return {}

    @staticmethod
    def pull(
        cls: Type['DocumentArray'],
        name: str,
        show_progress: bool = False,
        local_cache: bool = False,
    ) -> 'DocumentArray':
        da = cls(  # type: ignore
            PushPullS3.pull_stream(
                cls, name, show_progress=show_progress, local_cache=local_cache
            )
        )
        return da

    @staticmethod
    def pull_stream(
        cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDocument']:
        bucket, name = name.split('/', 1)

        return _from_binary_stream(
            cls.document_type,
            open(f"s3://{bucket}/{name}.da", 'rb'),
            protocol='protobuf',
            compress='gzip',
            show_progress=show_progress,
        )
