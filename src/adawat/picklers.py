from abc import ABC, abstractmethod
import logging
import os
import pickle
import tempfile


log = logging.getLogger(__name__)


class ObjectNotFoundError(RuntimeError):
    """
    This exception is thrown be the `load` methods of picklers to indicate that
    load failed. This is then used by the @stateful decorator to proceed with
    the normal initialization of an object.;
    """
    pass


class Pickler(ABC):
    @abstractmethod
    def dump(self, obj_id: str, obj):
        pass

    @abstractmethod
    def load(self, obj_id: str):
        pass


class FilePickler(Pickler):
    def _filepath(self, key):
        filename = f'f{key}.state'
        filepath = os.path.join(tempfile.gettempdir(), filename)
        return filepath

    def dump(self, obj_id: str, obj):
        filepath = self._filepath(obj_id)
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, obj_id: str):
        filepath = self._filepath(obj_id)

        if not os.path.isfile(filepath):
            raise ObjectNotFoundError(
                f"Couldn't find the file {filepath} while trying to load object with ID {obj_id}.")

        with open(filepath, 'rb') as file:
            return pickle.load(file)


class S3Pickler(Pickler):
    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    def dump(self, obj_id: str, obj):
        obj_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self.client.put_object(
                Body=obj_bytes, Bucket=self.bucket, Key=obj_id)
        except Exception as e:
            # dump() methods silently fail so it doesn't impact execution, so
            # we just log the exception for visibility.
            log.exception(e)

    def load(self, obj_id: str):
        try:
            response = self.client.get_object(Bucket=self.client, Key=obj_id)
            obj_bytes = response['Body'].read()
            return pickle.loads(obj_bytes)
        except Exception as e:
            raise ObjectNotFoundError() from e


def set_lifecycle_policy(s3, bucket: str, expire_in_days: int):
    s3.put_bucket_lifecycle(Bucket=bucket, LifecycleConfiguration={
        'Rules': [
            {
                'ID': 'expiry-rule',
                'Expiration': {
                    'Days': expire_in_days,
                },
                'Prefix': '',
                'Status': 'Enabled',
            },
        ]
    })
