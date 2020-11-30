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

    @abstractmethod
    def delete(self, obj_id: str):
        pass


class MemoryPickler(Pickler):
    def __init__(self):
        self.storage = {}

    def dump(self, obj_id: str, obj):
        obj_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        log.debug(f"Saving object with id {obj_id} to memory.")
        self.storage[obj_id] = obj_bytes
        log.debug(f"Saved object with id {obj_id} to memory.")

    def load(self, obj_id: str):
        if not obj_id in self.storage:
            raise ObjectNotFoundError(
                f"Couldn't find object with id {obj_id} in memory.")
        try:
            log.debug(f"Loading object with id {obj_id} from memory.")
            obj_bytes = self.storage[obj_id]
            obj = pickle.loads(obj_bytes)
            log.debug(f"Loaded object with id {obj_id} from memory.")
            return obj
        except Exception as e:
            log.exception(e)
            raise ObjectNotFoundError() from e

    def delete(self, obj_id: str):
        if obj_id in self.storage:
            log.debug(f"Deleting object with id {obj_id} from memory.")
            del self.storage[obj_id]
            log.debug(f"Deleted object with id {obj_id} from memory.")


class FilePickler(Pickler):
    def _filepath(self, key):
        filename = f'f{key}.state'
        filepath = os.path.join(tempfile.gettempdir(), filename)
        return filepath

    def dump(self, obj_id: str, obj):
        filepath = self._filepath(obj_id)
        log.debug(f"Saving object with id {obj_id} to {filepath}.")
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        log.debug(f"Saved object with id {obj_id} to {filepath}.")

    def load(self, obj_id: str):
        filepath = self._filepath(obj_id)

        if not os.path.isfile(filepath):
            raise ObjectNotFoundError(
                f"Couldn't find the file {filepath} while trying to load object with ID {obj_id}.")

        with open(filepath, 'rb') as file:
            log.debug(f"Loading object with id {obj_id} from {filepath}.")
            obj = pickle.load(file)
            log.debug(f"Loaded object with id {obj_id} from {filepath}.")
            return obj

    def delete(self, obj_id: str):
        filepath = self._filepath(obj_id)
        if os.path.exists(filepath):
            log.debug(
                f"Deleting object with id {obj_id}. Deleting the file {filepath}.")
            os.remove(filepath)
            log.debug(
                f"Loading object with id {obj_id}. Deleted the file {filepath}.")


class S3Pickler(Pickler):
    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket
        self.ready = False

        self._create_bucket()
        # delete after a week to reduce costs.
        self._set_lifecycle_policy(expire_in_days=7)

    def _bucket_exists(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except self.client.exceptions.ClientError:
            # The bucket does not exist or you have no access.
            return False

    def _object_exists(self, obj_id: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=obj_id)
            return True
        except self.client.exceptions.ClientError:
            # The bucket does not exist or you have no access.
            return False

    def _create_bucket(self):
        if not self._bucket_exists():
            try:
                log.info(f"Bucket {self.bucket} doesn't exist. Creating it...")
                self.client.create_bucket(Bucket=self.bucket)
                self.ready = True
                log.info(f"Bucket {self.bucket} created.")
            except Exception as e:
                log.exception(e)
                # The bucket doesn't exist and we couldn't create it. Mark the
                # pickler as not ready for use.
                self.ready = False
        else:
            log.debug(
                f"Bucket {self.bucket} already exists. No need to create it.")

    def _set_lifecycle_policy(self, expire_in_days: int):
        try:
            log.info(f'Set lifecycle policy for bucket {self.bucket} ' +
                     f'with expiration of {expire_in_days} days')
            self.client.put_bucket_lifecycle(Bucket=self.bucket, LifecycleConfiguration={
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
        except Exception as e:
            log.fatal(f"Failed to set life")
            log.exception(e)

    def dump(self, obj_id: str, obj):
        obj_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            log.debug(
                f"Saving object with id {obj_id} to s3://{self.bucket}/{obj_id} .")

            self.client.put_object(
                Body=obj_bytes, Bucket=self.bucket, Key=obj_id)

            log.debug(
                f"Saved object with id {obj_id} to s3://{self.bucket}/{obj_id} .")
        except Exception as e:
            # dump() methods silently fail so it doesn't impact execution, so
            # we just log the exception for visibility.
            log.exception(e)

    def load(self, obj_id: str):
        try:
            log.debug(
                f"Loading object with id {obj_id} from s3://{self.bucket}/{obj_id} .")

            response = self.client.get_object(Bucket=self.bucket, Key=obj_id)
            obj_bytes = response['Body'].read()
            obj = pickle.loads(obj_bytes)

            log.debug(
                f"Loaded object with id {obj_id} from s3://{self.bucket}/{obj_id} .")

            return obj
        except Exception as e:
            log.exception(e)
            raise ObjectNotFoundError() from e

    def delete(self, obj_id: str):
        if self._object_exists(obj_id):
            log.debug(
                f"Deleting object with id {obj_id}. Removing S3 object s3://{self.bucket}/{obj_id} .")
            self.client.delete_object(Bucket=self.bucket, Key=obj_id)
            log.debug(
                f"Deleted object with id {obj_id}. Removed S3 object s3://{self.bucket}/{obj_id} .")
