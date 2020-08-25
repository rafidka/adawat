from abc import ABC, abstractmethod
import os
import pickle
import tempfile


class ObjectNotFoundError(RuntimeError):
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

    def dump(self, obj_id, obj):
        filepath = self._filepath(obj_id)
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, obj_id):
        filepath = self._filepath(obj_id)

        if not os.path.isfile(filepath):
            raise ObjectNotFoundError(
                f"Couldn't find the file {filepath} while trying to load object with ID {obj_id}.")

        with open(filepath, 'rb') as file:
            return pickle.load(file)
