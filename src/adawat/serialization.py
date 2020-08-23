# %%

import math
import hashlib
import pickle
import os
import tempfile
import logging
from typing import Any, List


logger = logging.getLogger(__name__)


def object_sig(obj, *args, **kwargs) -> str:
    """
    Generates a string that uniquely identifies an instances of a class
    based on its initialization arguments. For example, given an instance
    of class TestClass constructed as TestClass(1, 2, 3, arg1='val1'),
    this method returns a unique signature based on the class name and
    the passed arguments. The format of the output is easiest discovered
    by a test call.
    """
    cls_name = obj.__class__.__name__
    args_str = ';;'.join(map(str, args)) or "<no_args>"
    kwargs_str = ';;'.join(['%s=%s' % (k, v)
                            for k, v in kwargs.items()]) or "<no_kwargs>"
    sig = ';;;'.join((cls_name, args_str, kwargs_str))
    return sig


def object_filepath(obj, *args, **kwargs) -> str:
    """
    Based on the id of an instance of an object with a certain initialization
    arguments, this method generates a unique path under the system's temporary
    directory which can be used for saving to and reading from the state of the
    object, respectively.

    Arguments:
    obj -- the object
    *args -- the positional arguments passed to __init__() during construction
    *kwargs -- the keyword arguments passed to __init__() during construction

    Returns:
    A unique path for storing to and reading from the object's state.
    """
    sig = object_sig(obj, *args, **kwargs)
    hash = hashlib.sha256(sig.encode('utf-8')).hexdigest()
    filename = f'f{hash}.state'
    return os.path.join(tempfile.gettempdir(), filename)


def get_state(obj, attrs: List[str]) -> List[Any]:
    """
    Given a list of attribute names, this method extracts the values of those
    attributes and return them in a list.

    Arguments:
    obj -- the object
    attrs -- a list containing the names of the attributes to be retrieved.

    Returns:
    A list containing the values of the requested attributes.
    """
    return [getattr(obj, attr) for attr in attrs]


def update_state(obj, attrs: List[str], state: List[Any]):
    """
    Given a list of attribute names and corresponding values, this method
    updates those attributes with the given values.

    Arguments:
    obj -- the object
    attrs -- a list containing the names of the attributes to be updated.
    state -- a list containing the updated new values.
    """
    for i, attr in enumerate(attrs):
        setattr(obj, attr, state[i])


def save_object(obj, attrs: List[str], filepath: str):
    """
    Saves the state of an object to a file. The file path is uniquely generated
    based on the initialization arguments (*args and **kwargs). The state of the
    object is the values of the attributes whose names are given.

    obj -- the object
    attrs -- a list containing the names of the attributes to be saved.
    filepath -- a path to a file to save the object attributes to.
    """

    if obj is None:
        raise ValueError('Object cannot be None.')
    if attrs is None or len(attrs) == 0:
        raise ValueError(
            'Invalid attrs. Expecting a list of at least one item.')

    # Retrieve the state of the object.
    state = get_state(obj, attrs)

    # Save the state of the object to a file.
    with open(filepath, 'wb') as file:
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(obj, attrs: List[str], filepath: str):
    """
    Loads the state of an object from a file. The file path is uniquely generated
    based on the initilization arguments (*args and **kwargs). The file should
    have been previously saved via save_object() method with the same list of
    attributes.

    obj -- the object
    attrs -- a list containing the names of the attributes to be saved.
    filepath -- a path to a file to load the object attributes from.

    Returns:
    True -- if the object was successfully loaded from a file. 
    False -- if a file for the given object, args, and keyword arguments was not
      found or its content doesn't match the expected state.
    """

    if obj is None:
        raise ValueError('Object cannot be None.')
    if attrs is None or len(attrs) == 0:
        raise ValueError(
            'Invalid attrs. Expecting a list of at least one item.')

    if not os.path.isfile(filepath):
        return False

    # Retrieve the state of the object from the file.
    with open(filepath, 'rb') as file:
        state = pickle.load(file)
    if not isinstance(state, list) or len(state) != len(attrs):
        # Expecting a list; ignoring this file.
        return False

    # Update the state of the object based on the file content.
    update_state(obj, attrs, state)

    return True


def stateful(attrs: List[str] = ["state_dict"], save_state_method_name='save_state'):
    """
    A decorator that can be applied to a class to make it save its status to the
    disk and automatically used the saved status next time the object is
    initialized with the same parameters. This is useful for objects that takes
    a long time to instantiate and rarely change.

    attrs -- The list of attributes to be saved. This is useful since it may be
             desired to only save some of the attributes. By default, this is set
             to ["state_dict"], i.e. an attribute called "state_dict" is expected
             to exists containing the state of the object.
    """

    # TODO Consider automatically using all attributes if none is provided.
    if not attrs:
        raise ValueError("""
You should specify a list of the atributes that will be used for saving and
loading the state of the object. For example:

@stateful(['attr1', 'attr2'])
class StatefulObject():
    ...
""".strip())

    def stateful_decorator(cls):
        if type(cls) is not type:
            raise TypeError(
                "The @stateful decorator can only be applied to classes.")
        init = cls.__init__

        if hasattr(cls, save_state_method_name):
            raise RuntimeError(
                f"Cannot add a method called '{save_state_method_name}' to " +
                "the class as it already has a method with such name.")

        def new_init(self, *args, **kwargs):
            self._state_filepath = object_filepath(self, *args, **kwargs)
            if 'force_init' in kwargs and kwargs['force_init'] == True:
                del kwargs['force_init']
                init(self, *args, **kwargs)
                save_object(self, attrs, self._state_filepath)
            elif not load_object(self, attrs, self._state_filepath):
                init(self, *args, **kwargs)
                save_object(self, attrs, self._state_filepath)

        def save_state(self):
            save_object(self, attrs, self._state_filepath)

        cls.__init__ = new_init
        setattr(cls, save_state_method_name, save_state)

        return cls

    return stateful_decorator


__all__ = [
    "stateful"
]
