# %%

import hashlib
import os
import logging
from typing import Any, List
from abc import ABC

from adawat.picklers import Pickler, FilePickler, ObjectNotFoundError


log = logging.getLogger(__name__)


def object_sig(obj, *args, **kwargs) -> str:
    """
    Generates a string that tries to uniquely identifies an instances of a class
    based on its initialization arguments. For example, given an instance of
    class TestClass constructed as TestClass(1, 2, 3, arg1='val1'), this method
    returns a unique signature based on the class name and the passed arguments.
    The format of the output is easiest discovered by a test call.
    """
    cls_name = obj.__class__.__name__
    # Use 'str' instead of 'repr'. This is not ideal, since multiple different
    # instances can have the same __str__ (which could also be the case for
    # __repr__ but definitely the latter is way less likely), but using __repr__
    #  can generate huge strings e.g. passing in a huge list.
    args_str = ';;'.join(map(str, args)) or "<no_args>"
    kwargs_str = ';;'.join(['%s=%s' % (str(k), str(v))
                            for k, v in kwargs.items()]) or "<no_kwargs>"
    sig = ';;;'.join((cls_name, args_str, kwargs_str))
    return sig


def object_id(obj, *args, **kwargs) -> str:
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
    return hash  # use the hash as as an ID


def get_obj_attrs(obj, attrs: List[str]) -> List[Any]:
    """
    Given a list of attribute names, this method extracts the values of those
    attributes and return them in a list.

    Arguments:
    obj -- the object.
    attrs -- a list containing the names of the attributes to retrieve.

    Returns:
    A list containing the values of the requested attributes.
    """
    return [getattr(obj, attr) for attr in attrs]


def set_obj_attrs(obj, attrs: List[str], values: List[Any]):
    """
    Given a list of attribute names and corresponding values, this method
    updates those attributes with the given values.

    Arguments:
    obj -- the object
    attrs -- a list containing the names of the attributes to be updated.
    values -- a list containing the new attribute values.
    """
    for i, attr in enumerate(attrs):
        setattr(obj, attr, values[i])


def pickle_obj_attrs(obj, obj_id: str, attrs: List[str], pickler: Pickler):
    """
    Saves the attributes of an object using a pickler.

    obj -- the object.
    obj_id -- a string that uniquely identify the object.
    attrs -- a list containing the names of the attributes to be saved.
    pickler -- the pickler to use.
    """

    if obj is None:
        raise ValueError('Object cannot be None.')
    if attrs is None or len(attrs) == 0:
        raise ValueError(
            'Invalid attrs. Expecting a list of at least one item.')

    # Retrieve the values of the attributes.
    values = get_obj_attrs(obj, attrs)

    # Save the values using the pickler.
    pickler.dump(obj_id, values)


def unpickle_obj_attrs(obj, obj_id: str, attrs: List[str], pickler: Pickler):
    """
    Loads the attributes of an object from a pickler.

    obj -- the object.
    obj_id -- a string that uniquely identify the object.
    attrs -- a list containing the names of the attributes to be loaded.
    pickler -- the pickler to load the attributes from.

    Returns:
    True -- if the attributes were successfully loaded from pickler.
    False -- if no attributes were found for the given object ID, or if the
      attributes found were more or less than the expected number of attributes.
    """

    if obj is None:
        raise ValueError('Object cannot be None.')
    if attrs is None or len(attrs) == 0:
        raise ValueError(
            'Invalid attrs. Expecting a list of at least one item.')

    # Retrieve the values of the attributes from the pickler.
    try:
        state = pickler.load(obj_id)
    except ObjectNotFoundError:
        return False

    if not isinstance(state, list) or len(state) != len(attrs):
        # Expecting a list; ignoring.
        return False

    # Update the state of the object based on the loaded content.
    set_obj_attrs(obj, attrs, state)

    return True


def stateful(attrs: List[str] = ["state_dict"],
             save_state_method_name: str = None,
             delete_state_method_name: str = None,
             pickler: Pickler = FilePickler()):
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

        # Override the constructor of the class with one that checks for a
        # state and calls the original constructor if there is none.
        def new_init(self, *args, **kwargs):
            self._obj_id = object_id(self, *args, **kwargs)
            if 'force_init' in kwargs and kwargs['force_init'] == True:
                del kwargs['force_init']
                init(self, *args, **kwargs)
                pickle_obj_attrs(self, self._obj_id, attrs, pickler)
            elif not unpickle_obj_attrs(self, self._obj_id, attrs, pickler):
                init(self, *args, **kwargs)
                pickle_obj_attrs(self, self._obj_id, attrs, pickler)

        cls.__init__ = new_init

        # Optionally, define a save_state method for the class.
        if save_state_method_name is not None:
            if hasattr(cls, save_state_method_name):
                raise RuntimeError(
                    f"Cannot add a method called '{save_state_method_name}' to " +
                    "the class as it already has a method with such name.")

            def save_state(self):
                pickle_obj_attrs(self, self._obj_id, attrs, pickler)

            setattr(cls, save_state_method_name, save_state)

        # Optionally, define a delete_state method for the class.
        if delete_state_method_name is not None:
            if hasattr(cls, delete_state_method_name):
                raise RuntimeError(
                    f"Cannot add a method called '{delete_state_method_name}' to " +
                    "the class as it already has a method with such name.")

            def delete_state(self):
                pickler.delete(self._obj_id)

            setattr(cls, delete_state_method_name, delete_state)

        return cls

    return stateful_decorator


__all__ = [
    "stateful"
]


@stateful(attrs=['next_state', 'data'],
          save_state_method_name="save",
          delete_state_method_name="delete")
class PersistentState:
    def __init__(self, id: str):
        self.id = id
        self.next_state = None
        self.data = {}


class StateMachine(ABC):
    def __init__(self, id: str, start_state: str):
        self.id = id
        self.start_state = start_state

        # Construct a persistent state for the state machine. This class is
        # decorated with the @stateful decorator, making it automatically
        # restore the last state.
        self.state = PersistentState(id)

    def get_state_func(self, func_name):
        """
        Retrieves a function in this class with the given name.

        Keyword arguments:
        func_name -- The name of the function to retrieve.

        Returns:
        The function.
        """
        func = getattr(self, func_name)
        if func is None:
            raise RuntimeError(
                f"Couldn't retrieve a function with the name {func_name}.")
        if not callable(func):
            raise RuntimeError(
                f"Found an attribute with the name {func_name} but it is not callable.")
        return func

    def run_next(self):
        next_state = self.state.next_state
        if next_state is None:
            next_state = self.start_state
        try:
            func = self.get_state_func(next_state)
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve the function for running the next " +
                f"step '{next_state}' of a state with Id {self.id}.") from e

        ret = func()

        if (not isinstance(ret, list) and not isinstance(ret, tuple)) or \
                len(ret) != 2:
            raise RuntimeError(
                f"Failed while executing the state '{next_state}''. The " +
                "function should return a tuple/list of 2 elements, the " +
                "first specifying the next state and the second specifying " +
                "the updated state data.")

        self.state.next_state, self.state.data = ret

        if self.state.next_state is None:
            # We are done. Delete the state.
            self.state.delete()
            return False
        else:
            # Still more states to go. Save the state.
            self.state.save()
            return True
