import hashlib
import logging
import os
from abc import ABC
from typing import Any, List

from adawat.picklers import MemoryPickler, ObjectNotFoundError, Pickler
from adawat.serialization import stateful


class PersistentStateMachine(ABC):
    def __init__(self, id: str, start_state: str, pickler=MemoryPickler()):
        self.id = id
        self.start_state = start_state

        @stateful(attrs=['next_state', 'data'],
                  save_state_method_name="save",
                  delete_state_method_name="delete",
                  pickler=pickler)
        class PersistentState:
            def __init__(self, id: str):
                self.id = id
                self.next_state = None
                self.data = {}

        # These will be defined by the @stateful decorator, but using this
        # trick so the IDE doesn't complain.
        PersistentState.save = PersistentState.save
        PersistentState.delete = PersistentState.delete

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
            # We are done. Delete the state so another instantiation of the
            # machine starts from the beginning.
            self.state.delete()
            return False
        else:
            # Still more states to go. Save the state so if this instance of
            # the state function die before completion, another instantiation
            # continue from the last state.
            self.state.save()
            return True
