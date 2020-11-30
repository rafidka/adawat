from datetime import datetime

import pytest

from adawat.picklers import FilePickler, MemoryPickler
from adawat.serialization import (get_obj_attrs, object_id, object_sig,
                                  pickle_obj_attrs, set_obj_attrs, stateful,
                                  unpickle_obj_attrs)
from adawat.state_machine import PersistentStateMachine


class Test_StateMachine:
    def test_simple_state_machine(self):
        states = []

        class TestMachine(PersistentStateMachine):
            def __init__(self, id: str):
                # state1 is the starting state
                super().__init__(id, "state1", pickler=MemoryPickler())

            def state1(self):
                states.append("state1")
                return "state2", {}

            def state2(self):
                states.append("state2")
                return "state3", {}

            def state3(self):
                states.append("state3")
                return None, {}

        test_machine = TestMachine("test_id")
        while test_machine.run_next():
            pass

        assert states == ["state1", "state2", "state3"]

    def test_slighty_complex_state_machine(self):
        states = []

        class TestMachine(PersistentStateMachine):
            def __init__(self, id: str):
                super().__init__(id, "state1")  # state1 is the starting state

            def state1(self):
                states.append("state1")
                return "state2", {}

            def state2(self):
                states.append("state2")
                counter = self.state.data['counter'] if 'counter' in self.state.data else 0
                if counter == 3:
                    return "state3", {}
                else:
                    return "state2", {'counter': counter + 1}

            def state3(self):
                states.append("state3")
                return None, {}

        test_machine = TestMachine("test_id")
        while test_machine.run_next():
            pass

        assert states == ["state1",
                          "state2", "state2", "state2", "state2",
                          "state3"]
