#!/usr/bin/python3

class State(object):
    def __init__(self, name: str):
        self.name = name

    def tick(self):
        pass

    def on_transition(self):
        return True


class StateMachine(object):
    def __init__(self, name: str):
        self._name = name
        self._current_state = None
        self._state_dict = {}

    def add_state(self, state: State):
        if state.name not in self._state_dict:
            self._state_dict[state.name] = state

    def move_to_state(self, name: str):
        pass

    def run(self):
        state: State
        for state in self._state_dict:
            if state.on_transition():
                self._current_state = state
                break
        self._current_state = state
        self._current_state.tick()

class CowBehavior(StateMachine):
    class Grazing(State):
        def __init__(self):
            super().__init__("grazing")

        def on_transition(self):
            return super().on_transition()
