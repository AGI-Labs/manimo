import time
from enum import Enum

HOMES = {
    "pour": [0.06174963, -0.23580325, 0.01359189, -2.61944, 1.5873924, 1.6115918, -1.5642371],
    # "pour": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
    "scoop": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
    "zip": [-0.1337, 0.3634, -0.1395, -2.3153, 0.1478, 2.7733, -1.1784],
    "insertion": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
}

class Rate:
    """
    Maintains constant control rate for the environment loop
    """

    def __init__(self, frequency):
        self._period = 1.0 / frequency
        self._last = time.time()

    def sleep(self):
        current_delta = time.time() - self._last
        sleep_time = max(0, self._period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self._last = time.time()

class ButtonState(Enum):
    OFF = 0,
    INPROGRESS = 1,
    ON = 2

class StateManager:
    def __init__(self):
        """
        Initialize the StateManager with an empty dictionary to store button states.
        """
        self.button_states = {}

    def handle_state(self, buttons, button_key, function):
        """
        Handles the state of a button and calls the provided function when the button state changes.

        Args:
            buttons (dict): A dictionary containing the current states of all buttons.
            button_key (str): The key representing the button in the 'buttons' dictionary.
            function (callable): The function to be called when the button state changes.
        """
        # Initialize the button state if not present in the button_states dictionary
        if button_key not in self.button_states:
            self.button_states[button_key] = ButtonState.OFF

        # Check the current state of the button
        button_toggle = buttons[button_key]

        # If the button is pressed, set the state to INPROGRESS
        if button_toggle:
            self.button_states[button_key] = ButtonState.INPROGRESS

        # If the button is released and the previous state was INPROGRESS, call the provided function
        # and set the state back to OFF
        elif not button_toggle and self.button_states[button_key] == ButtonState.INPROGRESS:
            function()
            self.button_states[button_key] = ButtonState.OFF