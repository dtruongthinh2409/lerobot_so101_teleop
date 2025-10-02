import carb
import omni.appwindow


class KeyboardControl:
    def __init__(self):
        self.reset_world = False

        # Get the window to register keyboard callbacks
        self._window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._window.get_keyboard()

        # Register keyboard callbacks
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Keyboard event handler"""
        # Only process key press events
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_world = True
                print(f"[INFO]: Reset world...")
                return True

        return False

    def cleanup(self):
        """Cleanup the keyboard interface"""
        if self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._sub_keyboard
            )
            self._sub_keyboard = None
