import keyboard as ky
import win32api
from pynput.mouse import Button, Controller
import screen_capture

process_name = "RotMGExalt"


class input_logger:
    def __init__(self):
        self.mouse = Controller()

    def get_keyboard_state(self):

        key_down = []

        for key in ['q', 'e', 'w', 'a','s','d','f','v','r',' ']:
            if ky.is_pressed(key):
                key_down.append(1)
            else:
                key_down.append(0)

        return key_down

    def get_mouse_state(self):

        mouse_pos = self.mouse.position
        lmb_state = 0
        state_left = win32api.GetKeyState(0x01)
        state_right = win32api.GetKeyState(0x02)

        top_left_coordinate, bot_right_coordinate = screen_capture.get_cords(
            process_name
        )

        width = bot_right_coordinate[0] - top_left_coordinate[0]
        height = bot_right_coordinate[1] - top_left_coordinate[1]

        # x pos
        mouse_pos_1 = (mouse_pos[0] - top_left_coordinate[0]) / width

        # y pos
        mouse_pos_2 = (mouse_pos[1] - top_left_coordinate[1]) / height

        if mouse_pos_1 > bot_right_coordinate[0] - top_left_coordinate[0]:
            mouse_pos_1 = bot_right_coordinate[0] - top_left_coordinate[0]
        elif mouse_pos_1 < 0:
            mouse_pos_1 = 0

        if mouse_pos_2 > bot_right_coordinate[1] - top_left_coordinate[1]:
            mouse_pos_2 = bot_right_coordinate[1] - top_left_coordinate[1]
        elif mouse_pos_2 < 0:
            mouse_pos_2 = 0

        print(str((mouse_pos_1, mouse_pos_2)))
        if state_left == -128 or state_left == -127:
            lmb_state = 1
        else:
            lmb_state = 0

        if state_right == -128 or state_right == -127:
            rmb_state = 1
        else:
            rmb_state = 0

        return (mouse_pos_1, mouse_pos_2, lmb_state, rmb_state)
