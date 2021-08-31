import pyautogui
import numpy as np
from PIL import Image
import cv2

import pygetwindow as gw

import mss
import mss.tools

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Pass in coordinate pairs as tuples
def capture(window_name, context):

    top_left_coordinate, bot_right_coordinate = get_cords(window_name)

    # The screen part to capture
    monitor = {
        "top": top_left_coordinate[1],
        "left": top_left_coordinate[0],
        "width": bot_right_coordinate[0] - top_left_coordinate[0],
        "height": bot_right_coordinate[1] - top_left_coordinate[1],
    }

    # Grab the data
    return context.grab(monitor)


def get_cords(window_name):
    window = gw.getWindowsWithTitle(window_name)[0]

    top_left_coordinate = window.topleft
    bot_right_coordinate = window.bottomright
    return top_left_coordinate, bot_right_coordinate
