import pyautogui 
import numpy as np
from PIL import Image
import cv2

import pygetwindow as gw

import mss
import mss.tools

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

offsetX = 0
offsetY = 0
# Pass in coordinate pairs as tuples
def capture(window_name):
    
    top_left_coordinate,bot_right_coordinate = get_cords(window_name)
    
    with mss.mss() as sct:
        # The screen part to capture
        monitor = {
        	"top": top_left_coordinate[1], 
        	"left": top_left_coordinate[0], 
        	"width": bot_right_coordinate[0] - top_left_coordinate[0], 
        	"height": bot_right_coordinate[1] - top_left_coordinate[1]
        }
        output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)

        # Grab the data
        img = np.array(sct.grab(monitor))
    return img 

def get_cords(window_name):
    window = gw.getWindowsWithTitle(window_name)[0]

    top_left_coordinate = window.topleft
    bot_right_coordinate = window.bottomright
    return top_left_coordinate, bot_right_coordinate

# def main():
#     window_name = "Adobe Flash Player 18"
#     image = capture(window_name)


# if __name__ == '__main__':
#     main()

