import screen_capture
from input_logger import input_logger as il
import csv
import time
import os
import shutil

import keyboard as ky

import subprocess

import cv2

filename = "output.mp4"

mp4 = f"generate_mp4.py -e jpg -fps 30 -d images"

application_name = "Galcon 2"


def main():

    CWD = os.getcwd()

    image_dir = os.path.join(CWD, "images")

    logger = il()

    with open("gameplay.csv", mode="w", newline="") as gameplay:
        gameplay_writer = csv.writer(gameplay)
        i = 0

        keyboard_state = []

        looper = 0

        while not ky.is_pressed("l"):
            time.sleep(1 / 30)
            image_name = f"{i}.jpg"

            looper += 1

            try:

                mouse_state = logger.get_mouse_state()

                keyboard_state = logger.get_keyboard_state()
                screen_shot = screen_capture.capture(application_name)
                # if 1 in keyboard_state or 1 in mouse_state: Noita
                cv2.imwrite(
                    os.path.join(image_dir, image_name),
                    screen_shot,
                    [cv2.IMWRITE_JPEG_QUALITY, 40],
                )
                gameplay_writer.writerow(
                    [
                        mouse_state[0],
                        mouse_state[1],
                        mouse_state[2]
                        # *keyboard_state,
                    ]
                )
                i += 1

            except KeyboardInterrupt:
                break

        print("Stopping Recording")


def cleanup():
    CWD = os.getcwd()

    image_dir = os.path.join(CWD, "images")

    output = str(time.time())

    for image in os.listdir(image_dir):
        os.remove(os.path.join(image_dir, image))

    os.mkdir(os.path.join(CWD, output))

    shutil.move(f"{str(filename)}", output)
    shutil.move("gameplay.csv", output)

    shutil.move(output, "recorded data")


if __name__ == "__main__":
    main()
    subprocess.Popen(mp4, shell=True).wait()
    cleanup()
