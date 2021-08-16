import numpy as np
import os

import time
import cv2

import screen_capture
from tensorflow import keras
from tensorflow.keras.optimizers import SGD

from pynput.mouse import Button, Controller
import keyboard as ky
from matplotlib import pyplot as plt

# Change image dimension here.
imgX = 90
imgy = 90
process_name = "noita"


def predict(model, X):
    return model.predict(X)


def load_model():
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, "model")

    with open(os.path.join(model_dir, "model.json"), "r") as json_file:
        loaded_model_json = json_file.read()
    model = keras.models.model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(os.path.join(model_dir, "model.h5"))
    print("Loaded model from disk")
    model.compile(
        loss="mean_squared_error",
        optimizer=SGD(lr=0.01, momentum=0.9),
        metrics=["accuracy"],
    )

    return model


def capture(game_name):

    img = screen_capture.capture(game_name)
    topleft, botright = screen_capture.get_cords(game_name)
    image = cv2.resize(
        img,
        (
            imgX,
            imgy,
        ),
        interpolation=cv2.INTER_AREA,
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height = img.shape[0]
    width = img.shape[1]

    normalized_frame = []
    print(image[0][0])
    for index, pixel in enumerate(image):
        for rgb_index, rgb_value in enumerate(pixel):
            normalized_frame.append([pixel[rgb_index] / 255])

    reshaped = np.reshape(
        normalized_frame,
        (imgX, imgy, 3),
    )

    numpy_frames = np.array([reshaped])

    return (numpy_frames, height, width, topleft)


def main():

    model = load_model()

    mouse = Controller()

    while True:
        ky.press
        time.sleep(1 / 10)
        img, h, w, topleft = capture(process_name)

        prediction = predict(model, img)
        ky.release("d")
        ky.release("a")
        ky.release("w")
        ky.release("s")
        ky.release("f")
        if ky.is_pressed("p"):
            break

        x = prediction[0][0] * w + topleft[0]
        y = prediction[0][1] * h + topleft[1]

        mouse.position = (x, y)

        # Left Click
        if prediction[0][2] > 0.5:
            mouse.press(Button.left)
        if prediction[0][2] < 0.5:
            mouse.release(Button.left)
        if prediction[0][3] > 0.5:
            ky.press("w")

        if prediction[0][4] > 0.5:
            ky.press("a")

        if prediction[0][5] > 0.5:
            ky.press("s")

        if prediction[0][6] > 0.5:
            ky.press("d")

        if prediction[0][7] > 0.3:
            ky.press("f")


if __name__ == "__main__":
    main()
