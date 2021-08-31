import os

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.models import Sequential, model_from_json
from keras.layers import (
    Dense,
    Reshape,
    Activation,
    LSTM,
    Flatten,
    Dropout,
    MaxPooling2D,
    Conv2D,
)
from keras.layers import LeakyReLU, SimpleRNN
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
import tensorflow as tf
from pympler import asizeof

import cv2
import json
import keras.backend as K
from numpy import loadtxt
import numpy as np
import time
from keras.callbacks import LambdaCallback

os.environ["SM_FRAMEWORK"] = "tf.keras"

# Change image dimension here.
imgX = 90
imgy = 90


class neural_network:
    def __init__(self, data_dir="recorded data", model_dir="model"):
        self.cwd = os.getcwd()
        self.data_dir = os.path.join(self.cwd, data_dir)
        self.model_dir = os.path.join(self.cwd, model_dir)

    def unpack_mp4(self, video_file):
        vidObj = cv2.VideoCapture(video_file)
        success = 1
        frames = []

        while success:
            success, image = vidObj.read()
            try:
                h, w, c = image.shape
            except Exception as e:
                print(e)
            try:
                image = cv2.resize(image, (imgX, imgy), interpolation=cv2.INTER_AREA)
                frames.append(image)
            except Exception as e:
                print(e)

        return (frames, h, w)

    #
    def customLoss(yTrue, yPred):
        return

    def optimizer(self):
        return SGD(lr=0.01, momentum=0.9)

    def create_model(self, model="model.json", model_weights="model.h5"):
        self.model_json = os.path.join(self.model_dir, model)
        self.model_weights = os.path.join(self.model_dir, model_weights)

        model = Sequential()
        model.add(
            Conv2D(
                24,
                (2, 2),
                data_format="channels_last",
                padding="valid",
                activation="relu",
                input_shape=(imgX, imgy, 3),
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add((Flatten()))
        model.add(Dense(200, activation="relu", use_bias=True))
        model.add(Dense(128, activation="relu", use_bias=True))
        model.add(Dense(128, activation="relu", use_bias=True))
        model.add(Dense(13, activation="sigmoid"))

        model.compile(
            loss="mean_squared_error",
            optimizer=self.optimizer(),
            metrics=["accuracy"],
        )
        self.model = model

        with open(self.model_json, "w") as json_file:
            json_file.write(self.model.to_json())

    def load_model(self, model="model.json", model_weights="model.h5"):
        self.model_json = os.path.join(self.model_dir, model)
        self.model_weights = os.path.join(self.model_dir, model_weights)

        if len(os.listdir(self.model_dir)) < 1:
            return False

        with open(self.model_json, "r") as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)

        # load weights into new model
        self.model.load_weights(self.model_weights)
        print("Loaded model from disk")
        self.model.compile(
            loss="mean_squared_error",
            optimizer=self.optimizer(),
            metrics=["accuracy"],
        )
        return True


    def train(self, epochs=50):
        if self.model is None:
            print("Model is not loaded, aborting training")
            return

        print(
            "Num GPUs Available: ", len(tf.config.list_physical_devices("GPU"))
        )

        frames = np.array(self.frames)

        reshaped_frames = [np.reshape(frame, (imgX, imgy, 3)) for frame in frames]

        numpy_reshaped_frames = np.array(reshaped_frames)

        self.model.summary()

        self.model.fit(numpy_reshaped_frames, self.inputs, epochs=epochs, batch_size=16)

        print("PREDICTION")

        for frame in reshaped_frames[:25]:
            test_ar = [frame]
            test_aaaa = np.array(test_ar)

            print(self.model.predict(test_aaaa))

        self.model.save_weights(self.model_weights)

    def load_data(self, path):
        if "dataset.json" not in os.listdir(os.path.join(self.data_dir, path)):
            print("Error: no dataset.json: {}".format(path))
            return
        with open(os.path.join(self.data_dir, path, "dataset.json"), 'r') as f:
            dataset_info = json.load(f)
        if (dataset_info["version"] != "1.0.0"):
            print("Error: unknown dataset.json version: {}".format(dataset_info["version"]))
            return

        self.frames = []
        self.inputs = []

        frames, h, w = self.unpack_mp4(
            os.path.abspath(os.path.join(self.data_dir, path, dataset_info["inputs"]))
        )
        inputs = loadtxt(
            os.path.abspath(os.path.join(self.data_dir, path, dataset_info["outputs"])),
            delimiter=",",
        )


        mouse_inputs = []

        normalized_frame = []
        normalized_frames = []

        for index, frame in enumerate(frames):
            for index, pixel in enumerate(frame):
                normalized_frame.append(pixel / 255)
            normalized_frames.append(normalized_frame)
            normalized_frame = []

        print(str(folder))
        print(f"{asizeof.asizeof(normalized_frames) / 1024}")

        self.frames = normalized_frames
        self.inputs = inputs


if __name__ == "__main__":
    the_bat = neural_network()
    if not the_bat.load_model():
        raise SystemExit("Couldn't load model")
    while True:
        for folder in os.listdir(the_bat.data_dir):
            if folder.startswith('.') or (not os.path.isdir(os.path.join(the_bat.data_dir, folder))):
                continue
            the_bat.load_data(folder)
            the_bat.train()
