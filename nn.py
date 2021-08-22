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
    def __init__(self):
        self.cwd = os.getcwd()
        self.data_dir = os.path.join(self.cwd, "recorded data")
        self.model_dir = os.path.join(self.cwd, "model")

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

    def create_model(self):

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
        model.add(Dense(8, activation="sigmoid"))

        model.compile(
            loss="mean_squared_error",
            optimizer=self.optimizer(),
            metrics=["accuracy"],
        )
        return model

    def train(self, frames, inputs):

        if len(os.listdir(self.model_dir)) < 1:
            model = self.create_model()
        else:
            with open(os.path.join(self.model_dir, "model.json"), "r") as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)

            # load weights into new model
            model.load_weights(os.path.join(self.model_dir, "model.h5"))
            print("Loaded model from disk")
            model.compile(
                loss="mean_squared_error",
                optimizer=self.optimizer(),
                metrics=["accuracy"],
            )

        frames = np.array(frames)

        reshaped_frames = [np.reshape(frame, (imgX, imgy, 3)) for frame in frames]

        numpy_reshaped_frames = np.array(reshaped_frames)

        model.summary()

        model.fit(numpy_reshaped_frames, inputs, epochs=150, batch_size=16)

        print("PREDICTION")

        for frame in reshaped_frames[:25]:
            test_ar = [frame]
            test_aaaa = np.array(test_ar)

            print(model.predict(test_aaaa))

        model_json = model.to_json()

        with open(os.path.join(self.model_dir, "model.json"), "w") as json_file:
            json_file.write(model_json)

        model.save_weights(os.path.join(self.model_dir, "model.h5"))

    def load_data(self):
        for i in range(100):
            for folder in os.listdir(self.data_dir):
                if folder.startswith('.') or (not os.path.isdir(os.path.join(self.data_dir, folder))):
                    continue
                for data in os.listdir(os.path.join(self.data_dir, folder)):
                    if data.endswith(".mp4"):
                        frames, h, w = self.unpack_mp4(
                            os.path.abspath(os.path.join(self.data_dir, folder, data))
                        )

                    if data.endswith(".csv"):
                        inputs = loadtxt(
                            os.path.abspath(os.path.join(self.data_dir, folder, data)),
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

                print(
                    "Num GPUs Available: ", len(tf.config.list_physical_devices("GPU"))
                )

                self.train(normalized_frames, inputs)


the_bat = neural_network()

the_bat.load_data()
