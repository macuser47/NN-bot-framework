import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Reshape, Activation, Conv2D , LSTM, Flatten, MaxPooling2D, TimeDistributed, Dropout, LSTM
from keras.layers import LeakyReLU, SimpleRNN
from keras.optimizers import Adam, Adagrad, SGD, RMSprop

from pympler import asizeof

from keras.datasets import cifar10

import cv2
import keras.backend as K
from numpy import loadtxt
import numpy as np
import time
from keras.callbacks import LambdaCallback

class neural_network:

    

    def __init__(self):
        self.cwd = os.getcwd()
        self.data_dir = os.path.join(self.cwd, 'recorded data')
        self.model_dir = os.path.join(self.cwd, 'model')

    def unpack_mp4(self,video_file):
        vidObj = cv2.VideoCapture(video_file) 
        success = 1
        frames = []

        while success:
            success, image = vidObj.read()
            try:
                h,w,c = image.shape
            except Exception as e:
                print(e)

            try:
                image = cv2.resize(image,(150,150),interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            except Exception as e:
                print(e)

        return (frames,h,w)    

    # 
    def customLoss(yTrue,yPred):
        return

    def optimizer(self):
        return SGD(lr=0.001, momentum=0.9)

    def create_model(self):

        model = Sequential()

        # model.add(Conv2D(64, (3, 3), data_format='channels_first', border_mode='valid', activation="relu",input_shape=(1,150,150)))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten(input_shape=(1,150,150)))

        model.add(Dense(128))
        model.add(Dense(64))
        model.add(Dense(22500))

        model.compile(loss='mean_squared_error', optimizer=self.optimizer(), metrics=['mse'])
        return model

    def train(self,frames,inputs):

        if len(os.listdir(self.model_dir)) < 1:
           model = self.create_model()
        else:
            json_file = open(os.path.join(self.model_dir,'cnmodel.json'), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)

            # load weights into new model
            model.load_weights(os.path.join(self.model_dir,"cnmodel.h5"))
            print("Loaded model from disk")
            model.compile(loss='mean_squared_error', optimizer=self.optimizer(), metrics=['mse'])

        frames = np.array(frames)

        reshaped_frames = [np.reshape(frame, (1,150,150)) for frame in frames]

        numpy_reshaped_frames = np.array(reshaped_frames)

        model.summary()

        model.fit(numpy_reshaped_frames,numpy_reshaped_frames, epochs=50, batch_size=64)

        print('PREDICTION')

        for frame in reshaped_frames:
            test_ar = [frame]
            test_aaaa = np.array(test_ar)

            print(model.predict(test_aaaa))

        model_json = model.to_json()

        with open(os.path.join(self.model_dir,"cnmodel.json"), "w") as json_file:
            json_file.write(model_json)

        model.save_weights(os.path.join(self.model_dir,"cnmodel.h5"))

    def load_data(self):
        for folder in os.listdir(self.data_dir):
            for data in os.listdir(os.path.join(self.data_dir, folder)):

                if data.endswith('.mp4'):
                    frames,h,w = self.unpack_mp4(os.path.abspath(os.path.join(self.data_dir,folder,data)))
                
                if data.endswith('.csv'):
                    inputs = loadtxt(os.path.abspath(os.path.join(self.data_dir,folder,data)), delimiter=',')               
            
            mouse_inputs = []

            for index,element in enumerate(inputs):
                inputs[index][0] = inputs[index][0] / h
                inputs[index][1] = inputs[index][1] / w
                        

            normalized_frame = []
            normalized_frames = []
        
            for index,frame in enumerate(frames):
                for index,pixel in enumerate(frame):
                    normalized_frame.append(pixel / 255)
                normalized_frames.append(normalized_frame)
                normalized_frame = []

                # for index,frame in enumerate(normalized_frames):
                #     for index,pixel in enumerate(frame):
                #         print(pixel)

            print(str(folder))
            print(f"{asizeof.asizeof(normalized_frames) / 1024}")

                    
            self.train(normalized_frames,inputs)



the_bat = neural_network()

the_bat.load_data()


        







        


