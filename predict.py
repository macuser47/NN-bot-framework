import numpy as np
import os

import time
import cv2

import screen_capture
import keras
from tensorflow.keras.optimizers import SGD

from pynput.mouse import Button, Controller
import keyboard as ky
import mss

from queue import Queue

key_data_queue = Queue()

def load_model():
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'model')

    json_file = open(os.path.join(model_dir,'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(os.path.join(model_dir,"model.h5"))
    print("Loaded model from disk")
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
    
    return model

def capture(game_name, ctx):
    
    img = cv2.cvtColor(np.array(screen_capture.capture(game_name, ctx)), cv2.COLOR_BGRA2BGR)
    topleft , botright = np.array(screen_capture.get_cords(game_name))
    image = cv2.resize(img,(90,90),interpolation=cv2.INTER_AREA)

    height, width, _ = img.shape 

    normalized_frame = []

    for index,pixel in enumerate(image):
        normalized_frame.append(pixel/255)


    reshaped = np.reshape(normalized_frame, (90,90,3))

    numpy_frames = np.array([reshaped])

    return (numpy_frames,height,width,topleft)



def main():
    model = load_model()
    mouse = Controller()

    paused = True 
    with mss.mss() as sct:
        while True:
            time.sleep(1/30)
            img,h,w,topleft = capture("RotMGExalt", sct)
            prediction = model.predict(img)

            if ky.is_pressed('b'):
                break

            if ky.is_pressed('n'):
                paused = True

            if ky.is_pressed('m'):
                paused = False 

            key_data_queue.put(prediction[0])

            if paused:
                continue

            x = prediction[0][0] * w + topleft[0]
            y = prediction[0][1] * h + topleft[1]

            mouse.position = (x,y)

            # Left Click
            if prediction[0][2] > 0.5:
                mouse.press(Button.left)

            if prediction[0][2] <= 0.5:
                mouse.release(Button.left)

            # Keys
            keys = ['q', 'e', 'w', 'a','s','d','f','v','r',' ']
            for i, key in enumerate(keys):
                if prediction[0][i+3] > 0.5:
                    ky.press(key)
                else:
                    ky.release(key)

import socket
import struct
MSG_MAGIC = 0xDEADBEEF

def key_message(values):
    return struct.pack('LL', MSG_MAGIC, len(values)) + struct.pack('d'*len(values), *values)

def serve_keyinfo(port):
    print("Starting keyinfo server...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', port))
        sock.listen()
        print("Server listening on port {}".format(port))
        conn, addr = sock.accept()
        with conn:
            while True:
                conn.send(key_message(key_data_queue.get()))


from threading import Thread
if __name__ == '__main__':
    print("THIS IS THE START OF THE PROGRAM")
    keyinfo_thread = Thread(target=serve_keyinfo, args=(6624,))
    keyinfo_thread.daemon = True
    keyinfo_thread.start()
    main()
