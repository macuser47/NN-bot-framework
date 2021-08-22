import numpy as np
import os

import time
import cv2

import screen_capture
import keras
from tensorflow.keras.optimizers import SGD

from pynput.mouse import Button, Controller
import keyboard as ky

def predict(model, X):

    Y = model.predict(X)

    return Y


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
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001), metrics=['categorical_crossentropy'])
    
    return model

def capture(game_name):
    
    img = screen_capture.capture(game_name)
    topleft , botright = screen_capture.get_cords(game_name)
    image = cv2.resize(img,(90,90),interpolation=cv2.INTER_AREA)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    burger = img.shape

    height = burger[0]
    width = burger[1]

    normalized_frame = []

    for index,pixel in enumerate(image):
        normalized_frame.append(pixel/255)


    reshaped = np.reshape(normalized_frame, (90,90, 3))

    numpy_frames = np.array([reshaped])

    return (numpy_frames,height,width,topleft)



def main():
    

    model = load_model()

    mouse = Controller()

    while True:
        ky.press
        time.sleep(1/30)
        img,h,w,topleft = capture("RotMGExalt")
        prediction = predict(model, img)

        if ky.is_pressed('p'):
            break

        print(prediction)

        

        # a = []

        # for i in range(4):
        #     a.append(prediction[0][i])

        # bigger = max(a)

        # print(bigger)

        # if a[0] == bigger:
        #     ky.press('left')
        #     time.sleep(1/30)
        #     ky.release('left')
        #     time.sleep(1/30)
        # if a[1] == bigger:
        #     ky.press('right')
        #     time.sleep(1/30)
        #     ky.release('right')
        #     time.sleep(1/30)

        x = prediction[0][0] * w + topleft[0]
        y = prediction[0][1] * h + topleft[1]

        mouse.position = (x,y)

        button_list = []

        for i in range(len(prediction) - 2):

            button_list.append(prediction[i + 2])

        # bigger = max(button_list)

        'q', 'e', 'w', 'a','s','d','f','v','r',' '

        #Left Click
        if prediction[0][2] > 0.5:
            mouse.press(Button.left)

        if prediction[0][2] < 0.5:
            mouse.release(Button.left)

        if prediction[0][3] > 0.5:
            ky.press('q')
        if prediction[0][3] < 0.5:
            ky.release('q')

        if prediction[0][4] > 0.5:
            ky.press('e')
        if prediction[0][4] < 0.5:
            ky.release('e')





        if prediction[0][5] > 0.5:
            ky.press('w')
        if prediction[0][5] < 0.5:
            ky.release('w')

        if prediction[0][6] > 0.5:
            ky.press('a')
        if prediction[0][6] < 0.5:
            ky.release('a')

        if prediction[0][7] > 0.5:
            ky.press('s')
        if prediction[0][7] < 0.5:
            ky.release('s')

        if prediction[0][8] > 0.5:
            ky.press('d')
        if prediction[0][8] < 0.5:
            ky.release('d')

        if prediction[0][9] > 0.5:
            ky.press('f')
            time.sleep(1/30)
            ky.release('f')
            time.sleep(1/30)
        if prediction[0][10] > 0.5:
            ky.press('v')
            time.sleep(1/30)
            ky.release('v')
            time.sleep(1/30)
        if prediction[0][11] > 0.5:
            ky.press('r')
            time.sleep(1/30)
            ky.release('r')
            time.sleep(1/30)





if __name__ == '__main__':
    print("THIS IS THE START OF THE PROGRAM")
    main()




