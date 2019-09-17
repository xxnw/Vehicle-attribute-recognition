# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io

from utils_recog import load_model

if __name__ == '__main__':
    img_width, img_height = 224, 224
    model = load_model()
    model.load_weights('models/model.96-0.89.hdf5')

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    #test_path = 'data/test/'
    #test_images = [f for f in os.listdir(test_path) if
    #               os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]
    
    #read and save video
    cap = cv.VideoCapture('test/north.avi')
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output_3.avi', fourcc, 25.0, (960, 540))

    #num_samples = 12
    #samples = random.sample(test_images, num_samples)
    results = []
    #for i, image_name in enumerate(samples):
        #filename = os.path.join(test_path, image_name)
        #print('Start processing image: {}'.format(filename))
        #bgr_img = cv.imread(filename)
        
    while(cap.isOpened()):
        # read image data
        ret, img = cap.read()
        img = cv.resize(img, (960, 540), cv.INTER_LINEAR)
        bgr_img = cv.resize(img, (img_width, img_height), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        #text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
        text = ('{}, {}'.format(class_names[class_id][0][0], prob))
        print(text)
        results.append({'label': class_names[class_id][0][0], 'prob': '{:.4}'.format(prob)})
        cv.putText(img, text, (20, 20), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        #cv.imwrite('images/{}_out.png'.format(i), bgr_img)
        out.write(img)
    cap.release()
    out.release()

    #print(results)
    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

    K.clear_session()
