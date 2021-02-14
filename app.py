import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import sys
import logging
from PIL import Image

from flask import Flask, request, jsonify

app = Flask(__name__)

logging.basicConfig(filename = "./log/app.log", level = logging.DEBUG)

@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    logging.debug("call hello")
    return 'Hello, World!'

model = tf.keras.models.load_model('bunnyfoot_model_20210124.h5')

def init_image(image):
    return np.array([img_to_array(image)])

@app.route('/predict', methods=['POST'])
def classifier():
    result = {"success":False, "probability":0}

    # FileStorage > array 전환
    img_data = Image.open(request.files['image'].stream).resize((64,64))
    input_img = np.array([img_to_array(img_data)])

    # model 입력
    logging.debug("============모델 입력 시작=====")
    output = model.predict_generator(input_img)
    #import json
    #result["probability"] = json.dumps(float(output[0][0]))
    result["probability"] = round(float(output[0][0]),3)
    result["success"] = True

    logging.debug(result)

    return result

if __name__ == "__main__":
    app.run()

