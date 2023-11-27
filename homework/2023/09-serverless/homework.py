from tensorflow.keras.models import load_model
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
import tensorflow as tf

from io import BytesIO
from urllib import request
from PIL import Image

import numpy as np
import os

MODEL_NAME = os.getenv('MODEL_NAME', 'bees-wasps.tflite')

def load_model(model):
    return load_model(model)

def compress_model():
    model = load_model("bees-wasps.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model

def save_model():
    with open('bees-wasps.tflite', 'wb') as f:
        f.write(compress_model())

def model_index():
    interpreter = tflite.Interpreter(model_path=MODEL_NAME)
    interpreter.allocate_tensors()

    index = {'input': interpreter.get_input_details()[0]['index'],
             'output': interpreter.get_output_details()[0]['index'],
             'interpreter': interpreter}
    return index

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(x):
    return (x/ 255.0)

def predict(X):
    model = model_index()
    interpreter = model['interpreter']
    interpreter.set_tensor(model['input'], X)
    interpreter.invoke()

    prediction = interpreter.get_tensor(model['output'])
    return prediction

def test_predict(image= "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"):
    print(f"Image: {image}")
    org_image = download_image(image)
    image = prepare_image(org_image, target_size=(150, 150))
    x = np.array(image , dtype='float32')
    X = np.array([x])
    X = preprocess_image(X)
    print('prediction : ', predict(X))
    return X

def lambda_handler(event, context):
    print("Lambda is running ")
    url = event['url']
    pred = test_predict(url)
    result = {
        'prediction': pred
    }

    return result

if __name__ == '__main__':
    test_predict()
