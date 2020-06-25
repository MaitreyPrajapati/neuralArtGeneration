import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL
import h5py
from load_vgg_model import model_class
from updatedModel import custom_model


def get_class_model():
    vgg_m = model_class()
    return vgg_m

## Loading Image
## How to normalize it properly ???
def load_image(image_path):
    return np.array(keras.preprocessing.image.load_img(image_path, target_size=(500,500)), dtype='float32').reshape(1,500,500,3) / 255.


modelClass = get_class_model() ## Model Class
vgg_model = modelClass.return_model() ## Actual Model
vgg_weights = modelClass.get_all_weights()

input_image = load_image('1.jpg')

custom_model1 = custom_model(vgg_model, vgg_weights)

custom_output = custom_model1(input_image)
print('{} \n {}'.format(custom_output, len(custom_output)))


