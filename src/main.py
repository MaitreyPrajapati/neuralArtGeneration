import cv2
import numpy as np
import tensorflow as tf
import h5py
from load_vgg_model import model_class
from updatedModel import custom_model


def get_class_model():
    vgg_m = model_class()
    return vgg_m

def resize(image):
    return cv2.resize(image,(500,500))


modelClass = get_class_model() ## Model Class
vgg_model = modelClass.return_model() ## Actual Model
vgg_weights = modelClass.get_all_weights()


input_image = cv2.imread('1.jpg')
input_image = resize(input_image)

custom_model1 = custom_model(vgg_model, vgg_weights)

print(custom_model1.summary())


