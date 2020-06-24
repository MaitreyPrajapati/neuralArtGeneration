import tensorflow as tf
import numpy as np

class model():

    def __init__(self):
        #Actual Model
        self.model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet', input_tensor=None, input_shape=(300, 300, 3),
            pooling='avg', classes=1000
            )
        #For Weights
        self.weights = {}

    def return_model(self):
        return self.model
