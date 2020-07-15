import tensorflow as tf
import numpy as np

class model_class():

    def __init__(self):
        #Actual Model
        self.model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet', input_tensor=None, input_shape=(500, 500, 3),
            pooling='avg', classes=1000
            )
        #For Weights
        self.weights = {}

        #Load weights in the dictionary
        self.load_model_in_dict()

    def return_model(self):
        return self.model

    def get_all_weights(self):
        return self.weights

    def get_layer_weights(self, layer_name):
        return self.weights[layer_name]

    def load_model_in_dict(self):

        layers = self.model.layers

        for layer in layers:
            layer_name = layer.name
            layer_weights = np.array(layer.get_weights())
            self.weights[layer_name] = layer_weights

        return True






