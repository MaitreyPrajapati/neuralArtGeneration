import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL
import h5py
from LoadVggModel import model_class
from UpdatedModel import custom_model
from LossFunctions import total_loss
from noisy_image import generate_noisy_image

STYLE_WEIGHTS = [0.2] * 5

def get_class_model():
    vgg_m = model_class()
    return vgg_m


# How to normalize it properly ???
def load_image(image_path):
    image = np.array(keras.preprocessing.image.load_img(image_path, target_size=(500,500)), dtype='float32').reshape(1,500,500,3) / 255.
    return image


modelClass = get_class_model() # Model Class
vgg_model = modelClass.return_model() # Actual Model
vgg_weights = modelClass.get_all_weights()

input_image = load_image('1.jpg')
style_image = load_image('2.jpg')
random_image = generate_noisy_image(input_image)


custom_model1 = custom_model(vgg_model, vgg_weights)

custom_output_content = custom_model1(input_image)
custom_output_style = custom_model1(style_image)
custom_random_image = custom_model1(random_image)

tl = total_loss(STYLE_WEIGHTS, custom_output_content, custom_output_style, custom_random_image)


print(tl)