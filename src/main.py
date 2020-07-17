import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL
from PIL import Image
import h5py
from LoadVggModel import model_class
from UpdatedModel import custom_model
from LossFunctions import total_loss
from noisy_image import generate_noisy_image
import time
import os

STYLE_WEIGHTS = [0.2] * 4
SAVE_IMAGE_PATH = '../interm_images/'
MEANS = np.array([123.68, 116.779, 103.939], dtype='float32').reshape((1,1,1,3))

def get_class_model():
    vgg_m = model_class()
    return vgg_m


# How to normalize it properly ???
def load_image(image_path):
    image = np.array(keras.preprocessing.image.load_img(image_path, target_size=(500, 500)), dtype='float32').reshape(1, 500, 500, 3) - MEANS
    return image


modelClass = get_class_model() # Model Class
vgg_model = modelClass.return_model() # Actual Model
vgg_weights = modelClass.get_all_weights()

custom_model1 = custom_model(vgg_model, vgg_weights)


def compute_grads(custom_output_content, custom_output_style, generated_image):
    with tf.GradientTape() as tape:

        #Calculating activations for the random image from inside the GradientTape
        generated_image_activations = custom_model1(generated_image)

        #Loss is calculated from here, custom_output_content is an output tensor of the model, custom_output_style is an output tensor of the model
        loss = total_loss(STYLE_WEIGHTS, custom_output_content, custom_output_style, generated_image_activations)

    return tape.gradient(loss, generated_image), loss


opt = tf.optimizers.Adam(learning_rate=10.0)

def nn_model(input_tensor, content_image, style_image, save_folder_path):

    custom_output_content = custom_model1(content_image) # Content output of the model for the image
    custom_output_style = custom_model1(style_image) # Style output of the model for the  image

    norm_means = np.array([103.939, 116.779, 123.68])
    _min = -norm_means
    _max = 255 - norm_means

    for iter in range(1000):

        grads, loss = compute_grads(custom_output_content, custom_output_style, input_tensor)
        opt.apply_gradients([(grads, input_tensor)])
        clipped = tf.clip_by_value(input_tensor, _min, _max)
        input_tensor.assign(clipped)
        print(loss)

        #Saving the randomly generated image at every 10 iterations at it's specific folder
        if(not iter%10):

            curr_image_path = '../interm_images/' + save_folder_path + '/' + str(iter) + '.jpg'

            image = np.squeeze(input_tensor.numpy() + MEANS )
            keras.preprocessing.image.save_img(curr_image_path, image, data_format='channels_last')

            print('Iter {} : Saved image at {}'.format(iter, curr_image_path))

    return True


def run_on_all():

    #Listing the style and content images
    style_images = os.listdir('style_image/')
    content_images = os.listdir('content_image/')

    #This had to be done because of hidden cache files in the folder are also listed in the list, we need only jpg files
    style_images = sorted([x for x in style_images if 'jpg' in x], reverse=True)
    content_images = sorted([x for x in content_images if 'jpg' in x], reverse=True)

    # For all images in content images apply all styles
    for ci in content_images:
        for si in style_images:

            #Loading images
            content_input = load_image('content_image/' + ci)
            style_input = load_image('style_image/' + si)

            #Generating random images
            random_image = generate_noisy_image(content_input)
            random_tensor = tf.Variable(random_image, dtype='float32')

            #Creating directory for each content for each style
            save_folder_path = ci.split('.')[0] + '_' + si.split('.')[0]
            os.system('mkdir ../interm_images/{}'.format(save_folder_path))

            nn_model(random_tensor, content_input, style_input, save_folder_path)


run_on_all()

