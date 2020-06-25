import tensorflow as tf
import numpy as np

def custom_model(vgg_model, vgg_weights):

    output_layers = []
    max_pool_set = {3,6,11,16,21}
    vgg_layers = vgg_model.layers


    for layer_index in range(len(vgg_layers)):
        if(layer_index in max_pool_set):
            output_layers.append(
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2), strides=(2,2), padding='valid', data_format='channels_last'
                )
            )
        else:
            output_layers.append(
                vgg_layers[layer_index]
            )

    output_model = tf.keras.layers.Input(shape= (500,500,3))
    input_layer = prev_layer = output_model

    for layer in output_layers[1:]:
        layer._inbound_nodes = []
        curr_layer = layer(prev_layer)
        prev_layer = curr_layer

    functional_model = tf.keras.models.Model(inputs=[input_layer], outputs=[prev_layer], name='custom_model_01')

    return functional_model

# def create_sequential_model(layers):
#     seq_model = tf.keras.Sequential()
#
#     for layer in layers:
#         seq_model.add(layer)
#
#     return seq_model
#



