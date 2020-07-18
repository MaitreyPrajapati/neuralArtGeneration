import tensorflow as tf


def content_loss(original_activation, generated_activation):

    # Activation shape = (1, 31, 31, 512)
    n_H, n_W, n_C = original_activation.get_shape().as_list()

    # Reshaping activations
    # a_C_unrolled = tf.reshape(original_activation, shape=(-1, n_C))
    # a_G_unrolled = tf.reshape(generated_activation, shape=(-1, n_C))

    loss = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(original_activation, generated_activation)))
    return loss

def overall_content_loss(original_activations, generated_activations):
    oC = 0

    for index in range(len(original_activations)):

        oC += content_loss(original_activations[index], generated_activations[index])

    oC /= len(original_activations)

    return oC

def gram_matrix(activation):

    activation = tf.matmul(activation, tf.transpose(activation))
    return activation


def layer_style_loss(original_activation, generated_activation):

    m, n_H, n_W, n_C = original_activation.get_shape().as_list() ## Only to be used when there are more than one layer defining the loss
    # n_H, n_W, n_C = original_activation.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.transpose(tf.reshape(original_activation, shape=(-1, n_C)))
    a_G = tf.transpose(tf.reshape(generated_activation, shape=(-1, n_C)))

    # gram matrices
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    loss = 1 / (4 * n_C ** 2 * (n_H * n_W) ** 2) * tf.reduce_sum((GS - GG) ** 2)
    return loss

def overall_style_loss(layer_weights, original_activations, generated_activations):

    total_style_loss = 0

    for index in range(len(layer_weights)):
        weight = layer_weights[index]
        loss = layer_style_loss(original_activations[index], generated_activations[index])
        total_style_loss += (weight * loss)

    return total_style_loss

def total_loss(layer_weights, content_activations, style_activations, generated_activations, alpha=1, beta=1000):

    c_loss = overall_content_loss(content_activations[0], generated_activations[0])
    s_loss = overall_style_loss(layer_weights, style_activations, generated_activations)

    t_loss = alpha * c_loss + beta * s_loss

    return t_loss





