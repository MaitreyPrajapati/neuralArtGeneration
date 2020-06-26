import tensorflow as tf


def content_loss(original_activation, generated_activation):

    # Activation shape = (1, 31, 31, 512)
    m, n_H, n_W, n_C = original_activation.get_shape().as_list()

    # Reshaping activations
    # a_C_unrolled = tf.reshape(original_activation, shape=(-1, n_C))
    # a_G_unrolled = tf.reshape(generated_activation, shape=(-1, n_C))

    loss = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(original_activation, generated_activation)))
    return loss


def gram_matrix(activation):

    activation = tf.matmul(activation, tf.transpose(activation))
    return activation


def layer_style_loss(original_activation, generated_activation):

    m, n_H, n_W, n_C = original_activation.get_shape().as_list()

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

def total_loss(layer_weights, content_activations, style_activations, generated_activations, alpha=0.6, beta=0.4):

    c_loss = content_loss(content_activations[-1], generated_activations[-1])
    s_loss = overall_style_loss(layer_weights, style_activations[:-1], generated_activations[:-1])

    t_loss = alpha * c_loss + beta * s_loss

    return t_loss





