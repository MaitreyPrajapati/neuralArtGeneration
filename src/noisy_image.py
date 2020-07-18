import numpy as np

def generate_noisy_image(content_image, noise_ratio=0.5):
    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20,(1, 800, 500, 3)).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image