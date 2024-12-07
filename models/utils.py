import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import TextVectorization

def preprocess_image(image_path, image_size=(299, 299)):
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image) / 255.0
    return tf.expand_dims(image, axis=0)

vectorization = TextVectorization(
    max_tokens=13000,
    output_mode="int",
    output_sequence_length=24
)