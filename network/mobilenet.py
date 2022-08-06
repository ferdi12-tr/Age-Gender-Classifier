import keras.applications.mobilenet_v2
from keras import backend
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import layers
import tensorflow as tf


class CustomMobileNet:
    def __init__(self, width, height, depth, classes):
        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes

        if backend.image_data_format() == "channels_first":
            self.inputShape = (self.depth, self.height, self.width)
            self.channel_axis = 1
        else:
            self.inputShape = (self.height, self.width, self.depth)
            self.channel_axis = -1

    def build(self):
        base_model = MobileNetV2(input_shape=self.inputShape, include_top=False, weights='imagenet')
        base_model.trainable = True  # fine tuning the entire model

        inputs = layers.Input(shape=self.inputShape)
        preprocess_input = keras.applications.mobilenet_v2.preprocess_input
        x = preprocess_input(inputs)
        x = base_model(x, training=False)  # it is still running in inference mode since we passed training=False

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return model

