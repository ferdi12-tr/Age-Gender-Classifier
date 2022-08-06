import tensorflow as tf
from config import age_gender_config as config


converter_best = tf.lite.TFLiteConverter.from_saved_model(config.BEST_MODEL_PATH)
tflite_model_best = converter_best.convert()

with open(config.BEST_TFLITE_PATH, "wb") as f:
    f.write(tflite_model_best)

