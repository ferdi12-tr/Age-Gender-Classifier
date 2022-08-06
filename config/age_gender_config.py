from os import path

DATASET_TYPE = "age"
BASE_PATH = r"C:\Users\fkoca\Desktop\thalespy\datasets\adience"
UTKFACE_PATH = r"C:\Users\fkoca\Desktop\thalespy\datasets\UTKFace"
OUTPUT_BASE = r"C:\Users\fkoca\Desktop\thalespy\output"
CHECKPOINT_BASE = r"C:\Users\fkoca\Desktop\thalespy\checkpoints"

IMAGE_PATH = path.sep.join([BASE_PATH, "aligned"])
LABELS_PATH = path.sep.join([BASE_PATH, "folds"])

NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

BATCH_SIZE = 8
NUM_DEVICES = 2

if DATASET_TYPE == "age":
    NUM_CLASSES = 8
    DATASET_MEAN = path.sep.join([OUTPUT_BASE, "age_mean.json"])

    TRAIN_CSV = path.sep.join([OUTPUT_BASE, "age_train.csv"])
    VAL_CSV = path.sep.join([OUTPUT_BASE, "age_val.csv"])
    TEST_CSV = path.sep.join([OUTPUT_BASE, "age_test.csv"])

    CHECKPOINT_PATH = path.sep.join([CHECKPOINT_BASE, "age"])
    MODEL_PATH = path.sep.join([CHECKPOINT_PATH, 'AgeClass'])
    BEST_MODEL_PATH = path.sep.join([CHECKPOINT_PATH, 'AgeClass_Best'])
    TFLITE_PATH = path.sep.join([OUTPUT_BASE, "age_model.tflite"])
    BEST_TFLITE_PATH = path.sep.join([OUTPUT_BASE, "best_age_model.tflite"])

elif DATASET_TYPE == "gender":
    NUM_CLASSES = 2
    DATASET_MEAN = path.sep.join([OUTPUT_BASE, "gender_mean.json"])

    TRAIN_CSV = path.sep.join([OUTPUT_BASE, "gender_train.csv"])
    VAL_CSV = path.sep.join([OUTPUT_BASE, "gender_val.csv"])
    TEST_CSV = path.sep.join([OUTPUT_BASE, "gender_test.csv"])

    CHECKPOINT_PATH = path.sep.join([CHECKPOINT_BASE, "gender"])
    MODEL_PATH = path.sep.join([CHECKPOINT_PATH, 'GenderClass'])
    BEST_MODEL_PATH = path.sep.join([CHECKPOINT_PATH, 'GenderClass_Best'])
    BEST_TFLITE_PATH = path.sep.join([OUTPUT_BASE, "best_gender_model.tflite"])
    TFLITE_PATH = path.sep.join([OUTPUT_BASE, "gender_model.tflite"])
"""
The DATASET_MEAN file will be used to store the average red, green, and blue pixel intensity
values across the entire (training) dataset. When we train our network, we’ll subtract the mean
RGB values from every pixel in the image (the same goes for testing and evaluation as well). This
method, called mean subtraction, is a type of data normalization technique and is more often used
than scaling pixel intensities to the range [0;1] as it’s shown to be more effective on large datasets
and deeper neural networks.
"""