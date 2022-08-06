import os.path
import tensorflow.python.keras.optimizers
from preprocessing import MeanPreprocessor
from preprocessing import SimplePreprocessor
from preprocessing import ImageToArrayPreprocessor
from utils import DatasetGenerator
from utils import TrainingMonitor
from network import CustomMobileNet
from config import age_gender_config as config
import pandas as pd
import json
import cv2
import matplotlib
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import adam_v2
from keras.utils import np_utils
from keras import losses

matplotlib.use(
    "Agg")  # ensuring the backend is set such that we can save figures and plots to disk as our network trains.

# load the RGB image means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize image processors
aug = ImageDataGenerator(rotation_range=10, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode='nearest')
sp = SimplePreprocessor(224, 224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()  # convert image to keras-compatible array

# set the dataset generators
trainGen = DatasetGenerator(config.TRAIN_CSV, config.BATCH_SIZE, config.NUM_CLASSES, [mp, sp, iap], aug=aug)
valGen = DatasetGenerator(config.VAL_CSV, config.BATCH_SIZE, config.NUM_CLASSES, [mp, sp, iap])

monitor_path = os.path.sep.join([config.CHECKPOINT_PATH, "{}_{}.png".format(os.getpid(), config.DATASET_TYPE)])  # the path to track loss and accuracy on graph (see checkpoint folder)
callbacks = [TrainingMonitor(monitor_path),
             ModelCheckpoint(filepath=config.BEST_MODEL_PATH, save_best_only=True, verbose=1),  # save the best model after each epoch
             EarlyStopping(monitor='val_loss', patience=7)]  # set early stopping

opt = adam_v2.Adam(learning_rate=1e-5)
model = CustomMobileNet(width=224, height=224, depth=3, classes=config.NUM_CLASSES).build()  # build the model
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])  # for gender
model.summary()

model.fit(trainGen.generator(),
          steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
          validation_data=valGen.generator(),
          validation_steps=valGen.numImages // config.BATCH_SIZE,
          epochs=100,
          max_queue_size=config.BATCH_SIZE * 2,
          callbacks=callbacks,
          verbose=1)

# save the model to file
model.save(config.MODEL_PATH)  # although best model saving, we are also saving the last model



