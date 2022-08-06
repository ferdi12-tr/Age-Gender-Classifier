# define a class responsible for yielding batches of images and labels from our HDF5 dataset.
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
import pandas as pd
import cv2


class DatasetGenerator:
    def __init__(self, dbPath, batchSize, classes, preprocessors=None, aug=None, binarize=True):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        df = pd.read_csv(dbPath)
        self.image_paths = df['image_path'].values
        self.labels = df['label'].values
        self.numImages = len(self.labels)

    def generator(self, passes=np.inf):  # Think of the passes value as the total number of epochs
        epochs = 0

        # keep looping infinitely -- the model will stop once we have reach the desired number of epochs
        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.image_paths[i: i + self.batchSize]
                labels = self.labels[i: i + self.batchSize]

                """
                binarize: Typically we will store class labels as single integers inside our dataset;
                however, as we know, if we are applying categorical cross-entropy or binary cross-entropy as
                our loss function, we first need to binarize the labels as one-hot encoded vectors – this switch
                indicates whether or not this binarization needs to take place (which defaults to True).
                """
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)

                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    procImages = []  # initialize the list of processed images
                    for image in images:
                        image = cv2.imread(image)

                        if image is None:
                            continue
                        else:
                            for p in self.preprocessors:
                                image = p.preprocess(image)
                            procImages.append(image)
                    images = np.array(procImages)

                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))

                yield (
                images, labels)  # yield a 2-tuple of the batch of images and labels to the calling Keras generator
            epochs += 1
