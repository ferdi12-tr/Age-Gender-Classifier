from config import age_gender_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import AgeGenderHelper
import pandas as pd
import csv
import numpy as np
import progressbar
import json
import cv2


agh = AgeGenderHelper(config)
(adience_paths, adience_labels) = agh.build_paths_labels_adience()
(utk_paths, utk_labels) = agh.build_paths_labels_utkface()

# combine the adience and utk dataset into this list
trainPaths = []
trainLabels = []

for path, label in zip(adience_paths, adience_labels):
    trainPaths.append(path)
    trainLabels.append(label)

for path, label in zip(utk_paths, utk_labels):
    trainPaths.append(path)
    trainLabels.append(label)


# number of images that should be used for validation and test
numVal = int(len(trainPaths) * config.NUM_VAL_IMAGES)
numTest = int(len(trainPaths) * config.NUM_TEST_IMAGES)

split = train_test_split(trainPaths, trainLabels, test_size=numVal, stratify=trainLabels)
(trainPaths, valPaths, trainLabels, valLabels) = split

split = train_test_split(trainPaths, trainLabels, test_size=numTest, stratify=trainLabels)
(trainPaths, testPaths, trainLabels, testLabels) = split

datasets = [  # image paths, label paths, output file paths
    ("train", trainPaths, trainLabels, config.TRAIN_CSV),
    ("val", valPaths, valLabels, config.VAL_CSV),
    ("test", testPaths, testLabels, config.TEST_CSV)
]
print("trainPaths: " + str(len(trainPaths)))
print("valPaths: " + str(len(valPaths)))
print("testPaths: " + str(len(testPaths)))

# initialize the lists of RGB channel averages
(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:

    print("-----> Building {}...".format(outputPath))
    header = ['image_path', 'label']
    with open(outputPath, 'w', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        # initialize the progress bar
        widgets = ["-----> Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ",
                   progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

        for (i, (path, label)) in enumerate(zip(paths, labels)):
            image = cv2.imread(path)

            if image is None:
                continue
            else:
                if dType == "train":
                    (b, g, r) = cv2.mean(image)[:3]
                    R.append(r)
                    G.append(g)
                    B.append(b)
            writer.writerow([path, label])
            pbar.update(i)
    pbar.finish()

# serialize the means
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()

print("****** The process has finished *****")

