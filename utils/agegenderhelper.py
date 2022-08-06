import numpy as np
import glob
import cv2
import os
from config import age_gender_config as config


class AgeGenderHelper:
    def __init__(self, config):
        self.config = config
        self.ageBins = self.__build_age_bins()
        self.age_dict = {  # we are doing label encoding
            "(0, 2)": 0,
            "(4, 6)": 1,
            "(8, 13)": 2,
            "(15, 20)": 3,
            "(25, 32)": 4,
            "(38, 43)": 5,
            "(48, 53)": 6,
            "(60, inf)": 7
        }

    def __build_age_bins(self):
        ageBins = [(0, 2), (4, 6), (8, 13), (15, 20), (25, 32), (38, 43), (48, 53), (
        60, np.inf)]  # store the configuration object and build the age bins used for constructing class labels
        return ageBins

    def __to_label(self, age, gender):
        if self.config.DATASET_TYPE == "age":
            return self.__to_age_label(age)
        return self.__to_gender_label(gender)

    def __to_age_label(self, age):  # inside in dataset age label is string
        label = None
        age = age.replace("(", "").replace(")", "").split(", ")
        (ageLower, ageUpper) = np.array(age, dtype="int")  # from string to int

        for (lower, upper) in self.ageBins:
            # determine if the age falls into the current bin
            if ageLower >= lower and ageUpper <= upper:
                label = "({}, {})".format(lower, upper)
                label = self.age_dict[label]
                break
        return label

    def __to_gender_label(self, gender):
        # return 0 if the gender is male, 1 if the gender is female
        return 0 if gender == "m" else 1

    def build_paths_labels_adience(self):
        paths = []
        labels = []
        foldPaths = os.path.sep.join([self.config.LABELS_PATH, "*.txt"])
        foldPaths = glob.glob(
            foldPaths)  # Return a possibly-empty list of path names that match pathname, which must be a string containing a path specification.

        for foldPath in foldPaths:  # loop over the folds paths
            rows = open(foldPath).read()
            rows = rows.strip().split("\n")[1:]  # skip the header

            for row in rows:
                row = row.split("\t")
                (userID, imagePath, faceID, age, gender) = row[:5]

                if age[0] != "(" or gender not in ("m", "f"):  # skip invalid age or gender
                    continue

                # construct path to each image and build class labels
                p = "landmark_aligned_face.{}.{}".format(faceID, imagePath)
                p = os.path.sep.join([self.config.IMAGE_PATH, userID, p])
                label = self.__to_label(age, gender)

                if label is None:  # if the label is None, then the age does not fit into our age brackets, ignore the sample
                    continue

                paths.append(p)
                labels.append(label)
        return (paths, labels)

    def build_paths_labels_utkface(
            self):  # this method for UTKFace dataset to create list of path to picture and list of labels
        paths = []
        labels = []
        count = 0
        for path in os.listdir(config.UTKFACE_PATH):
            try:
                (age, gender, etnicty, r) = path.strip().split("_")  # we are looking for age and gender
            except:
                continue

            age = int(age)  # from string to int

            # we have unbalanced data at 25 - 32 age, about 6 times compared to other age groups,
            # reducing that group 6 times may be good for overfitting
            if 25 <= age <= 32 and count < 5 and config.DATASET_TYPE == "age":
                count += 1
                continue
            else:
                count = 0

            label = self.__to_utkface_label(age, gender)

            if label is None:
                continue
            else:
                labels.append(label)
                paths.append(os.path.sep.join([config.UTKFACE_PATH, path]))

        return (paths, labels)

    def __to_utkface_label(self, age, gender):
        label = None
        if config.DATASET_TYPE == "gender":
            return 0 if gender == "0" else 1
        else:
            for lower, upper in self.ageBins:
                if lower <= age <= upper:
                    label = "({}, {})".format(lower, upper)
                    label = self.age_dict[label]
                    return label

            return label








