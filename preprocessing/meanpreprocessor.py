import cv2


class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):  # take the channel means
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        (B, G, R) = cv2.split(image.astype("float32"))

        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        return cv2.merge([B, G, R])  # return the normalized channel as image

