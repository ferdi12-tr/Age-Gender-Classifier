from config import age_gender_config as config
from preprocessing import MeanPreprocessor
from preprocessing import SimplePreprocessor
from preprocessing import ImageToArrayPreprocessor
from utils import DatasetGenerator
from utils.rank_accuracy import rank5_accuracy
from keras.models import load_model
import numpy as np
import json

means = json.loads(open(config.DATASET_MEAN).read())

# set the preprocessing like we did at train stage
sp = SimplePreprocessor(224, 224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()  # convert image to keras-compatible array

# load the trained model
model = load_model(config.BEST_MODEL_PATH)

# fetching test dataset by adjusting generator
testGen = DatasetGenerator(config.TEST_CSV,
                           64,
                           preprocessors=[sp, mp, iap],
                           classes=config.NUM_CLASSES
                           )

predictions = model.predict(testGen.generator(),
                            steps=testGen.numImages // 64,
                            max_queue_size=64*2)

(rank1, rank5) = rank5_accuracy(predictions, testGen.labels)
print("rank-1 accuracy: {:.2f}%".format(rank1 * 100))
print("rank-5 accuracy: {:.2f}%".format(rank5 * 100))









