# -*- coding: utf-8 -*-
"""
Created on Mon May 24 06:23:53 2021

@author: fkoca

we’ll create a TrainingMonitor callback that will be called at the
end of every epoch when training a network with Keras. This monitor will serialize the loss and
accuracy for both the training and validation set to disk, followed by constructing a plot of the data.
Applying this callback during training will enable us to babysit the training process and spot
overfitting early, allowing us to abort the experiment and continue trying to tune our parameters.
"""

# import the necessary packages
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store the output path for the figure, the path to the JSON serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
        
    def on_train_begin(self, logs={}):
        
        # initialize the history dictionary
        self.H = {}
        
        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                
                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and trim any entries that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):  # The on_epoch_end method is automatically supplied to parameters from Keras.
        
        # loop over the logs and update the loss, accuracy, etc. for the entire training process
        for (k,v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
            """
            After this code executes, the dictionary H now has four keys:
            1. train_loss
            2. train_acc
            3. val_loss
            4. val_acc
            """
        # check to see if the training history should be serialized to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()
        
        # ensure at least two epochs have passed before plotting (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_acc")
            
            plt.title("Train loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("loss/accuracy")
            plt.legend()
            
            # save the figure
            plt.savefig(self.figPath)
            plt.close()
            
            

                        





















