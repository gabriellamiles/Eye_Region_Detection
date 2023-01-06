"Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory"

import pandas as pd
import numpy as np
import datetime
import random

import matplotlib.style
import matplotlib.pyplot as plt
plt.style.use('ggplot')   # any style.

from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import os

import model_utils
from data import construct_dataset

from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

from PIL import Image

if __name__ == '__main__':

    IMAGES_PATH = os.path.join(os.getcwd(), "data", "processed", "mnt", "eme2_square_imgs")
    ANNOTS_FOLDER = os.path.join(os.getcwd(), "data", "raw", "labels")

    # hyperparameters for models
    INIT_LR = 1e-4
    NUM_EPOCHS = 1
    BATCH_SIZE = 32

    print(">>> Loading dataset...")
    data, targets, filenames = construct_dataset.initialise_dataset(ANNOTS_FOLDER, IMAGES_PATH)

    print(">>> Splitting dataset into train/test images and targets...")
    trainImages, testImages, trainTargets, testTargets, trainFilenames, testFilenames = construct_dataset.partition_dataset(data, targets, filenames)

    # test performance of different models
    for i in range(0, 1):

        print(">>> Preparing model...")
        # load and set up model
        model, keyword = model_utils.load_model(i)
        opt = Adam(lr=INIT_LR)
        model.compile(loss="mse", optimizer=opt)
        # print(model.summary())

        ##############################################
        # Callbacks - recording the intermediate training results which can be visualised on tensorboard
        subFolderLog = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboardPath = os.path.join(os.getcwd(), "models", subFolderLog)
        checkpointPath = os.path.join(os.getcwd(), "models" , subFolderLog)
        checkpointPath = checkpointPath + "/" + keyword + "-weights-improve-{epoch:02d}-{val_loss:02f}.hdf5"

        callbacks = [
            TensorBoard(log_dir=tensorboardPath, histogram_freq=0, write_graph=True, write_images=True),
            ModelCheckpoint(filepath=checkpointPath, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        ]

        # train the network for bounding box regression
        print("[INFO] training bounding box regressor...")
        H = model.fit(
        	trainImages, trainTargets,
        	validation_data=(testImages, testTargets),
        	batch_size=BATCH_SIZE,
        	epochs=NUM_EPOCHS,
            callbacks=callbacks,
        	verbose=1)

        # plot the model training history

        plt.figure()
        plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
        plt.title("Bounding Box Regression Loss on Training Set")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        PLOT_PATH = os.path.join(os.getcwd(), "src", "visualisation", keyword + "plot.png")
        plt.savefig(PLOT_PATH)
