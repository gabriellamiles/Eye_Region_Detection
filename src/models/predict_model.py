"Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory"

import os
import time

import pandas as pd
import numpy as np

from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from PIL import Image

import model_utils
from data import construct_dataset


def predict_unseen_images(model, imgFilepaths, unseenImages, model_directory):

    count = 0
    # for each participant trial
    for set in unseenImages:

        if len(imgFilepaths[count]) == 0:
            print("Filenames count not be retrieved from directory...")
            count += 1
            continue

        predicted_filenames = []
        tmp_predictions = []
        filename_count = 0

        # extract key information for saving data from filename string
        participant = imgFilepaths[count][0].split("eme2_square_imgs/")[-1].split("/")[0]
        trial = imgFilepaths[count][0].split(participant+"/")[-1].split("/")[0]

        # for each image in a participant trial
        for image in set:

            preds = model.predict(image, batch_size=1)
            preds = preds[0]*960

            (startLX, startLY, endLX, endLY, startRX, startRY, endRX, endRY) = preds

            tmp_predictions.append([imgFilepaths[count][filename_count].split("eme2_square_imgs/")[-1], startLX, startLY, endLX, endLY, startRX, startRY, endRX, endRY])

            filename_count +=1

        # save predictions for individual participant trials separately
        trialResults = pd.DataFrame(tmp_predictions, columns = ['filename', 'LE_left', 'LE_top', 'LE_right', 'LE_bottom', 'RE_left', 'RE_top', 'RE_right', 'RE_bottom'])
        saveUnder = os.path.join(os.getcwd(), "models", model_directory, participant + "_" + trial + ".csv")
        trialResults.to_csv(saveUnder)

        count += 1

def test_on_testset(model, testFilenames, testImages, testTargets, ImgFolder, model_directory):

    count = 0 # use to correspond filenames, targets with predictions

    predictions = []

    for image in testImages:

        image = np.array([image])

        # make bounding box predictions on the input image
        preds = model.predict([image], batch_size=1)
        (startLX, startLY, endLX, endLY, startRX, startRY, endRX, endRY) = preds[0]

        # scale the predicted bounding box coordinates based on the image
        # dimensions
        im = Image.open(os.path.join(ImgFolder, testFilenames['filename'][count]))
        (w, h) = im.size
        startLX = int(startLX * w)
        startLY = int(startLY * h)
        endLX = int(endLX * w)
        endLY = int(endLY * h)
        startRX = int(startRX * w)
        startRY = int(startRY * h)
        endRX = int(endRX * w)
        endRY = int(endRY * h)

        predictions.append([testFilenames['filename'][count], startLX, startLY, endLX, endLY, startRX, startRY, endRX, endRY])

        count += 1

    # save predictions
    df = pd.DataFrame(predictions, columns = ['filename', 'LE_left', 'LE_top', 'LE_right', 'LE_bottom', 'RE_left', 'RE_top', 'RE_right', 'RE_bottom'])
    predictionSavePath = os.path.join(os.getcwd(), "models", model_directory, "test_set_predictions.csv")
    df.to_csv(predictionSavePath)

if __name__ == '__main__':

    UNSEEN = 1 # 0 for test data, 1 for unseen (all images in dataset)
    ImgFolder = os.path.join(os.getcwd(), "data", "processed", "mnt", "eme2_square_imgs")

    # load test set to evaluate accuracy
    establishedTestSet = os.path.join(os.getcwd(), "data", "test.csv")
    testFilenames, testImages, testTargets = construct_dataset.load_images_to_test(establishedTestSet, ImgFolder)

    # load unseen images for prediction
    unseen_filepaths, unseen_images = construct_dataset.load_unseen_images(ImgFolder)

    # load and compile trained model
    model_directory = "20230106_114326"
    model_file = "vgg-weights-improve-01-0.038387.hdf5"
    model_filepath = os.path.join(os.getcwd(), "models", model_directory, model_file)
    model, keyword = model_utils.load_model(0)
    model.load_weights(model_filepath)

    # compile, and key parameters for compile function
    INIT_LR = 1e-4
    opt = Adam(lr=INIT_LR)
    model.compile(loss="mse", optimizer=opt)

    if UNSEEN:
        print("[INFO] making predictions on unseen data...")
        predict_unseen_images(model, unseen_filepaths, unseen_images, model_directory)
    else:
        print("[INFO] evaulating model performance on test set...")
        test_on_testset(model, testFilenames, testImages, testTargets, ImgFolder, model_directory)
