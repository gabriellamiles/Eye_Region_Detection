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

def progress_tracker(stagepoints, count):

    if count == stagepoints[0]:
        print(" >> Starting predictions... ")
    elif count == stagepoints[1]:
        print(" ************************************ Predictions 25% complete... ")
    elif count == stagepoints[2]:
        print("************************************* Predictions 55% complete... ")
    elif count == stagepoints[3]:
        print("************************************* Predictions 75% complete... ")
    else:
        pass

def predict_unseen_images(model, imgFilepaths):

    for set in imgFilepaths:
        # print(imgFilepaths)
        predictions = []
        print(len(set))

        stage_points = [0, int(len(set)/4), int(len(set)/2), int(3*len(set)/4)]
        count = 0

        participant = "number"
        try:
            participant = set[0].split("eme2_square_imgs/")[-1].split("/")[0]
        except IndexError:
            continue

        trial = set[0].split(participant+"/")[-1].split("/")[0]

        if participant == "003" or participant == "029" or participant == "071":
            print("Participant " + str(participant) + " already completed, skip to next participant.")
            continue

        for img_filepath in set:
            print(img_filepath)

            progress_tracker(stage_points, count)
            image = load_img(img_filepath, target_size=(224,224))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            # make bounding box predictions on the input image
            preds = model.predict(image)[0]
            (startLX, startLY, endLX, endLY, startRX, startRY, endRX, endRY) = preds

            im = Image.open(img_filepath)

            (w, h) = im.size
            # scale the predicted bounding box coordinates based on the image
            # dimensions
            startLX = int(startLX * w)
            startLY = int(startLY * h)
            endLX = int(endLX * w)
            endLY = int(endLY * h)
            startRX = int(startRX * w)
            startRY = int(startRY * h)
            endRX = int(endRX * w)
            endRY = int(endRY * h)

            predictions.append([img_filepath.split("eme2_square_imgs/")[-1], startLX, startLY, endLX, endLY, startRX, startRY, endRX, endRY])
            count += 1

        df = pd.DataFrame(predictions, columns = ['filename', 'LE_left', 'LE_top', 'LE_right', 'LE_bottom', 'RE_left', 'RE_top', 'RE_right', 'RE_bottom'])
        df.to_csv("./output/eye_region/" + participant + "_" + trial + "_eye_region_predictions.csv")

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

    UNSEEN = 0 # 0 for test data, 1 for unseen (all images in dataset)
    ImgFolder = os.path.join(os.getcwd(), "data", "processed", "mnt", "eme2_square_imgs")

    # load test set to evaluate accuracy
    establishedTestSet = os.path.join(os.getcwd(), "data", "test.csv")
    testFilenames, testImages, testTargets = construct_dataset.load_images_to_test(establishedTestSet, ImgFolder)

    # load unseen images for prediction
    # unseen_filepaths = construct_dataset.load_unseen_images(ImgFolder)

    # load and compile trained model
    model_directory = "20230106_114326"
    model_file = "vgg-weights-improve-01-0.038387.hdf5"
    model_filepath = os.path.join(os.getcwd(), "models", model_directory, model_file)
    model, keyword = model_utils.load_model(0)
    model.load_weights(model_filepath)

    # model = load_model(model_filepath, compile=False) # set compile to False otherwise doesn't load
    # compile, and key parameters for compile function
    INIT_LR = 1e-4
    opt = Adam(lr=INIT_LR)
    model.compile(loss="mse", optimizer=opt)

    if UNSEEN:
        print("[INFO] making predictions on unseen data...")
        predict_unseen_images(model, unseen_filepaths)
    else:
        print("[INFO] evaulating model performance on test set...")

        test_on_testset(model, testFilenames, testImages, testTargets, ImgFolder, model_directory)
