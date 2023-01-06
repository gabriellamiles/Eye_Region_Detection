import os

import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

from PIL import Image

def initialise_dataset(ANNOTS_FOLDER, IMAGES_PATH):


    # load all labels
    ANNOTS_PATHS = [os.path.join(ANNOTS_FOLDER, i ) for i in os.listdir(ANNOTS_FOLDER)]

    df_all = pd.read_csv(ANNOTS_PATHS[0])[["filename", "lx_top", "ly_top", "lx_bottom", "ly_bottom", "rx_top", "ry_top", "rx_bottom", "ry_bottom"]]

    for i in range(1, len(ANNOTS_PATHS)):
        tmp_df = pd.read_csv(ANNOTS_PATHS[i])[["filename", "lx_top", "ly_top", "lx_bottom", "ly_bottom", "rx_top", "ry_top", "rx_bottom", "ry_bottom"]]
        df_all = pd.concat([df_all, tmp_df])

    # remove any points that are (1, 1, 1, 1, 1, 1, 1, 1) across the board - indicating an error during labelling, or images where two eyes are not visible
    df_all = df_all[df_all.lx_top != 1]

    data, targets, filenames = [], [], [] # intiliase empty lists

    for row in range(df_all.shape[0]):
        imagePath = os.path.join(IMAGES_PATH, df_all.iloc[row, 0])
        # image = cv2.imread(imagePath)
        im = Image.open(imagePath)
        (w, h) = im.size

        # divide by width/height to scale coordinates to range [0, 1]
        lx_top = (float(df_all.iloc[row, 1])) / w
        ly_top = (float(df_all.iloc[row, 2])) / h
        lx_bottom = (float(df_all.iloc[row, 3])) / w
        ly_bottom = (float(df_all.iloc[row, 4])) / h

        rx_top = (float(df_all.iloc[row, 5])) / w
        ry_top = (float(df_all.iloc[row, 6])) / h
        rx_bottom = (float(df_all.iloc[row, 7])) / w
        ry_bottom = (float(df_all.iloc[row, 8])) / h

        image = load_img(imagePath, target_size=(224,224))
        image = img_to_array(image)

        filenames.append(df_all.iloc[row, 0])
        targets.append((lx_top, ly_top, lx_bottom, ly_bottom, rx_top, ry_top, rx_bottom, ry_bottom)) # just left eye bounding box for now
        data.append(image)

    return data, targets, filenames

def partition_dataset(data, targets, filenames):

    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")

    # partition the data into training and testing splits using 90% of the data for
    # training and remaining 10% for testing

    split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainTargets, testTargets) = split[2:4]
    (trainFilenames, testFilenames) = split[4:]

    filenamesToSave = pd.Series(testFilenames, name="filename")
    testTargetsToSave = pd.DataFrame(testTargets, columns = ["lx_top", "ly_top", "lx_bottom", "ly_bottom", "rx_top", "ry_top", "rx_bottom", "ry_bottom"])*960 # multiplied by size of image
    testDataToSave = pd.concat([filenamesToSave, testTargetsToSave], axis=1)
    testDataToSave.to_csv(os.path.join(os.getcwd(), "data", "test.csv"))

    return trainImages, testImages, trainTargets, testTargets, trainFilenames, testFilenames

def load_images_to_test(filepath, ImgFolder):

    df = pd.read_csv(filepath)[["filename"]]

    testImages = []
    testTargets = 0 # update this to be correct.

    for row in range(df.shape[0]):

        imagePath = os.path.join(ImgFolder, df["filename"][row])
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        testImages.append(image)

    return df, testImages, testTargets

def load_unseen_images(unseenDataFolder):

    print("[INFO] Loading unseen images, large dataset: may take some time...")

    # get path to directories that contain the images
    participantDirectories = [os.path.join(unseenDataFolder, i) for i in os.listdir(unseenDataFolder)]
    imgDirectories = []
    for participantFolder in participantDirectories:
        for testFolder in os.listdir(participantFolder):
                imgDirectories.append(os.path.join(participantFolder, testFolder))

    # obtain all img filepaths stored like [[filepaths for run 1], [filepaths for run 2], ...]
    imgFilepaths = []
    for imgDirectory in imgDirectories:
        tmp_filepaths = []

        for img_filepath in os.listdir(imgDirectory):
            tmp_filepaths.append(os.path.join(imgDirectory, img_filepath))

        imgFilepaths.append(tmp_filepaths)

    # load images
    unseenImages = []
    for set in imgFilepaths:
        tmp_img_storage = []

        for img_filepath in set:
            image = load_img(img_filepath, target_size=(224,224))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            tmp_img_storage.append(image)


        unseenImages.append(tmp_img_storage)

    return imgFilepaths, unseenImages
