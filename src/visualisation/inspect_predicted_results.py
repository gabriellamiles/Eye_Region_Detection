# Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory

import os
import cv2

import pandas as pd

def load_predictions(predictionsFolder):
    """
    Function takes in (string) which is a filepath to the folder containing the ML-predicted results
    and returns these results as a list of dataframes, where each dataframe is the predictions of an
    individual trial.
    """

    all_predictions = [os.path.join(predictionsFolder, i) for i in os.listdir(predictionsFolder)]

    listOfPredictions = []

    for path in all_predictions:

        # ignore everything that isn't a csv file
        if not path[-3:] == "csv":
            continue

        # tmp_df = pd.read_csv(path)[['filename', 'startLX', 'startLY', 'endLX', 'endLY', 'startRX', 'startRY', 'endRX', 'endRY']]
        tmp_df = pd.read_csv(path)[['filename', 'LE_left', 'LE_top', 'LE_right', 'LE_bottom', 'RE_left', 'RE_top', 'RE_right', 'RE_bottom']]
        listOfPredictions.append(tmp_df)

    return listOfPredictions

def reorder_predictions(predictions, saveFolder):
    """
    Function receives list of dataframes, and orders the dataframe based on the first column.
    Returns list of ordered dataframes.
    """

    for trial in predictions:

        participant = str(trial.iloc[0,0]).split("/")[0]
        trialNum = str(trial.iloc[0,0]).split("/")[1]

        for i in range(trial.shape[0]):

            print("Looking for: " + str(i))

            current_min = int(trial.iloc[i,0].split("-")[-1][:-4]) # retrieve img number
            min_index = i # min_index_initialiser

            if current_min == i:
                continue
            else:
                # iterate over remaining unsorted items
                for j in range(i, trial.shape[0]):

                    # check if jth value is less than current min
                    checknum = int(trial.iloc[j,0].split("-")[-1][:-4]) # retrieve img number

                    if checknum == i:
                        min_index = j
                        swap(trial, i, min_index)
                        break

        trial.to_csv(os.path.join(saveFolder, participant + "_" + trialNum + ".csv"))

    return predictions

def swap(trial, a, b):
    """ Function swaps rows a and b in pandas dataframe (trial) """

    tmp = trial.iloc[a,:].copy()
    trial.iloc[a, :] = trial.iloc[b,:]
    trial.iloc[b,:] = tmp

def display_results_from_predictions(predictions, imgFolder, saveFolder):

    for trial in predictions:

        # initialise video
        videoTitle = trial.iloc[0, 0].split("/")[0] + "_" + trial.iloc[0, 0].split("/")[1] + ".avi"
        print(videoTitle)
        out = cv2.VideoWriter(os.path.join(saveFolder, videoTitle), cv2.VideoWriter_fourcc('M','J','P','G'), 160, (960,960))


        for idx in range(trial.shape[0]):
            print("Image: " + str(idx) + "/" + str(trial.shape[0]))
            # find corresponding images
            imgPath = os.path.join(imgFolder, trial.iloc[idx, 0])
            im = cv2.imread(imgPath)

            # plot LEFT EYE prediction on corresponding imgs
            start_point = (int(trial.iloc[idx, 1]), int(trial.iloc[idx, 2]))
            end_point = (int(trial.iloc[idx, 3]), int(trial.iloc[idx, 4]))
            cv2.rectangle(im, start_point, end_point, (255,0,0), 4) # blue

            # plot RIGHT EYE prediction on corresponding imgs
            start_point = (int(trial.iloc[idx, 5]), int(trial.iloc[idx, 6]))
            end_point = (int(trial.iloc[idx, 8]), int(trial.iloc[idx, 8]))
            cv2.rectangle(im, start_point, end_point, (0,255,0) , 4) # green

            # save video for inspection later
            out.write(im)

        out.release()


if __name__ == '__main__':

    # initialise key paths
    rootFolder = os.getcwd()
    imgFolder = os.path.join(rootFolder, "data", "processed", "mnt", "eme2_square_imgs")
    predictionsFolder = os.path.join(rootFolder, "models", "20230106_114326", "predictions")
    reorderedPredictionsFolder = os.path.join(rootFolder, "models", "20230106_114326", "predictions_reordered")
    videoFolder = os.path.join(rootFolder, "src", "visualisation", "resultsInspectionVideos")

    # acquire output of ML models
    predictions = load_predictions(predictionsFolder)
    print(len(predictions), len(predictions[0]))

    # sort this into consecutive order
    # predictions = reorder_predictions(predictions, reorderedPredictionsFolder)

    display_results_from_predictions(predictions, imgFolder, videoFolder)
