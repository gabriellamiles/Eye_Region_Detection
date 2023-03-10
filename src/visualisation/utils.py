import os
import cv2

import pandas as pd

def load_files(list_of_filepaths, column_key):
    """
    Function takes in (list) which is a list of filepaths to the folder containing the ML-predicted results
    and returns these results as a list of dataframes, where each dataframe is the predictions of an
    individual trial.


    column_key: list of strings
    """

    all_df = []
    for path in list_of_filepaths:

        # ignore everything that isn't a csv file
        if not path[-3:] == "csv":
            continue

        # tmp_df = pd.read_csv(path)[['filename', 'startLX', 'startLY', 'endLX', 'endLY', 'startRX', 'startRY', 'endRX', 'endRY']]
        tmp_df = pd.read_csv(path)[column_key]
        all_df.append(tmp_df)

    return all_df

def display_results_from_predictions(listOfFiles, imgFolder, saveFolder):

    for trial in listOfFiles:

        # make sure in consecutive order
        print(trial.head())
        trial = trial.sort_values(by=['filename'])
        print(trial.head())

        # initialise video
        videoTitle = trial.iloc[0, 0].split("/")[0] + "_" + trial.iloc[0, 0].split("/")[1] + ".avi"
        print(videoTitle)
        out = cv2.VideoWriter(os.path.join(saveFolder, videoTitle), cv2.VideoWriter_fourcc('M','J','P','G'), 160, (960,960))


        # for idx in range(0, trial.shape[0]):
        for idx in range(0, 160*10):

            try:
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
                end_point = (int(trial.iloc[idx, 7]), int(trial.iloc[idx, 8]))
                cv2.rectangle(im, start_point, end_point, (0,255,0) , 4) # green

                # save video for inspection later
                out.write(im)
            except IndexError:
                break

        out.release()
