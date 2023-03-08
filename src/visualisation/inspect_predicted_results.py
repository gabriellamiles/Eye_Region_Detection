# Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory

import os
import utils

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




if __name__ == '__main__':

    # initialise key paths
    rootFolder = os.getcwd()
    imgFolder = os.path.join(rootFolder, "data", "processed", "mnt", "eme2_square_imgs")
    predictionsFolder = os.path.join(rootFolder, "models", "20230109_142938", "predictions")
    reorderedPredictionsFolder = os.path.join(rootFolder, "models", "20230106_114326", "predictions_reordered")
    videoFolder = os.path.join(rootFolder, "src", "visualisation", "resultsInspectionVideos")

    # acquire output of ML models
    column_key = ['filename', 'LE_left', 'LE_top', 'LE_right', 'LE_bottom', 'RE_left', 'RE_top', 'RE_right', 'RE_bottom']
    predictions = utils.load_files(predictionsFolder, column_key)
    print(len(predictions), len(predictions[0]))

    utils.display_results_from_predictions(predictions, imgFolder, videoFolder)
