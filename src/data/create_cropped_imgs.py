"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"
""" General purpose of this code is to take bounding box predictions, and to create crops from these predictions. """

import os
import cv2
import pandas as pd

def retrieve_csv_filepaths_from_directory(directory):

    csv_filepaths = []

    for i in os.listdir(directory):
        if i[-4:]==".csv":
            tmp = os.path.join(directory, i)
            csv_filepaths.append(tmp)

    return csv_filepaths

def filepaths_not_yet_cropped(list_of_filepaths, directory):
    """Function checks what filepaths "appear" in directory, and returns list of those that don't exist. """

    left_eye_missing, right_eye_missing = [], []
        
    for file in list_of_filepaths: 

        participant_info = file.rsplit("/", 1)[1][:-4]
        participant_num = participant_info.split("_")[0]
        trial = participant_info.split("_")[1]
        
        check_path = os.path.join(directory, "left_eye", participant_num, trial)

        # check if folder exists for left eye
        if os.path.exists(check_path):
            continue
        else:
            left_eye_missing.append(file)
    
        check_path = check_path.replace("left", "right")
        # check if folder exists for right eye
        if os.path.exists(check_path):
            continue
        else:
            right_eye_missing.append(file)

    return left_eye_missing, right_eye_missing

def save_crops(list_of_filepaths, output_folder, img_folder):

    print("Here")
    for filepath in list_of_filepaths:
    
        # load dataframe
        tmp_df = pd.read_csv(filepath)[["filename",	"LE_left", "LE_top", "LE_right", "LE_bottom", "RE_left", "RE_top", "RE_right", "RE_bottom"]]
        

        participant_info = tmp_df.iloc[0,0]
        participant_num = participant_info.split("/")[0]
        trial = participant_info.split("/")[1]
         # create necessary folders if they don't exists
        save_folder = os.path.join(output_folder, "left_eye", participant_num, trial)
        print(save_folder)
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(save_folder.replace("left", "right")):
            os.makedirs(save_folder.replace("left", "right"))

        for row in range(tmp_df.shape[0]):

            # load image
            img_filepath = os.path.join(img_folder, tmp_df.iloc[row, 0])

            # important points
            LE_left = int(tmp_df.iloc[row, 1])
            LE_top = int(tmp_df.iloc[row, 2])
            LE_right = int(tmp_df.iloc[row, 3])
            LE_bottom = int(tmp_df.iloc[row, 4])

            RE_left = int(tmp_df.iloc[row, 5])
            RE_top = int(tmp_df.iloc[row, 6])
            RE_right = int(tmp_df.iloc[row, 7])
            RE_bottom = int(tmp_df.iloc[row, 8])

            # read full image and create crops
            im = cv2.imread(img_filepath)
            left_cropped_im = im[LE_top:LE_bottom, LE_left:LE_right]
            right_cropped_im = im[RE_top:RE_bottom, RE_left:RE_right]

            # save cropped images
            left_save_path = os.path.join(output_folder, "left_eye", tmp_df.iloc[row, 0])
            right_save_path = left_save_path.replace("left", "right")
            cv2.imwrite(left_save_path, left_cropped_im)
            cv2.imwrite(right_save_path, right_cropped_im)

            cv2.destroyAllWindows()


    

if __name__ == '__main__':
    
    # initialise key filepaths
    root_folder = os.getcwd()
    img_folder = os.path.join(root_folder, "data", "processed", "mnt", "eme2_square_imgs")
    predictions_folder = os.path.join(root_folder, "models", "20230109_142938", "predictions")
    output_folder = os.path.join(root_folder, "data", "processed", "mnt", "cropped_eye_imgs")

    # get data
    prediction_filepaths = retrieve_csv_filepaths_from_directory(predictions_folder)

    # compare prediction filepaths to existing cropped img folders
    left_missing, right_missing = filepaths_not_yet_cropped(prediction_filepaths, output_folder)
    
    # determine if left and right are missing exactly same ones
    left_missing = sorted(left_missing)
    right_missing = sorted(right_missing)

    if left_missing == right_missing:
        save_crops(left_missing, output_folder, img_folder)      
