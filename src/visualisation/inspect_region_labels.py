# Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory

import os
import cv2
import utils

import pandas as pd

if __name__ == '__main__':
    rootFolder = os.getcwd()
    labelFolder = os.path.join(rootFolder, "data", "raw", "labels")
    imgFolder = os.path.join(rootFolder, "data", "processed", "mnt", "eme2_square_imgs")
    videoFolder = os.path.join(rootFolder, "src", "visualisation", "labelInspectionVideos")

    column_key = ['filename', 'lx_top', 'ly_top', 'lx_bottom', 'ly_bottom', 'rx_top', 'ry_top', 'rx_bottom', 'ry_bottom']
    all_labels = utils.load_files(labelFolder, column_key)
    utils.display_results_from_predictions(all_labels, imgFolder, videoFolder)
