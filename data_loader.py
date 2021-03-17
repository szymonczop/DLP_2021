
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Image recognition project")
    parser.add_argument("--data", default = "fer2013.csv",
                        required=False, help ="Loading and splitting data from (default: %(default)s)")
    args = parser.parse_args()
    return args.data

data_df = parse_arguments()

#os.chdir("/Users/szymonczop/Desktop/SemestrIV/Nowak_projekt/DLP_2021/")
data = pd.read_csv(data_df)

if not os.path.exists("data"):
    os.mkdir("data")

os.chdir("./data")

if not os.path.exists("train"):
    os.mkdir("train")

if not os.path.exists("test"):
    os.mkdir("test")

#feelings = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
feelings_dict= {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

for k,v in feelings_dict.items():
    if not os.path.exists("train/" + v):
        os.makedirs("train/" + v, exist_ok=True)
    if not os.path.exists("test/" + v):
        os.makedirs("test/" + v, exist_ok=True)



data_train = data[data.Usage == "Training"]
#data_train.shape
data_test = data[data.Usage.isin(["PublicTest", "PrivateTest"])]
#data_test.shape
all_data = [data_train, data_test]

def assign_img_to_folder(df):
    test_or_train = df.Usage.iloc[0]
    if test_or_train == "Training":
        test_or_train = "train"
    else:
        test_or_train = "test"

    for row in tqdm(range(df.shape[0])):
        pix = df.iloc[row, df.columns.get_loc("pixels")]
        feel = df.iloc[row, df.columns.get_loc("emotion")]
        pixels = np.array([float(x) for x in pix.split(" ")])
        image = pixels.reshape(48, 48)
        global feelings_dict
        path = "./" + test_or_train + "/" + str(feelings_dict[feel]) + "/" + str(row) + ".jpg"
        cv2.imwrite(path, image)

for data in all_data:
    assign_img_to_folder(data)




