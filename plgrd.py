import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

os.getcwd()

data = pd.read_csv("/Users/szymonczop/Desktop/SemestrIV/Nowak_projekt/DLP_2021/fer2013.csv")
data.shape
data.head()
data.columns  # Index(['emotion', 'pixels', 'Usage'], dtype='object')

data.emotion.unique()  # array([0, 2, 4, 6, 3, 5, 1])
data.Usage.unique()  # ['Training', 'PublicTest', 'PrivateTest']
data.Usage.value_counts()

data[data.Usage.isin(["PublicTest", "PrivateTest"])].shape[0] # 7178
data[data.Usage.isin(["Training"])].shape[0] # 28709

data[(data.emotion == 0) & (data.Usage.isin(["PublicTest", "PrivateTest"]))].shape[0]  # 958
data[(data.emotion == 1) & (data.Usage.isin(["PublicTest", "PrivateTest"]))].shape[0]  # 111
data[(data.emotion == 2) & (data.Usage.isin(["PublicTest", "PrivateTest"]))].shape[0]  # 1024
data[(data.emotion == 3) & (data.Usage.isin(["PublicTest", "PrivateTest"]))].shape[0]  # 1776
data[(data.emotion == 4) & (data.Usage.isin(["PublicTest", "PrivateTest"]))].shape[0]  # 4953
data[(data.emotion == 5) & (data.Usage.isin(["PublicTest", "PrivateTest"]))].shape[0]  # 1247
data[(data.emotion == 6) & (data.Usage.isin(["PublicTest", "PrivateTest"]))].shape[0]  # 1233

data[(data.emotion == 0) & (data.Usage.isin(["Training"]))].shape[0]  # 3995
data[(data.emotion == 1) & (data.Usage.isin(["Training"]))].shape[0]  # 436
data[(data.emotion == 2) & (data.Usage.isin(["Training"]))].shape[0]  # 4097
data[(data.emotion == 3) & (data.Usage.isin(["Training"]))].shape[0]  # 7215
data[(data.emotion == 4) & (data.Usage.isin(["Training"]))].shape[0]  # 4830
data[(data.emotion == 5) & (data.Usage.isin(["Training"]))].shape[0]  # 3171
data[(data.emotion == 6) & (data.Usage.isin(["Training"]))].shape[0]  # 4965






data.loc[0,"pixels"] # to jest wgl w formacie string masakra jakaś

float(data.loc[0,"pixels"].split(" "))

img_pixels =list(map(float, data.loc[0,"pixels"].split(" ")))
img_pixels = np.array(img_pixels).reshape(48,48)
plt.imshow(img_pixels)
cv2.imwrite("/Users/szymonczop/Desktop/test_image.jpg", img_pixels)