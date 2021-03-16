import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/szymonczop/Desktop/SemestrIV/Nowak_projekt/DLP_2021/fer2013.csv")
data.head()
data.columns # Index(['emotion', 'pixels', 'Usage'], dtype='object')

data.emotion.unique() # array([0, 2, 4, 6, 3, 5, 1])
data.Usage.unique() # ['Training', 'PublicTest', 'PrivateTest']
data.loc[0,"pixels"] # to jest wgl w formacie string masakra jakaś

float(data.loc[0,"pixels"].split(" "))

img_pixels =list(map(float, data.loc[0,"pixels"].split(" ")))
img_pixels = np.array(img_pixels).reshape(48,48)
plt.imshow(img_pixels)
cv2.imwrite("/Users/szymonczop/Desktop/test_image.jpg", img_pixels)