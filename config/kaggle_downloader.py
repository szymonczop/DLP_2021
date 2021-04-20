import kaggle
import os

os.system("ls")
os.system("kaggle competitions download -c 'challenges-in-representation-learning-facial-expression-recognition-challenge'")
os.system("unzip challenges-in-representation-learning-facial-expression-recognition-challenge.zip")
os.system("rm challenges-in-representation-learning-facial-expression-recognition-challenge.zip")
os.system("rm example_submission.csv icml_face_data.csv test.csv train.csv")
os.system(" tar -xvf fer2013.tar.gz")
os.system("rm fer2013.tar.gz")