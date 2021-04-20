# DLP_2021
Deep Learning project lead by Rafał Nowak from Wroclaw University at Computer Science faculty.

Team: 

* [Marta Kałużna](https://github.com/mkaluzna)
* [Szymon Czop](https://github.com/szymonczop) 

In the first project we will focus on __image classification problem__. 

We will use [Kaggle's fer2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
which consists of 48x48 pixel grayscale images of faces of various emotions which are labeled. It was firstly published during the International Conference on Machine learning.

The task is to categorize each face based on the emotion to one of seven categories:

* 0 - Angry
* 1 - Disgust 
* 2 - Fear 
* 3 - Happy
* 4 - Sad
* 5 - Surprise 
* 6 - Neutral  

## Loading data 
In order to load data, download it into your local computer from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
and save it to your present directory. To open it and store in tree of catalogues use:

``` python data_loader.py --data.csv```

Where data.csv should be changed to however you changed the name of the dataset from Kaggle.
The default argument is "fer2013.csv" so if you left the name without changes there is no need to specify that argument.
After running this command we should get the following structure of folders.
We can find and inspect images that are stored both in the train and test subdirectory.

```
   .
    ├── ...
    ├── data                  # Folder with test and train data   
    │   ├── train             # Train examples 
    │   │      ├── Angry        # Folder with images labeled as Angry
    │   │      ├── Disgust      # Folder with images labeled as Disgust
    │   │      ├── Fear         .  
    │   │      ├── Happy        .
    │   │      ├── Neutral      .
    │   │      ├── Sad          .
    │   │      ├── Surprise     .
    │   │      
    │   │        
    │   ├── test            # Test examples 
    │        ├──Angry           .
    │        ├── ...            .
    │        ├── Surprise       .  
    └── ...
```
#### Numbers behind data
Now small description about amount of image data with respect to train/test partition and specific emotion 

Emotion | Train| Test
------------ | ------------- | -------------
Angry | 3995 | 958
Disgust | 436| 111
Fear | 4097 | 1024
Happy | 7215 | 1776
Sad | 4830 | 4953
Surprise | 3171 | 1247
Neutral |4965 | 1233