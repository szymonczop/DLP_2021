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

``` python data_loader.py --data data_name.csv```

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

#### Data augmentation
Augmentation of data is used in order to increase number of pictures on which given model will be trained.
File that is responsible for it we called ```data_augmentation.py``` there are applied some basic techiques such as 
rotation, zoom, with and height shift. For more details visit [documentation website](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator).
The functionality of this file is used inside of ```train.py```.

#### Models

In file ```models_experiments.py``` we added 10 different models to expeiments. Half of them are MLP and second half is 
based on CNN. The start from the smallest to the biggest in terms of parameters to tune. Of course
CNN networks has less of them by it's usage definition but they are also sorted in their own group. This file is also used in
```train.py``` but if we would like to add another model, this network should be placed in ```models_experiments.py```.

#### Training 
The most important function in the whole project, here all the magic happends, we paramtrize and trian model in this
very function. This function is made with ```argparse``` module and can be called like this 

```python --model <model_name>   ```

where ```<model_name>``` should be replaced with the name of the model that is defined in ```models_experments.py```

There are also other argument's that can be defined

```--logs``` -> directory where logs from training proces should be stored (default: 'logs/model' )

```--checkpoint```-> directory where best weights will be stored (default: 'model.h5')

```--lr``` -> learning rate of the training process (default: 0.01)

```--num_epochs``` ->  number of epochs made during training process (default: 50)

```--early_stop_patience``` -> Indicates after how many epchos without accuracy improvement our training process in terminated 
(default: 5 )

```--reduce_lr_patience``` -> Indicates after how many steps without improvement learning rate is decreased (default 3)

```--params``` -> as value to this argument we should pass ```.json``` file with all arguments that are listed above. We
should use ```--params ``` we have to define all parameters without ```--model``, this still has to be done manually for the 
convinence of the user

Example architecture of ```.json``` file

```
{
    "num_epochs" : 50,
    "early_stop_patience" : 10
    "reduce_lr_patience": 5
    "logs" : "logs/name_space
    "checkpoint": "example_name.h5" 
}
```

Having this set we are able to check model fast, play with hyperparameters and have acces to best weights as well as
monitor training process on tensorboard.

### Example usage
In folder ```notebooks``` you can find example downloading, training and tuning process with usage of function in this repo only !
 
 ### Aditional files
 
 Some helpfull additional files can be found in ```config folder```, there is another example of ```.json``` file, 
 ```kaggle_downloader.py``` that allow us to download emotion dataset directly from Kaggle with one command and
 ```kaggle_to_colab.txt``` with instruction how to do the same in ```Colab``` environment. In this file also information about
 further step of locall downloading process can be found. 