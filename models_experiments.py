import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from keras import optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, InputLayer, BatchNormalization, Dropout


def mlp_model1():
    '''
    Simple neural network with activation function - relu.
    '''
    model = Sequential([
                        Flatten(input_shape=(48, 48, 1)),
                        Dense(256, activation='relu', use_bias = True),
                        Dense(128, activation = 'relu', use_bias = True),
                        Dense(7, use_bias = True)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def mlp_model2():
    '''
    Simple neural network with dropout and activation function relu.
    '''
    model = Sequential([
                        Flatten(input_shape=(48, 48, 1)),
                        Dense(256, activation='relu', use_bias=True),
                        Dropout(0.1),
                        Dense(128, activation='relu', use_bias=True),
                        Dropout(0.1),
                        Dense(7, use_bias=True)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def mlp_model3():
    '''
    Simple neural network with more hidden layers, dropout and tanh as activation function.
    '''
    model = Sequential([
                        Flatten(input_shape=(48, 48, 1)),
                        Dense(512, activation='tanh', use_bias=True),
                        Dropout(0.4),
                        Dense(256, activation='tanh', use_bias=True),
                        Dropout(0.3),
                        Dense(128, activation='tanh', use_bias=True),
                        Dropout(0.2),
                        Dense(7, use_bias=True)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def mlp_model4():
    '''
    Simple neural network with more hidden layers of bigger size, dropout and tanh as activation function.
    '''
    model = Sequential([
                         Flatten(input_shape = (48, 48, 1)),
                         Dense(1000, activation = 'tanh', use_bias = True),
                         Dropout(0.3),
                         Dense(750, activation = 'tanh', use_bias = True),
                         Dropout(0.2),
                         Dense(300, activation = 'tanh', use_bias = True),
                         Dropout(0.3),
                         Dense(7, use_bias = True)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def mlp_model5():
    '''
    Simple neural network with relu and more hidden layers of bigger size.
    '''
    model = Sequential([
                        Flatten(input_shape=(48, 48, 1)),
                        Dense(2000, activation='relu', use_bias=True),
                        Dense(1000, activation='relu', use_bias=True),
                        Dense(500, activation='relu', use_bias=True),
                        Dense(250, activation='relu', use_bias=True),
                        Dense(7, use_bias=True)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def cnn_model1():
    '''
    Example of CNN.
    '''
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(7)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model


def cnn_model2():
    '''
    Example of CNN with additional conv layer.
    '''
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(7)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model


def cnn_model3():
    '''
    Example of CNN with additional conv layer and dropout.
    '''
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model


def cnn_model4():
    '''
    More conv layers and dropout at the end of the network.
    '''
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        # BatchNormalization(),
        Dropout(0.5),
        Dense(7)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model


def cnn_model5():
    '''
    Added BatchNormalization do CNN model 4.
    '''
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model
