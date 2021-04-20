from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_classes = 7

Img_Height = 48
Img_width = 48

batch_size = 32

train_dir = "./data/train"
test_dir = "./data/test"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=60,
                                   shear_range=0.5,
                                   zoom_range=0.5,
                                   width_shift_range=0.5,
                                   height_shift_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    color_mode='grayscale',
                                                    target_size=(Img_Height, Img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  color_mode='grayscale',
                                                  target_size=(Img_Height, Img_width),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)