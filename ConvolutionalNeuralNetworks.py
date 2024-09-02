import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rescales pixel from 255 to 1
    shear_range=0.2,
    # tilting of image at random
    zoom_range=0.2,
    # zoom image at random by 20%
    horizontal_flip=True
    # flip the image at random
)
training_set = train_datagen.flow_from_directory(
    'dataset/mytrain',
    # path of the file
    target_size=(64, 64),
    # resize the image to uniform size
    batch_size=32,
    # no of images processed together during training
    class_mode='binary'
    # categorical for multiple class nad binary for 2 classes
)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(
    'dataset/mytest',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=5)

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('test22.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
var = training_set.class_indices
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
