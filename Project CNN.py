import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt


training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05)
#  raw pixel values;
# randomly increase or decrease the size;
# rotates the image between [-15,15] degrees;
# shift the image along its width by up to +/- 5%.
tdg = training_data_generator
training_iterator = tdg.flow_from_directory('C:\\Users\\xiaoyi\\pythonProject2\\Covid19_dataset\\train',
                                            color_mode='grayscale',
                                            class_mode='categorical',
                                            batch_size=16)

validation_data_generator = ImageDataGenerator(
    rescale=1. / 255)

vdg = validation_data_generator
validation_iterator = vdg.flow_from_directory('C:\\Users\\xiaoyi\\pythonProject2\\Covid19_dataset\\test',
                                              color_mode='grayscale',
                                              class_mode='categorical',
                                              batch_size=16)


model = Sequential()
model.add(layers.Input(shape=(256, 256, 1)))
model.add(layers.Conv2D(2, 5, padding='valid', strides=1, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5)))
model.add(layers.Conv2D(4, 3, padding='valid', strides=1, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(3, activation='softmax'))

print(model.summary())
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])
es = EarlyStopping(monitor='val_loss', patience=2, mode='min')

history = model.fit(training_iterator, steps_per_epoch=training_iterator.samples / 16,
                    epochs=20, validation_data=validation_iterator, validation_steps=validation_iterator.samples / 16)

# Do Matplotlib below
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
print(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])

ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
print(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

fig.tight_layout()
#  plt.show()
plt.savefig('images.png')

print('program is over')
