import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# dataset path
dataset_path = "dataset/processed"

# image size
IMG_SIZE = 128
BATCH_SIZE = 32

# data generator (train + validation split)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    interpolation="bilinear"
)

# validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# number of classes
num_classes = train_data.num_classes

print("Number of classes:", num_classes)

# CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# show model architecture
model.summary()

# train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# create models folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# save trained model
model.save("models/rice_classifier.h5")

print("Model training completed and saved!")