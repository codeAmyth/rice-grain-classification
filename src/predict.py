import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("models/rice_classifier.h5")

classes = ["Arborio","Basmati","Ipsala","Jasmine","Karacadag"]

IMG_SIZE = 128


def predict(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)

    print("Rice Type:",classes[index])
    print("Confidence:",np.max(prediction))


predict("dataset/Rice_Image_Dataset/Basmati/basmati (4).jpg")