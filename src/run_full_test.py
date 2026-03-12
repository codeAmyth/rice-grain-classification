import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from extract_feature import extract_features

IMG_SIZE = 128

classes = ['Arborio','Basmati','Ipsala','Jasmine','Karacadag']

print("\nLoading trained CNN model...")
model = tf.keras.models.load_model("models/rice_classifier.h5")

test_folder = "dataset/test"

y_true = []
y_pred = []

print("\nRunning predictions on test dataset...\n")

for label in classes:

    folder_path = os.path.join(test_folder, label)

    for img_name in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img = img/255.0
        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img,verbose=0)

        pred_class = np.argmax(prediction)

        y_pred.append(pred_class)
        y_true.append(classes.index(label))

accuracy = np.mean(np.array(y_pred)==np.array(y_true))

print("\n==============================")
print("Model Accuracy:",round(accuracy*100,2),"%")
print("==============================")

cm = confusion_matrix(y_true,y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt='d',
xticklabels=classes,
yticklabels=classes,
cmap="Reds")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:\n")

print(classification_report(y_true,y_pred,target_names=classes))

print("\nExtracting physical features from sample image...")

sample_image = os.path.join(test_folder, classes[0], os.listdir(os.path.join(test_folder, classes[0]))[0])

extract_features(sample_image)