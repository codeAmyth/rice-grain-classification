import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# load trained model
model = tf.keras.models.load_model("models/rice_classifier.h5")

classes = ["Arborio","Basmati","Ipsala","Jasmine","Karacadag"]

IMG_SIZE = 128

test_path = "dataset/test"

y_true = []
y_pred = []

for label, rice_class in enumerate(classes):

    folder = os.path.join(test_path, rice_class)

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder,img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img = img/255.0

        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img)

        pred_class = np.argmax(prediction)

        y_true.append(label)
        y_pred.append(pred_class)

# accuracy
accuracy = np.mean(np.array(y_true)==np.array(y_pred))
print("Test Accuracy:",accuracy)

# confusion matrix
cm = confusion_matrix(y_true,y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm,annot=True,fmt='d',
            xticklabels=classes,
            yticklabels=classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# classification report
print(classification_report(y_true,y_pred,target_names=classes))