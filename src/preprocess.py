import cv2
import numpy as np
import os

# dataset location
dataset_path = "dataset/Rice_Image_Dataset"
processed_path = "dataset/processed"

# create processed folder if it doesn't exist
if not os.path.exists(processed_path):
    os.makedirs(processed_path)

# image size for CNN
IMG_SIZE = 128

# get only folder names (ignore txt files)
classes = [
    c for c in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, c))
]

print("Rice Classes:", classes)

for rice_class in classes:

    class_path = os.path.join(dataset_path, rice_class)
    save_class_path = os.path.join(processed_path, rice_class)

    if not os.path.exists(save_class_path):
        os.makedirs(save_class_path)

    print(f"Processing {rice_class}...")

    for img_name in os.listdir(class_path):

        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # remove noise
        img = cv2.GaussianBlur(img, (5,5), 0)

        # normalize
        img = img / 255.0

        save_path = os.path.join(save_class_path, img_name)

        # convert back to uint8 before saving
        cv2.imwrite(save_path, (img * 255).astype("uint8"))

print("Preprocessing completed successfully!")