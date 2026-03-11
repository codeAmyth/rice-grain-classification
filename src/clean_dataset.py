from PIL import Image
import os

dataset_path = "dataset/processed"

removed = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)

        try:
            img = Image.open(path)
            img.verify()
        except:
            print("Removing corrupted file:", path)
            os.remove(path)
            removed += 1

print("Total removed files:", removed)