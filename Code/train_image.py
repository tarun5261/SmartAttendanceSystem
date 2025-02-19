import os
import cv2
import numpy as np
from threading import Thread

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    os.makedirs("TrainingImageLabel", exist_ok=True)

    # Function to get images and labels
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        if not imagePaths:
            print("No training images found in the 'TrainingImage' folder.")
            return [], []

        faces = []
        Ids = []

        for imagePath in imagePaths:
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            Id = int(imagePath.split(os.sep)[-1].split('.')[1])
            faces.append(img)
            Ids.append(Id)

        return faces, Ids

    # Get images and labels
    faces, Ids = getImagesAndLabels("TrainingImage")

    if not faces or not Ids:
        print("Error: Training data is empty. Please register and capture images first.")
        return

    # Train the recognizer
    Thread(target=recognizer.train, args=(faces, np.array(Ids))).start()
    recognizer.save("TrainingImageLabel/Trainner.yml")
    print("Model trained successfully.")
