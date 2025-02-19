import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'TrainingImage'

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to get images and labels
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for imagePath in image_paths:
        # Convert image to grayscale
        img = Image.open(imagePath).convert('L')
        img_numpy = np.array(img, 'uint8')
        
        # Extract the user ID from the image file name
        id = int(os.path.split(imagePath)[-1].split('.')[1])
        
        # Detect the face in the image
        faces_detected = face_cascade.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in faces_detected:
            faces.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

        print(f"Processed {imagePath}, Faces Detected: {len(faces_detected)}")

    return faces, ids

# Training the recognizer
print("Starting training...")
faces, ids = get_images_and_labels(path)

if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    recognizer.write("TrainingImageLabel/Trainner.yml")
    print("Model trained and saved successfully.")
else:
    print("No faces found. Training aborted.")
