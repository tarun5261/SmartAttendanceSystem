import csv
import cv2
import os

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def takeImages():
    Id = input("Enter Your Id: ")
    name = input("Enter Your Name: ")

    if(is_number(Id) and name.isalpha()):
        # Ensure that the "TestImage" directory exists
        if not os.path.exists('TestImage'):
            os.makedirs('TestImage')
        
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Unable to access the camera.")
            return

        harcascadePath = "haarcascade_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            if not ret:
                print("Error: Unable to capture frame from the camera.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
                sampleNum += 1
                # Save the face images
                img_path = f"TestImage/{name}.{Id}.{sampleNum}.jpg"
                cv2.imwrite(img_path, gray[y:y + h, x:x + w])
                print(f"Image saved: {img_path}")  # Debug: print image path
                cv2.imshow('frame', img)

            if cv2.waitKey(75) & 0xFF == ord('q'):
                break
            elif sampleNum >= 75:
                break

        cam.release()
        cv2.destroyAllWindows()

        if sampleNum > 0:
            print(f"Images Saved for ID: {Id}, Name: {name}")
        else:
            print("No images captured. Please try again.")
    else:
        if is_number(Id):
            print("Error: Enter a valid alphabetical name.")
        if name.isalpha():
            print("Error: Enter a numeric ID.")

takeImages()
