import cv2
import os
import csv

student_details_path = os.path.join("StudentDetails", "StudentDetails.csv")

# Ensure directories exist
os.makedirs("StudentDetails", exist_ok=True)
os.makedirs("TrainingImage", exist_ok=True)

# Ensure the CSV file exists
if not os.path.exists(student_details_path):
    with open(student_details_path, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["Id", "Name"])

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

    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Unable to access the camera.")
            return

        detector = cv2.CascadeClassifier(os.path.join("Code", "haarcascade_default.xml"))
        if detector.empty():
            print("Error: Failed to load Haar cascade file. Check the file path.")
            return
        else:
            print("Haar cascade file loaded successfully.")

        sampleNum = 0

        while True:
            ret, img = cam.read()
            if not ret:
                print("Error: Unable to capture frame from the camera.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            # Ensure we detect faces
            if len(faces) == 0:
                print("No faces detected, please try again.")
                continue

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
                sampleNum += 1
                # Save each face image in the TrainingImage folder
                cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
                cv2.imshow('frame', img)

            if cv2.waitKey(150) & 0xFF == ord('q'):
                break
            elif sampleNum >= 150:
                break

        cam.release()
        cv2.destroyAllWindows()

        if sampleNum < 150:
            print(f"Error: Not enough images captured. Captured {sampleNum} images.")
            return

        print(f"Images Saved for ID: {Id}, Name: {name}")

        row = [Id, name]
        try:
            with open(student_details_path, 'a+', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            print("Student registered.")
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    else:
        if is_number(Id):
            print("Error: Enter a valid alphabetical name.")
        if name.isalpha():
            print("Error: Enter a numeric ID.")

