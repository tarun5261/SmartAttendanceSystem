import datetime
import os
import time
import cv2
import pandas as pd


def recognize_attendence():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    df = pd.read_csv("StudentDetails/StudentDetails.csv", header=None)
    df.columns = ['Id', 'Name']
    df['Id'] = df['Id'].astype(str)

    col_names = ['Id', 'Name', 'Date']
    attendance = pd.DataFrame(columns=col_names)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    duration_limit = 30

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 55:
                str_id = str(int(Id))
                matching_rows = df[df['Id'] == str_id]
                name = matching_rows['Name'].iloc[0] if not matching_rows.empty else "Unknown"
                conf_str = f"  {100 - int(conf)}%"
                
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

                if attendance[(attendance['Id'] == str_id) & (attendance['Date'] == date)].empty:
                    attendance.loc[len(attendance)] = [str_id, name, date]
                    if name != "Unknown":
                        print(f"Marked attendance for {name} ({str_id})")

            else:
                name = "Unknown"
                str_id = "Unknown"
                conf_str = f"  {100 - int(conf)}%"

            text = f"{name} ({str_id})"
            cv2.putText(im, text, (x + 5, y - 5), font, 0.8, (0, 255, 0), 2)
            cv2.putText(im, conf_str, (x + 5, y + h - 5), font, 0.6, (255, 255, 255), 1)

        cv2.imshow('Attendance', im)

        if (time.time() - start_time > duration_limit) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    file_path = f"Attendance/Attendance_{timestamp}.csv"
    os.makedirs("Attendance", exist_ok=True)
    attendance.to_csv(file_path, index=False)

    cam.release()
    cv2.destroyAllWindows()