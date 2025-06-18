import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import easyocr
import sqlite3
from datetime import datetime
import re

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 models
face_model = YOLO('yolov8n-face.pt')  # face detection
plate_model = YOLO('license_plate_detector.pt')  # number plate detection

# Load FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# EasyOCR reader
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Transform for FaceNet input
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Compile regex for Indian number plate
plate_regex = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$')

# Load known face embeddings
known_embeddings = []
known_names = []

for file in os.listdir("known_faces"):
    if file.lower().endswith(('.jpg', '.png')):
        name = os.path.splitext(file)[0]
        img_path = os.path.join("known_faces", file)
        img = cv2.imread(img_path)
        results = face_model(img)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = img[y1:y2, x1:x2]
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(device)
                embedding = facenet(face_tensor).detach().cpu().numpy()[0]
                known_embeddings.append(embedding)
                known_names.append(name)

# === SQLite Database Setup ===
conn = sqlite3.connect('log.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS access_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        name TEXT,
        plate TEXT
    )
''')
conn.commit()

def log_to_db(name, plate):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO access_log (timestamp, name, plate) VALUES (?, ?, ?)',
                   (timestamp, name, plate))
    conn.commit()

# === Webcam Stream ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    recognized_name = "Unknown"
    plate_text = "N/A"

    # FACE DETECTION + RECOGNITION
    face_results = face_model(frame)
    for r in face_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            try:
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(device)
                embedding = facenet(face_tensor).detach().cpu().numpy()[0]

                # Compare with known
                sims = [cosine_similarity([embedding], [ke])[0][0] for ke in known_embeddings]
                max_sim = max(sims) if sims else 0
                name = "Unknown"
                if max_sim > 0.6:
                    name = known_names[np.argmax(sims)]
                recognized_name = name

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({max_sim:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except:
                continue

    # PLATE DETECTION + OCR
    plate_results = plate_model(frame)
    for r in plate_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

            try:
                result = reader.readtext(gray_plate)
                if result:
                    plate_text_raw = result[0][1]
                    plate_text_clean = plate_text_raw.replace(" ", "").upper()

                    if plate_regex.match(plate_text_clean):
                        plate_text = plate_text_clean
                        # Draw valid plate
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, plate_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        if recognized_name != "Unknown":
                            log_to_db(recognized_name, plate_text)
            except:
                continue

    # SHOW FINAL FRAME
    cv2.imshow("Face + Plate Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
conn.close()
cv2.destroyAllWindows()