# face-number-plate-detection
# 🚘 Real-Time Face and Indian Number Plate Recognition

A Python-based real-time system for recognizing faces and reading **Indian vehicle number plates** using:

- 🧠 **YOLOv8** for object detection (face + number plate)
- 📸 **FaceNet** for face recognition
- 🔤 **EasyOCR** for reading plate text
- 💾 **SQLite** for logging detections
- ✅ Logs **only valid Indian plate formats**

---

## 📂 Project Structure

```

project/
├── license\_plate\_detector.pt       # YOLOv8 model for plates
├── yolov8n-face.pt                 # YOLOv8 model for faces
├── known\_faces/                   # Folder with known face images
│   ├── aditya.jpg
│   └── ...
├── main.py                         # Combined face + plate recognition script
├── log.db                          # SQLite DB (auto-created)
├── view\_logs.py                    # Script to export logs as CSV
├── access\_log.csv                  # CSV output of logs
└── README.md

````

---

## ✅ Features

- Real-time detection from webcam feed
- Face recognition using cosine similarity
- OCR with EasyOCR on cropped plates
- Regex-based validation for **Indian plate format**
- SQLite logging (name, plate, timestamp)
- Export logs to CSV

---

## 📦 Installation

### 1. Install Dependencies

```bash
pip install torch torchvision facenet-pytorch opencv-python easyocr ultralytics pillow scikit-learn
````

### 2. Download YOLOv8 Models

* [YOLOv8n-Face from Ultralytics](https://github.com/ultralytics/ultralytics/releases)
* [License Plate Detector - Custom](#note-license-plate-detector-model)

Place them in your project directory.

---

## 🚀 Usage

### 1. Prepare `known_faces/`

Add clear front-facing images to the `known_faces/` folder. Filenames will be used as the person's name:

```
known_faces/
├── user1.jpg      → name: "user1"
├── user2.png       → name: "user2"
```

### 2. Run Main Script

```bash
python main.py
```

### 3. Export Logs

```bash
python view_logs.py
```

> This creates `access_log.csv` with all valid detections.

---

## 🔤 Indian Plate Format (Regex)

Only logs plates matching this format:

```
^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$
```

Examples:

* ✅ `MH12AB1234`
* ❌ `MH 12 AB 1234` (OCR may fail to read without preprocessing)

---

## 🧠 Face Recognition

* Uses [FaceNet](https://github.com/timesler/facenet-pytorch) with VGGFace2 weights.
* Compares cosine similarity to known face embeddings.
* Recognition threshold: **0.6**

---

## 💾 Database Format

`log.db` will store:

| ID | Timestamp           | Name   | Plate      |
| -- | ------------------- | ------ | ---------- |
| 1  | 2025-06-14 10:32:05 | User   | MH12AB1234 |

---

## 📤 Future Improvements

* Flask GUI / Dashboard
* Duplicate detection timeout
* Email/SMS alert integration
* Upload logs to cloud (Firebase/Google Sheets)

---

## 🛠️ Note: License Plate Detector Model

The YOLOv8-based `license_plate_detector.pt` is a **custom-trained model**. If you don’t have it, use public alternatives or [train your own using YOLOv8](https://docs.ultralytics.com/tasks/detect/#train-on-custom-data).

---

## 🙌 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* [FaceNet PyTorch](https://github.com/timesler/facenet-pytorch)

---

## 📸 Demo

> Coming soon: GIF/Video demo of real-time face + plate recognition

---

## 📬 Contact

For queries, reach out via GitHub Issues or email the project maintainer.

```
