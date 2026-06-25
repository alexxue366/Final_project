import cv2
import pickle
import os
import time  # Added for cooldown tracking
from datetime import datetime

MODEL_FILE = "trained_faces.yml"
LABELS_FILE = "labels.pkl"
UNKNOWN_DIR = "Data Set/Unknown faces"

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# --- FIX 1: Load the label names from your pickle file ---
if not os.path.exists(LABELS_FILE):
    print(f"Error: {LABELS_FILE} not found. Run your training script first.")
    exit()

with open(LABELS_FILE, "rb") as f:
    label_names = pickle.load(f)

# Load trained model 
if not os.path.exists(MODEL_FILE):
    print(f"Error: {MODEL_FILE} not found. Run your training script first.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Camera
cap = cv2.VideoCapture(0)

print("Starting face recognition...")

# Cooldown tracking variable (Initializes to 0)
last_saved_time = 0  

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # --- FIX 2: Removed stray backslash at the end of this line ---
    for (x, y, w, h) in faces:
        
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (200, 200))

        label_id, confidence = recognizer.predict(face_roi)

        # Lower confidence = better match
        if confidence < 80:
            name = label_names.get(label_id, "Unknown")
            color = (0, 255, 0)  # Green for recognized faces
        else:
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown faces

            # --- OPTIMIZATION: 2-second saving cooldown logic ---
            current_time = time.time()
            if current_time - last_saved_time > 2.0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(
                    UNKNOWN_DIR,
                    f"unknown_{timestamp}.jpg"
                )
                cv2.imwrite(filename, frame)
                print(f"Saved unknown intruder: {filename}")
                last_saved_time = current_time

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{name} ({confidence:.0f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Recognition AI", frame)

    # --- FIX 3: Corrected lowercase 'k' to uppercase 'K' ---
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
