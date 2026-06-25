import os
import pickle
import cv2
import numpy as np

KNOWN_FACES_DIR = "Data Set/Known Faces"
MODEL_FILE = "trained_faces.yml"
LABELS_FILE = "labels.pkl"

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}

label_id = 0

print("Training started...")

# Step 1: Image Processing Loop
for filename in os.listdir(KNOWN_FACES_DIR):

    if filename.lower().endswith((".jpg", "jpeg", "png")):
        
        path = os.path.join(KNOWN_FACES_DIR, filename)
        name = os.path.splitext(filename)[0]

        image = cv2.imread(path)
        if image is None:
            print(f"Could not read {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (80, 80)
        )

        if len(detected_faces) == 0:
            print(f"No face found in {filename}")
            continue  # Skip to the next image if no face is found

        # Extract face regions safely
        x, y, w, h = detected_faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        faces.append(face_roi)
        labels.append(label_id)
        label_names[label_id] = name

        print(f"Loaded: {name}")
        label_id += 1

# Step 2: Training and Saving (Executed OUTSIDE the loop)
if len(faces) == 0:
    print("No faces found. Training failed.")
    exit()

print("\nProcessing images complete. Training model...")
recognizer.train(faces, np.array(labels))
recognizer.save(MODEL_FILE)

with open(LABELS_FILE, "wb") as f:
    pickle.dump(label_names, f)

print("\nTraining Complete!")
print(f"Saved model to: {MODEL_FILE}")
print(f"Saved labels: {LABELS_FILE}")
print(f"Total people trained: {len(faces)}")
