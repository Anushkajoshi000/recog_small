import cv2
import os
import numpy as np

# Path to dataset
dataset_path = "datasets"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    image_paths = []
    faces = []
    ids = []
    names = {}

    current_id = 0
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            person_path = os.path.join(root, dir_name)
            names[current_id] = dir_name
            for file in os.listdir(person_path):
                if file.endswith("jpg") or file.endswith("png"):
                    img_path = os.path.join(person_path, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    faces_rects = detector.detectMultiScale(img)
                    for (x, y, w, h) in faces_rects:
                        faces.append(img[y:y+h, x:x+w])
                        ids.append(current_id)
            current_id += 1
    return faces, ids, names

print("[INFO] Collecting faces and labels...")
faces, ids, names = get_images_and_labels(dataset_path)
print(f"[INFO] Total faces: {len(faces)}")
print(f"[INFO] Labels: {len(set(ids))} -> {names}")

# Train the recognizer
print("[INFO] Training...")
recognizer.train(faces, np.array(ids))

# Save the trained model
recognizer.write("trainer.yml")

# Save label mappings
import pickle
with open("labels.pkl", "wb") as f:
    pickle.dump(names, f)

print("[INFO] Training complete. Model saved as 'trainer.yml' and labels as 'labels.pkl'")
