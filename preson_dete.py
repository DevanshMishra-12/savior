import cv2
import tensorflow as tf
import numpy as np
import subprocess
import sys
import os
import time
from datetime import datetime

# Firebase imports
import firebase_admin
from firebase_admin import credentials, storage, firestore

# -------------------- Firebase Setup --------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json.json")  # your Firebase key file
    firebase_admin.initialize_app(cred, {
        "storageBucket": "person-detection-ai.appspot.com"  # replace with your Firebase project bucket
    })
    db = firestore.client()
    bucket = storage.bucket()

def upload_to_firebase(local_path, people_count):
    file_name = os.path.basename(local_path)
    blob = bucket.blob(f"detections/{file_name}")
    blob.upload_from_filename(local_path)
    blob.make_public()  # optional (gives you a public URL)

    # Save metadata to Firestore
    doc_ref = db.collection("detections").document(file_name)
    doc_ref.set({
        "file_name": file_name,
        "people_count": people_count,
        "timestamp": datetime.utcnow().isoformat(),
        "url": blob.public_url
    })
    print(f"[FIREBASE] Uploaded {file_name} with {people_count} people. URL: {blob.public_url}")

# -------------------- Load TFLite Model --------------------
interpreter = tf.lite.Interpreter(model_path="yolov5s-fp16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------- Helper Functions --------------------
def call_voice_script():
    script_path = os.path.join(os.path.dirname(__file__), "ai_voice.py")
    if os.path.isfile(script_path):
        subprocess.Popen([sys.executable, script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("[ERROR] ai_voice.py not found!")

def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_resized / 255.0
    return np.expand_dims(img_norm, axis=0).astype(np.float32)

def nms(boxes, scores, iou_threshold=0.4):
    if len(boxes) == 0:
        return []
    boxes_array = np.array(boxes)
    scores_array = np.array(scores)
    x1 = boxes_array[:,0] - boxes_array[:,2]/2
    y1 = boxes_array[:,1] - boxes_array[:,3]/2
    x2 = boxes_array[:,0] + boxes_array[:,2]/2
    y2 = boxes_array[:,1] + boxes_array[:,3]/2
    areas = (x2 - x1) * (y2 - y1)
    order = scores_array.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return boxes_array[keep]

# -------------------- Setup --------------------
save_dir = os.path.join(os.path.dirname(__file__), "detections")
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 800, 600)

frame_count = 0
person_present = False

# -------------------- Main Loop --------------------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    frame_count += 1
    h, w, _ = frame.shape

    # Inference
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    preds = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

    boxes = preds[:, :4]
    scores = preds[:, 4]
    classes = np.argmax(preds[:, 5:], axis=-1)

    # Only detect people (class 0 in COCO dataset)
    mask = (classes == 0) & (scores > 0.5)
    person_boxes = boxes[mask]
    person_scores = scores[mask]

    filtered_boxes = nms(person_boxes, person_scores)
    people_count = len(filtered_boxes)

    # Draw boxes
    for box in filtered_boxes:
        x, y, bw, bh = box
        x1 = int((x - bw/2) * w / 640)
        y1 = int((y - bh/2) * h / 640)
        x2 = int((x + bw/2) * w / 640)
        y2 = int((y + bh/2) * h / 640)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Total Persons: {people_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Trigger operations on detection
    if people_count > 0 and not person_present:
        call_voice_script()
        person_present = True
        save_path = os.path.join(save_dir, f"frame_{frame_count}_people{people_count}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"[INFO] Saved {save_path}")

        # Upload snapshot + metadata to Firebase
        upload_to_firebase(save_path, people_count)

    elif people_count == 0 and person_present:
        person_present = False

    cv2.imshow("Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
