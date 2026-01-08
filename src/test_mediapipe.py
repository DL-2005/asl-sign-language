import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

print("MediaPipe version:", mp.__version__)

# Download model file nếu chưa có
MODEL_PATH = "./models/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Dang tai model hand_landmarker...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Da tai xong model!")

# Tạo options cho Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Tạo Hand Landmarker
detector = vision.HandLandmarker.create_from_options(options)

print("MediaPipe OK! Dang mo webcam...")

# Hàm vẽ landmarks
def draw_landmarks(image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        
        height, width, _ = image.shape
        
        # Vẽ các điểm landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        # Vẽ connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            start_x, start_y = int(start.x * width), int(start.y * height)
            end_x, end_y = int(end.x * width), int(end.y * height)
            
            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        
        # Hiển thị Left/Right
        hand_label = handedness[0].category_name
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - 10
        
        cv2.putText(image, hand_label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hands
    detection_result = detector.detect(mp_image)
    
    # Vẽ landmarks
    if detection_result.hand_landmarks:
        frame = draw_landmarks(frame, detection_result)
        print(f"Phat hien {len(detection_result.hand_landmarks)} tay!")
    
    cv2.putText(frame, "Nhan 'q' de thoat", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Test MediaPipe', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")