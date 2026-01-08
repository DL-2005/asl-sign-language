import os
import cv2
import numpy as np
import json
import tensorflow as tf
import mediapipe as mp

print("="*50)
print("ASL SIGN LANGUAGE PREDICTION + MEDIAPIPE")
print("="*50)

# ============================================
# SETUP MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# LOAD MODEL VÀ CLASS NAMES
# ============================================
print("\nDang load model...")
model = tf.keras.models.load_model('./models/best_model.h5')
print("Da load model thanh cong!")

with open('./models/class_names.json', 'r') as f:
    class_names = json.load(f)
print(f"So luong classes: {len(class_names)}")

IMG_SIZE = 64

def preprocess_image(image):
    """Tiền xử lý ảnh để predict"""
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image):
    """Dự đoán class của ảnh"""
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

def get_hand_bbox(hand_landmarks, frame_width, frame_height, padding=30):
    """Lấy bounding box từ hand landmarks"""
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    x_min = int(min(x_coords) * frame_width) - padding
    x_max = int(max(x_coords) * frame_width) + padding
    y_min = int(min(y_coords) * frame_height) - padding
    y_max = int(max(y_coords) * frame_height) + padding
    
    # Clamp to frame boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_width, x_max)
    y_max = min(frame_height, y_max)
    
    return x_min, y_min, x_max, y_max

def test_with_sample_images():
    """Test với ảnh mẫu từ dataset"""
    print("\n" + "="*50)
    print("TEST VOI ANH MAU TU DATASET")
    print("="*50)
    
    TRAIN_DIR = './data/asl_alphabet_train/asl_alphabet_train'
    
    test_classes = ['A', 'B', 'C', 'H', 'L', 'N', 'S', 'U', 'X', 'Y']
    
    correct = 0
    total = len(test_classes)
    
    for cls in test_classes:
        cls_path = os.path.join(TRAIN_DIR, cls)
        img_name = sorted(os.listdir(cls_path))[-1]
        img_path = os.path.join(cls_path, img_name)
        
        img = cv2.imread(img_path)
        predicted, confidence = predict_image(img)
        
        status = "DUNG" if predicted == cls else "SAI"
        if predicted == cls:
            correct += 1
        
        print(f"   [{status}] Actual: {cls} | Predicted: {predicted} ({confidence:.1f}%)")
    
    print(f"\nKet qua: {correct}/{total} ({correct/total*100:.1f}%)")

def run_webcam_mediapipe():
    """Webcam với MediaPipe hand landmarks"""
    print("\n" + "="*50)
    print("WEBCAM + MEDIAPIPE HAND LANDMARKS")
    print("="*50)
    print("Nhan 'q' de thoat")
    print("Dua tay vao camera de nhan dien")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Khong the mo webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Vẽ landmarks đẹp như hình mẫu
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Lấy bounding box
                x_min, y_min, x_max, y_max = get_hand_bbox(
                    hand_landmarks, frame_width, frame_height
                )
                
                # Vẽ bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Crop hand và predict
                hand_roi = frame[y_min:y_max, x_min:x_max]
                if hand_roi.size > 0:
                    predicted, confidence = predict_image(hand_roi)
                    
                    # Hiển thị kết quả
                    text = f"{predicted}: {confidence:.1f}%"
                    cv2.putText(frame, text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Hiển thị Left/Right hand
                if results.multi_handedness:
                    hand_label = results.multi_handedness[idx].classification[0].label
                    cv2.putText(frame, hand_label, (x_min, y_max + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Nhan 'q' de thoat", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "MediaPipe Hand Landmarks", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('ASL Sign Language Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nDa dong webcam.")

def run_webcam_roi():
    """Webcam với vùng ROI cố định"""
    print("\n" + "="*50)
    print("WEBCAM VOI VUNG ROI CO DINH")
    print("="*50)
    print("Nhan 'q' de thoat")
    print("Dat tay vao vung mau XANH")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Khong the mo webcam!")
        return
    
    roi_size = 300
    roi_top, roi_left = 100, 100
    roi_bottom = roi_top + roi_size
    roi_right = roi_left + roi_size
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Draw ROI
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        
        # Get ROI
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        # Predict
        predicted, confidence = predict_image(roi)
        
        # Display result
        text = f"{predicted}: {confidence:.1f}%"
        cv2.putText(frame, text, (roi_left, roi_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Instructions
        cv2.putText(frame, "Nhan 'q' de thoat", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Dat tay vao vung xanh", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('ASL Sign Language Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDa dong webcam.")

# ============================================
# MENU CHÍNH
# ============================================
print("\n" + "="*50)
print("CHON CHE DO")
print("="*50)
print("1. Test voi anh mau tu dataset")
print("2. Webcam + MediaPipe Hand Landmarks")
print("3. Webcam voi vung ROI co dinh")
print("4. Thoat")

choice = input("\nNhap lua chon (1/2/3/4): ").strip()

if choice == '1':
    test_with_sample_images()
elif choice == '2':
    run_webcam_mediapipe()
elif choice == '3':
    run_webcam_roi()
elif choice == '4':
    print("Tam biet!")
else:
    print("Lua chon khong hop le!")