import os
import cv2
import numpy as np
import json
import tensorflow as tf

print("="*50)
print("ASL SIGN LANGUAGE PREDICTION")
print("="*50)

# Load model và class names
print("\nDang load model...")
model = tf.keras.models.load_model('./models/best_model.keras')
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

def test_with_sample_images():
    """Test với ảnh mẫu từ dataset"""
    print("\n" + "="*50)
    print("TEST VOI ANH MAU TU DATASET")
    print("="*50)
    
    TRAIN_DIR = './data/asl_alphabet_train/asl_alphabet_train'
    
    # Test 10 class ngẫu nhiên
    test_classes = ['A', 'B', 'C', 'H', 'L', 'N', 'S', 'U', 'X', 'Y']
    
    correct = 0
    total = len(test_classes)
    
    for cls in test_classes:
        cls_path = os.path.join(TRAIN_DIR, cls)
        # Lấy ảnh cuối cùng (ít được train)
        img_name = sorted(os.listdir(cls_path))[-1]
        img_path = os.path.join(cls_path, img_name)
        
        img = cv2.imread(img_path)
        predicted, confidence = predict_image(img)
        
        status = "✓" if predicted == cls else "✗"
        if predicted == cls:
            correct += 1
        
        print(f"   {status} Actual: {cls} | Predicted: {predicted} ({confidence:.1f}%)")
    
    print(f"\nKet qua: {correct}/{total} ({correct/total*100:.1f}%)")

def run_webcam():
    """Chạy nhận diện real-time với webcam"""
    print("\n" + "="*50)
    print("WEBCAM REAL-TIME PREDICTION")
    print("="*50)
    print("Nhan 'q' de thoat")
    print("Dat tay vao vung mau XANH")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Khong the mo webcam!")
        return
    
    # Vùng ROI (Region of Interest)
    roi_top, roi_bottom = 100, 400
    roi_left, roi_right = 100, 400
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame (mirror effect)
        frame = cv2.flip(frame, 1)
        
        # Vẽ vùng ROI
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        
        # Cắt vùng ROI
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        # Predict
        predicted, confidence = predict_image(roi)
        
        # Hiển thị kết quả
        text = f"{predicted}: {confidence:.1f}%"
        cv2.putText(frame, text, (roi_left, roi_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Hiển thị hướng dẫn
        cv2.putText(frame, "Nhan 'q' de thoat", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Hiển thị frame
        cv2.imshow('ASL Sign Language Recognition', frame)
        
        # Nhấn 'q' để thoát
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
print("2. Test voi webcam (real-time)")
print("3. Thoat")

choice = input("\nNhap lua chon (1/2/3): ").strip()

if choice == '1':
    test_with_sample_images()
elif choice == '2':
    run_webcam()
elif choice == '3':
    print("Tam biet!")
else:
    print("Lua chon khong hop le!")