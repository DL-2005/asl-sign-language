import os
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

print("="*50)
print("DANH GIA MODEL")
print("="*50)

# ============================================
# LOAD MODEL VÀ CLASS NAMES
# ============================================
print("\nDang load model...")
model = tf.keras.models.load_model('./models/best_model.keras')
print("Da load model: ./models/best_model.keras")

with open('./models/class_names.json', 'r') as f:
    class_names = json.load(f)
print(f"So luong classes: {len(class_names)}")

# ============================================
# CHUẨN BỊ VALIDATION DATA
# ============================================
TRAIN_DIR = './data/asl_alphabet_train/asl_alphabet_train'
IMG_SIZE = 64
BATCH_SIZE = 32

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_generator = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ============================================
# ĐÁNH GIÁ TRÊN VALIDATION SET
# ============================================
print("\n" + "="*50)
print("DANH GIA TREN VALIDATION SET")
print("="*50)

loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================
# DỰ ĐOÁN VÀ TẠO CONFUSION MATRIX
# ============================================
print("\n" + "="*50)
print("TAO CONFUSION MATRIX")
print("="*50)

print("Dang du doan tren validation set...")
val_generator.reset()
y_pred = model.predict(val_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

# Classification Report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print(report)

# Lưu report vào file
with open('./models/classification_report.txt', 'w') as f:
    f.write("CLASSIFICATION REPORT\n")
    f.write("="*50 + "\n")
    f.write(f"Validation Accuracy: {accuracy*100:.2f}%\n\n")
    f.write(report)
print("Da luu report: ./models/classification_report.txt")

# ============================================
# VẼ CONFUSION MATRIX
# ============================================
print("\nDang ve Confusion Matrix...")

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./models/confusion_matrix.png', dpi=150)
plt.close()
print("Da luu: ./models/confusion_matrix.png")

# ============================================
# PHÂN TÍCH ACCURACY THEO CLASS
# ============================================
print("\n" + "="*50)
print("PHAN TICH THEO CLASS")
print("="*50)

class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100

print("\n5 Class co accuracy THAP nhat:")
worst_indices = np.argsort(class_accuracy)[:5]
for idx in worst_indices:
    print(f"   {class_names[idx]}: {class_accuracy[idx]:.2f}%")

print("\n5 Class co accuracy CAO nhat:")
best_indices = np.argsort(class_accuracy)[-5:][::-1]
for idx in best_indices:
    print(f"   {class_names[idx]}: {class_accuracy[idx]:.2f}%")

# Vẽ biểu đồ
plt.figure(figsize=(14, 6))
bars = plt.bar(class_names, class_accuracy, color='steelblue')
for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
    if acc < 85:
        bar.set_color('salmon')
plt.axhline(y=85, color='r', linestyle='--', label='85% threshold')
plt.title('Accuracy theo tung Class', fontsize=14)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 105)
plt.legend()
plt.tight_layout()
plt.savefig('./models/accuracy_per_class.png', dpi=150)
plt.close()
print("Da luu: ./models/accuracy_per_class.png")

# ============================================
# HOÀN THÀNH
# ============================================
print("\n" + "="*50)
print("HOAN THANH DANH GIA!")
print("="*50)
print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
print(f"\nCac file da luu:")
print("   - ./models/classification_report.txt")
print("   - ./models/confusion_matrix.png")
print("   - ./models/accuracy_per_class.png")