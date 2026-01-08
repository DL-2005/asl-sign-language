import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print()

# ============================================
# CẤU HÌNH
# ============================================
TRAIN_DIR = './data/asl_alphabet_train/asl_alphabet_train'
IMG_SIZE = 64          # Resize ảnh về 64x64
BATCH_SIZE = 32
EPOCHS = 15

# ============================================
# BƯỚC 1: TẠO DATA GENERATORS
# ============================================
print("="*50)
print("BUOC 1: Tao Data Generators")
print("="*50)

# Data Augmentation cho training
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values 0-1
    rotation_range=10,        # Xoay ngẫu nhiên ±10 độ
    width_shift_range=0.1,    # Dịch ngang 10%
    height_shift_range=0.1,   # Dịch dọc 10%
    zoom_range=0.1,           # Zoom 10%
    horizontal_flip=False,    # KHÔNG flip (quan trọng cho sign language!)
    validation_split=0.2      # 20% cho validation
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Number of classes: {train_generator.num_classes}")
print(f"Class names: {list(train_generator.class_indices.keys())}")

# Lưu class names để dùng sau
class_names = list(train_generator.class_indices.keys())
import json
with open('./models/class_names.json', 'w') as f:
    json.dump(class_names, f)
print("\nDa luu class names vao: ./models/class_names.json")

# ============================================
# BƯỚC 2: XÂY DỰNG MODEL CNN
# ============================================
print("\n" + "="*50)
print("BUOC 2: Xay dung Model CNN")
print("="*50)

def create_cnn_model(input_shape=(64, 64, 3), num_classes=29):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Fully Connected
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Tạo model
model = create_cnn_model(num_classes=len(class_names))
model.summary()

# ============================================
# BƯỚC 3: COMPILE MODEL
# ============================================
print("\n" + "="*50)
print("BUOC 3: Compile Model")
print("="*50)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Model compiled thanh cong!")

# ============================================
# BƯỚC 4: CALLBACKS
# ============================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        './models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ============================================
# BƯỚC 5: TRAINING
# ============================================
print("\n" + "="*50)
print("BUOC 5: Bat dau Training")
print("="*50)
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# ============================================
# BƯỚC 6: LƯU MODEL VÀ VẼ BIỂU ĐỒ
# ============================================
print("\n" + "="*50)
print("BUOC 6: Luu Model va Ve Bieu Do")
print("="*50)

# Lưu model cuối cùng
model.save('./models/asl_model_final.h5')
print("Da luu model: ./models/asl_model_final.h5")

# Vẽ biểu đồ training
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('./models/training_history.png', dpi=150)
plt.close()
print("Da luu bieu do: ./models/training_history.png")

# ============================================
# HOÀN THÀNH
# ============================================
print("\n" + "="*50)
print("HOAN THANH TRAINING!")
print("="*50)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print("\nCac file da luu:")
print("  - ./models/best_model.h5")
print("  - ./models/asl_model_final.h5")
print("  - ./models/class_names.json")
print("  - ./models/training_history.png")