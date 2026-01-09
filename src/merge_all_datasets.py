import os
import shutil
from pathlib import Path

print("="*50)
print("GOP TAT CA DATASET")
print("="*50)

# Thư mục chính
MAIN_DIR = './data/asl_alphabet_train/asl_alphabet_train'

# ============================================
# 1. GOP SYNTHETIC ASL ALPHABET
# ============================================
print("\n[1/2] Dang gop Synthetic ASL Alphabet...")

SYNTHETIC_DIR = './data/Train_Alphabet'

if os.path.exists(SYNTHETIC_DIR):
    synthetic_classes = os.listdir(SYNTHETIC_DIR)
    total_synthetic = 0
    
    for cls in synthetic_classes:
        src_dir = os.path.join(SYNTHETIC_DIR, cls)
        
        if not os.path.isdir(src_dir):
            continue
        
        # Map "Blank" to "nothing"
        if cls == 'Blank':
            dst_cls = 'nothing'
        else:
            dst_cls = cls
        
        dst_dir = os.path.join(MAIN_DIR, dst_cls)
        
        # Tạo thư mục nếu chưa có
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        # Đếm ảnh hiện có
        existing = len([f for f in os.listdir(dst_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Copy ảnh
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for i, img in enumerate(images):
            src_path = os.path.join(src_dir, img)
            new_name = f"synthetic_{dst_cls}_{existing + i + 1:05d}{Path(img).suffix}"
            dst_path = os.path.join(dst_dir, new_name)
            
            shutil.copy2(src_path, dst_path)
            total_synthetic += 1
        
        print(f"   {cls} -> {dst_cls}: +{len(images)} anh")
    
    print(f"   TONG: +{total_synthetic} anh tu Synthetic ASL")
else:
    print("   Khong tim thay Synthetic ASL dataset")

# ============================================
# 2. GOP SIGN LANGUAGE MNIST
# ============================================
print("\n[2/2] Dang gop Sign Language MNIST...")

import pandas as pd
import numpy as np
from PIL import Image

MNIST_CSV = './data/sign_mnist_train.csv'

if os.path.exists(MNIST_CSV):
    df = pd.read_csv(MNIST_CSV)
    total_mnist = 0
    
    for idx, row in df.iterrows():
        label = int(row['label'])
        
        # Map label to letter (skip J=9)
        if label >= 9:
            letter = chr(65 + label + 1)
        else:
            letter = chr(65 + label)
        
        # Lấy pixels
        pixels = row.drop('label').values.astype(np.uint8)
        img_array = pixels.reshape(28, 28)
        
        # Tạo ảnh RGB 200x200
        img = Image.fromarray(img_array, mode='L')
        img = img.convert('RGB')
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        
        # Thư mục đích
        dst_dir = os.path.join(MAIN_DIR, letter)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        # Đếm ảnh hiện có
        existing = len([f for f in os.listdir(dst_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Lưu
        filename = f"mnist_{letter}_{existing + 1:05d}.jpg"
        filepath = os.path.join(dst_dir, filename)
        img.save(filepath)
        
        total_mnist += 1
        if total_mnist % 5000 == 0:
            print(f"   Da convert: {total_mnist}/{len(df)}")
    
    print(f"   TONG: +{total_mnist} anh tu MNIST")
else:
    print("   Khong tim thay Sign Language MNIST")

# ============================================
# THONG KE CUOI CUNG
# ============================================
print("\n" + "="*50)
print("THONG KE DATASET SAU KHI GOP")
print("="*50)

total_all = 0
for cls in sorted(os.listdir(MAIN_DIR)):
    cls_dir = os.path.join(MAIN_DIR, cls)
    if os.path.isdir(cls_dir):
        count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_all += count
        print(f"   {cls}: {count} anh")

print(f"\nTONG CONG: {total_all} anh")
print("="*50)
print("HOAN THANH! Bay gio co the train:")
print("   python src/train.py")