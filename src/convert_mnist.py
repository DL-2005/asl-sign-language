import os
import numpy as np
import pandas as pd
from PIL import Image

print("="*50)
print("CONVERT SIGN LANGUAGE MNIST TO IMAGES")
print("="*50)

# Đọc CSV
TRAIN_CSV = './data/sign_mnist_train.csv'
OUTPUT_DIR = './data/asl_alphabet_train/asl_alphabet_train'

print(f"Dang doc: {TRAIN_CSV}")
df = pd.read_csv(TRAIN_CSV)
print(f"So luong samples: {len(df)}")

# Labels: 0-25 (A-Z, nhưng không có J=9 và Z=25 vì cần chuyển động)
# Mapping label to letter
label_to_letter = {i: chr(65 + i) for i in range(26)}
# J (9) và Z (25) không có trong dataset này
# Dataset dùng 0-24 mapping to A-I, K-Y

print("\nDang convert...")

count = 0
for idx, row in df.iterrows():
    label = int(row['label'])
    
    # Map label to letter (skip J which is 9)
    if label >= 9:
        letter = chr(65 + label + 1)  # Skip J
    else:
        letter = chr(65 + label)
    
    # Lấy pixel data (784 pixels = 28x28)
    pixels = row.drop('label').values.astype(np.uint8)
    
    # Reshape thành 28x28
    img_array = pixels.reshape(28, 28)
    
    # Tạo ảnh và resize lên 200x200 (kích thước của ASL Alphabet dataset)
    img = Image.fromarray(img_array, mode='L')  # Grayscale
    img = img.convert('RGB')  # Convert to RGB
    img = img.resize((200, 200), Image.Resampling.LANCZOS)
    
    # Tạo thư mục nếu chưa có
    letter_dir = os.path.join(OUTPUT_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)
    
    # Đếm số ảnh hiện có trong thư mục
    existing = len([f for f in os.listdir(letter_dir) if f.endswith(('.jpg', '.png'))])
    
    # Lưu ảnh
    filename = f"mnist_{letter}_{existing + 1:05d}.jpg"
    filepath = os.path.join(letter_dir, filename)
    img.save(filepath)
    
    count += 1
    if count % 1000 == 0:
        print(f"   Da convert: {count}/{len(df)}")

print(f"\n" + "="*50)
print(f"HOAN THANH! Da convert {count} anh")
print("="*50)

# Thống kê
print("\nThong ke theo class:")
for letter in sorted(os.listdir(OUTPUT_DIR)):
    letter_dir = os.path.join(OUTPUT_DIR, letter)
    if os.path.isdir(letter_dir):
        num_images = len([f for f in os.listdir(letter_dir) if f.endswith(('.jpg', '.png'))])
        print(f"   {letter}: {num_images} anh")