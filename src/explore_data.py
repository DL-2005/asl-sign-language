import os
import cv2
import matplotlib
matplotlib.use('Agg')  # Không dùng GUI, lưu file trực tiếp
import matplotlib.pyplot as plt
import numpy as np

# Đường dẫn dataset
TRAIN_DIR = './data/asl_alphabet_train/asl_alphabet_train'

# Lấy danh sách classes
classes = sorted(os.listdir(TRAIN_DIR))
print(f"So luong classes: {len(classes)}")
print(f"Classes: {classes}")
print()

# Đếm số ảnh mỗi class
print("So anh moi class:")
for cls in classes[:5]:
    path = os.path.join(TRAIN_DIR, cls)
    count = len(os.listdir(path))
    print(f"   {cls}: {count} anh")
print("   ...")

# Xem kích thước ảnh mẫu
sample_class = classes[0]
sample_path = os.path.join(TRAIN_DIR, sample_class)
sample_img_name = os.listdir(sample_path)[0]
sample_img = cv2.imread(os.path.join(sample_path, sample_img_name))
print(f"\nKich thuoc anh: {sample_img.shape}")
print(f"   Height: {sample_img.shape[0]}")
print(f"   Width: {sample_img.shape[1]}")
print(f"   Channels: {sample_img.shape[2]}")


def show_samples():
    """Lưu ảnh mẫu từ mỗi class"""
    fig, axes = plt.subplots(3, 9, figsize=(18, 6))
    axes = axes.flatten()
    
    for idx, cls in enumerate(classes[:27]):
        cls_path = os.path.join(TRAIN_DIR, cls)
        img_name = os.listdir(cls_path)[0]
        img_path = os.path.join(cls_path, img_name)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f"{cls}", fontsize=12)
        axes[idx].axis('off')
    
    plt.suptitle("ASL Alphabet - Mau tu moi class", fontsize=14)
    plt.tight_layout()
    plt.savefig('./data/sample_images.png', dpi=150)
    plt.close()
    print("\nDa luu: ./data/sample_images.png")


def show_one_class_samples(class_name='A', num_samples=10):
    """Lưu nhiều ảnh từ một class"""
    cls_path = os.path.join(TRAIN_DIR, class_name)
    img_names = os.listdir(cls_path)[:num_samples]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, img_name in enumerate(img_names):
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f"{class_name} - {idx+1}", fontsize=10)
        axes[idx].axis('off')
    
    plt.suptitle(f"10 mau cua class '{class_name}'", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'./data/class_{class_name}_samples.png', dpi=150)
    plt.close()
    print(f"Da luu: ./data/class_{class_name}_samples.png")


# Chạy
print("\n" + "="*50)
print("Dang tao hinh anh mau...")
print("="*50)

show_samples()
show_one_class_samples('A', 10)
show_one_class_samples('B', 10)

print("\n" + "="*50)
print("HOAN THANH! Mo thu muc 'data' de xem cac file anh.")
print("="*50)