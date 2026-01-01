import os

# Đường dẫn dataset
DATA_DIR = './data/asl_alphabet_train/asl_alphabet_train'

# Kiểm tra thư mục tồn tại
if os.path.exists(DATA_DIR):
    classes = sorted(os.listdir(DATA_DIR))
    print("Dataset da tai thanh cong!")
    print(f"So luong classes: {len(classes)}")
    print(f"Danh sach classes: {classes}")
    print()
    
    # Đếm số ảnh mỗi class
    total_images = 0
    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            total_images += count
            print(f"   {cls}: {count} anh")
    
    print(f"\nTong so anh: {total_images}")
else:
    print("Khong tim thay dataset!")
    print(f"Kiem tra duong dan: {DATA_DIR}")