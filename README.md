# ASL Sign Language Recognition

Nhận diện ngôn ngữ ký hiệu Mỹ (ASL) bằng Machine Learning với TensorFlow/Keras và MediaPipe.

## 🎯 Kết quả

- **Training Accuracy:** 97.06%
- **Validation Accuracy:** 73.27%
- **Best Validation Accuracy:** 73.27% (Epoch 14)
- **Classes:** 29 (A-Z + space, del, nothing)
- **Total Training Images:** ~260,000+

## 📦 Dataset

Project sử dụng 3 dataset:

| Dataset                | Số ảnh   | Nguồn                                                                   |
| ---------------------- | -------- | ----------------------------------------------------------------------- |
| ASL Alphabet           | 87,000   | [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)      |
| Sign Language MNIST    | 27,455   | [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) |
| Synthetic ASL Alphabet | 150,000+ | [Kaggle](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) |

**Tổng cộng: ~260,000+ ảnh**

- Sau quá trình tiền xử lý, lọc dữ liệu và chuẩn hóa nhãn,
  một tập con gồm 166,210 ảnh được sử dụng trong quá trình
  huấn luyện và đánh giá mô hình, bao gồm:

- 132,978 ảnh cho training
- 33,232 ảnh cho validation

## 🛠️ Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/DL-2005/asl-sign-language.git
cd asl-sign-language
```

### 2. Tạo virtual environment (Python 3.11)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

**Lưu ý quan trọng:** Nếu gặp lỗi `AttributeError: module 'mediapipe' has no attribute 'solutions'`, chạy lệnh sau:

```bash
pip uninstall tensorflow mediapipe -y
pip install tensorflow==2.13.0 mediapipe==0.10.9
```

### 4. Tải dataset

```bash
cd data
kaggle datasets download -d grassknoted/asl-alphabet
kaggle datasets download -d datamunge/sign-language-mnist
kaggle datasets download -d lexset/synthetic-asl-alphabet
```

### 5. Giải nén dataset

```bash
tar -xf asl-alphabet.zip
tar -xf sign-language-mnist.zip
tar -xf synthetic-asl-alphabet.zip
```

### 6. Gộp dataset

```bash
cd ..
python src/merge_all_datasets.py
```

## 🚀 Sử dụng

### Training model

```bash
python src/train.py
```

### Đánh giá model

```bash
python src/evaluate.py
```

### Nhận diện real-time với webcam

```bash
python src/predict.py
```

Chọn option 2 để dùng MediaPipe Hand Landmarks.

## 📁 Cấu trúc thư mục

```
asl-sign-language/
├── data/
│   ├── asl_alphabet_train/    # Dataset chính
│   ├── Train_Alphabet/        # Synthetic ASL
│   └── sign_mnist_train.csv   # MNIST CSV
├── models/
│   ├── best_model.h5          # Model tốt nhất
│   ├── class_names.json       # Danh sách classes
│   └── training_history.png   # Biểu đồ training
├── src/
│   ├── train.py               # Training model
│   ├── evaluate.py            # Đánh giá model
│   ├── predict.py             # Nhận diện real-time
│   ├── merge_all_datasets.py  # Gộp dataset
│   └── convert_mnist.py       # Convert MNIST
├── requirements.txt
└── README.md
```

## 🔧 Công nghệ

- Python 3.11
- TensorFlow 2.13
- MediaPipe 0.10.9
- OpenCV
- NumPy
- Matplotlib

## 📊 Model Architecture

```
CNN với 4 Convolutional Blocks:
- Block 1: Conv2D(32) → BatchNorm → MaxPool → Dropout
- Block 2: Conv2D(64) → BatchNorm → MaxPool → Dropout
- Block 3: Conv2D(128) → BatchNorm → MaxPool → Dropout
- Block 4: Conv2D(256) → BatchNorm → Dropout
- Fully Connected: Dense(512) → Dense(29)
```

## 👤 Tác giả

- **DL-2005** - [GitHub](https://github.com/DL-2005)
- **AlphaJCut** - [GitHub](https://github.com/AlphaJCut)

## 📄 License

MIT License
