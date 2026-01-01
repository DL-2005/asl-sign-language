# ASL Sign Language Recognition

Nhan dien ngon ngu ky hieu My (ASL) bang Machine Learning voi TensorFlow/Keras.

## Ket qua

- Training Accuracy: 98.89%
- Validation Accuracy: 88.90%

## Cai dat

```bash
git clone https://github.com/DL-2005/asl-sign-language.git
cd asl-sign-language
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Tai dataset

```bash
cd data
kaggle datasets download -d grassknoted/asl-alphabet
```

## Su dung

```bash
python src/train.py      # Training
python src/evaluate.py   # Danh gia
python src/predict.py    # Nhan dien real-time
```

## Dataset

- Nguon: ASL Alphabet Dataset - Kaggle
- So luong: 87,000 anh
- Classes: 29 (A-Z + space, del, nothing)

## Cong nghe

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
