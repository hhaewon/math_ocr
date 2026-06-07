import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# 데이터 경로
DATA_PATH = r"C:\Users\haewon\Desktop\Coding\math_ocr\archive\train"

# 우리가 쓸 클래스만 선택
CLASSES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "plus cleaned",
    "minus",
    "times",
    "div",
    "equal",
    "decimal",
]

IMG_SIZE = 32


def load_data():
    images, labels = [], []
    for label in CLASSES:
        folder = os.path.join(DATA_PATH, label)
        if not os.path.exists(folder):
            print(f"폴더 없음: {folder}")
            continue
        files = [
            f for f in os.listdir(folder) if f.endswith(".png") or f.endswith(".jpg")
        ]
        print(f"{label}: {len(files)}개 로딩")  # 추가
        for fname in files:
            img = Image.open(os.path.join(folder, fname)).convert("RGB").convert("L")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img))
            labels.append(label)
    return np.array(images), np.array(labels)


print("데이터 로딩 중...")
X, y = load_data()
print(f"총 이미지 수: {len(X)}")

# 전처리
X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# LabelEncoder 대신 CLASSES 리스트의 인덱스를 직접 사용 (알파벳 정렬 방지)
y_enc = np.array([CLASSES.index(label) for label in y])
y_cat = keras.utils.to_categorical(y_enc, num_classes=len(CLASSES))

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

# CNN 모델
model = keras.Sequential(
    [
        keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)
        ),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(len(CLASSES), activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("학습 시작...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save("math_model.h5")
print("모델 저장 완료!")
