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
        print(f"{label}: {len(files)}개 로딩")
        for fname in files:
            img = Image.open(os.path.join(folder, fname)).convert("L")
            # verify.py와 동일하게 LANCZOS 리사이징 사용
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            arr = np.array(img).astype(np.float32)
            
            # verify.py와 동일하게 명암 정규화 적용 (콘트라스트 강화)
            a_min, a_max = arr.min(), arr.max()
            if a_max > a_min:
                arr = (arr - a_min) / (a_max - a_min) * 255.0
            
            images.append(arr)
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

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 더 깊고 강력한 CNN 모델 구조
model = keras.Sequential([
    keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    
    # 첫 번째 블록
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),
    
    # 두 번째 블록
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),
    
    # 세 번째 블록
    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    
    # 완전 연결 층
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 데이터 증강 (Data Augmentation) - 손글씨의 변형에 강인하게 만듦
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

print("학습 시작...")
# 증강된 데이터로 학습
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_test, y_test)
)

model.save("math_model.h5")
print("모델 저장 완료!")

