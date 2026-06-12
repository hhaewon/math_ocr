import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# [수정] 두 가지 경로를 모두 정의합니다.
TRAIN_PATH = r"C:\Users\haewon\Desktop\Coding\math_ocr\archive\train"
EVAL_PATH = r"C:\Users\haewon\Desktop\Coding\math_ocr\archive\eval"

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
    "plus",
    "minus",
    "times",
    "div",
    "equal",
    "decimal",
]

IMG_SIZE = 32


def preprocess_image(img):
    img = img.convert("L")
    arr = np.array(img).astype(np.float32)

    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    if np.mean(corners) < 127:
        arr = 255.0 - arr

    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(arr_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh_corners = [
        thresh[0, 0],
        thresh[0, -1],
        thresh[-1, 0],
        thresh[-1, -1],
    ]
    if np.mean(thresh_corners) < 127:
        thresh = 255 - thresh

    img = Image.fromarray(thresh)

    w, h = img.size
    size = max(w, h)
    padded = Image.new("L", (size, size), 255)
    padded.paste(img, ((size - w) // 2, (size - h) // 2))

    margin = int(size * 0.2)
    final_size = size + margin * 2
    final_img = Image.new("L", (final_size, final_size), 255)
    final_img.paste(padded, (margin, margin))

    final_img = final_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(final_img).astype(np.float32)


# [수정] 특정 경로를 인자로 받아 로딩하도록 변경
def load_data(data_path):
    images, labels = [], []
    for label in CLASSES:
        folder = os.path.join(data_path, label)
        if not os.path.exists(folder):
            continue

        files = [
            f for f in os.listdir(folder) if f.endswith(".png") or f.endswith(".jpg")
        ]

        for fname in files:
            img = Image.open(os.path.join(folder, fname))
            arr = preprocess_image(img)
            images.append(arr)
            labels.append(label)

    return np.array(images), np.array(labels)


# 1. Train 데이터 로딩 및 전처리
print("1. Train 데이터 로딩 시작...")
X_raw, y_raw = load_data(TRAIN_PATH)
print(f"   Train 이미지 수: {len(X_raw)}")

X_raw = X_raw / 255.0
X_raw = X_raw.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_enc = np.array([CLASSES.index(label) for label in y_raw])
y_cat = keras.utils.to_categorical(y_enc, num_classes=len(CLASSES))

X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y_cat, test_size=0.2, random_state=42
)

# 2. Eval (Test) 데이터 로딩 및 전처리
print("2. Eval(최종 평가용) 데이터 로딩 시작...")
X_eval_raw, y_eval_raw = load_data(EVAL_PATH)
print(f"   Eval 이미지 수: {len(X_eval_raw)}")

X_eval = X_eval_raw / 255.0
X_eval = X_eval.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_eval_enc = np.array([CLASSES.index(label) for label in y_eval_raw])
y_eval_cat = keras.utils.to_categorical(y_eval_enc, num_classes=len(CLASSES))


# 고성능 CNN 레이어
model = keras.Sequential(
    [
        keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(CLASSES), activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode="constant",
    cval=1.0,
)

callbacks = [
    ReduceLROnPlateau(
        monitor="val_accuracy", patience=3, factor=0.5, min_lr=1e-5, verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1
    ),
]

print("학습 시작...")
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=40,
    validation_data=(X_val, y_val),  # train에서 쪼갠 20%로 중간 검증
    callbacks=callbacks,
)

model.save("math_model_best.keras")
print("✅ 최고 성능 모델 저장 완료 (math_model_best.keras)")

# 📌 [추가] 3. 완전히 격리된 Eval 데이터셋으로 진짜 최종 실력 평가
print("\n📢 [최종 검증] 한 번도 보지 못한 Eval 데이터셋으로 정확도 측정...")
eval_loss, eval_acc = model.evaluate(X_eval, y_eval_cat, verbose=0)
print(f"🔥 Eval 데이터셋 최종 정확도 (Accuracy): {eval_acc * 100:.2f}%")
