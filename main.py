import os
import numpy as np
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# 데이터 경로
DATA_PATH = r"C:\Users\haewon\Desktop\Coding\math_ocr\archive\train"

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

IMG_SIZE = 45  # 32→45: 더 많은 특징 보존


def preprocess_image(img):
    """학습/추론 공통 전처리 (반드시 동일하게 유지)"""
    img = img.convert("L")
    arr = np.array(img).astype(np.float32)

    # 배경 자동 감지 후 반전 (어두운 배경 → 흰 배경으로 통일)
    if np.mean(arr) < 127:
        arr = 255.0 - arr

    # 명암 정규화 (콘트라스트 강화)
    a_min, a_max = arr.min(), arr.max()
    if a_max > a_min:
        arr = (arr - a_min) / (a_max - a_min) * 255.0

    img = Image.fromarray(arr.astype(np.uint8))

    # 정사각형 패딩 (이미지 왜곡 방지)
    w, h = img.size
    size = max(w, h)
    padded = Image.new("L", (size, size), 255)
    padded.paste(img, ((size - w) // 2, (size - h) // 2))

    # 여백 추가 (학습 데이터 스타일과 일치)
    margin = int(size * 0.2)
    final_size = size + margin * 2
    final_img = Image.new("L", (final_size, final_size), 255)
    final_img.paste(padded, (margin, margin))

    # 고품질 리사이징
    final_img = final_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(final_img).astype(np.float32) / 255.0
    return arr


def load_data():
    images, labels = [], []
    for label in CLASSES:
        folder = os.path.join(DATA_PATH, label)
        if not os.path.exists(folder):
            print(f"폴더 없음: {folder}")
            continue
        files = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"{label}: {len(files)}개 로딩 중...")
        for fname in files:
            try:
                img = Image.open(os.path.join(folder, fname))
                arr = preprocess_image(img)
                images.append(arr)
                labels.append(label)
            except Exception:
                pass
    return np.array(images), np.array(labels)


print("데이터 로딩 중...")
X_raw, y_raw = load_data()
print(f"총 이미지 수: {len(X_raw)}")

X = X_raw.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_enc = np.array([CLASSES.index(label) for label in y_raw])
y_cat = keras.utils.to_categorical(y_enc, num_classes=len(CLASSES))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.1, random_state=42
)


# ── 개선된 CNN 모델 (ResNet-style skip connections) ──
def build_model():
    inputs = keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Block 1
    x = keras.layers.Conv2D(64, 3, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.25)(x)

    # Block 2
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Block 3
    x = keras.layers.Conv2D(256, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(256, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)  # Flatten 대신 GAP (과적합 감소)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(len(CLASSES), activation="softmax")(x)

    return keras.Model(inputs, outputs)


model = build_model()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# 데이터 증강 (배경=1.0 흰색 기준)
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    fill_mode="constant",
    cval=1.0,
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=3, monitor="val_loss", verbose=1
    ),
    keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True, monitor="val_accuracy"
    ),
    keras.callbacks.ModelCheckpoint(
        "math_model_best.keras", save_best_only=True, monitor="val_accuracy"
    ),
]

print("학습 시작...")
model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
)

model.save("math_model.keras")
print("모델 저장 완료!")
print(f"최종 테스트 정확도: {model.evaluate(X_test, y_test)[1]:.4f}")
