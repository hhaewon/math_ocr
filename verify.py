import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import sys
import os

IMG_SIZE = 32
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

SYMBOL_MAP = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "plus cleaned": "+",
    "minus": "-",
    "times": "*",
    "div": "/",
    "equal": "=",
    "decimal": ".",
}

model = None
if os.path.exists("math_model.h5"):
    model = keras.models.load_model("math_model.h5")


def predict_symbol(img_path):
    if model is None:
        print("모델 파일(math_model.h5)이 없습니다. 먼저 학습을 진행해주세요.")
        return None
    img = Image.open(img_path).convert("RGB").convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    pred = model.predict(arr, verbose=0)
    label = CLASSES[np.argmax(pred)]
    confidence = np.max(pred)
    print(f"{img_path}: {label} ({confidence:.2f})")
    return SYMBOL_MAP[label]


def verify_equation(symbols):
    # None이 섞여있으면 중단
    if None in symbols:
        return

    # 등호 기준으로 분리
    if "=" not in symbols:
        print("등호(=)가 없어서 계산할 수 없어요!")
        return
    
    eq_count = symbols.count("=")
    if eq_count > 1:
        print("등호가 여러 개입니다. 하나만 넣어주세요.")
        return

    eq_idx = symbols.index("=")
    left = symbols[:eq_idx]
    right = symbols[eq_idx + 1 :]

    if not left or not right:
        print("등호 좌변이나 우변이 비어있어요!")
        return

    left_expr = "".join(str(s) for s in left)
    right_expr = "".join(str(s) for s in right)

    try:
        # eval 보안 및 실수 계산을 위해 float 비교
        left_val = eval(left_expr)
        right_val = eval(right_expr)
        print(f"좌변: {left_expr} = {left_val}")
        print(f"우변: {right_expr} = {right_val}")
        
        # 부동소수점 오차 고려
        if abs(left_val - right_val) < 1e-7:
            print("✅ 맞아요!")
        else:
            print("❌ 틀렸어요!")
    except Exception as e:
        print(f"계산 오류: {e}")


# 테스트: 기호 이미지 파일들을 순서대로 넣으면 됨
# 예시: python verify.py 3.png plus.png 5.png equal.png 8.png
if __name__ == "__main__":
    image_files = sys.argv[1:]
    symbols = [predict_symbol(f) for f in image_files]
    print(f"인식된 기호: {symbols}")
    verify_equation(symbols)
