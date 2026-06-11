import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import sys
import os
import cv2

IMG_SIZE = 45

# main.py의 CLASSES와 100% 일치
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
# ★ .h5 파일을 최우선으로 로드하도록 변경
if os.path.exists("math_model.h5"):
    model = keras.models.load_model("math_model.h5")
    print("✅ 성공: math_model.h5 파일 로드 완료")
elif os.path.exists("math_model_best.keras"):
    model = keras.models.load_model("math_model_best.keras")
    print("⚠️ 확인: math_model_best.keras 로드 완료")
else:
    print("❌ 에러: 모델 파일을 찾을 수 없습니다. (math_model.h5를 확인하세요)")


def preprocess_image(img):
    img = img.convert("L")
    arr = np.array(img).astype(np.float32)

    if np.mean(arr) < 127:
        arr = 255.0 - arr

    a_min, a_max = arr.min(), arr.max()
    if a_max > a_min:
        arr = (arr - a_min) / (a_max - a_min) * 255.0

    img = Image.fromarray(arr.astype(np.uint8))

    w, h = img.size
    size = max(w, h)
    padded = Image.new("L", (size, size), 255)
    padded.paste(img, ((size - w) // 2, (size - h) // 2))

    margin = int(size * 0.2)
    final_size = size + margin * 2
    final_img = Image.new("L", (final_size, final_size), 255)
    final_img.paste(padded, (margin, margin))

    final_img = final_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(final_img).astype(np.float32) / 255.0
    return arr


def predict_symbol(img_input, debug_idx=None):
    if model is None:
        return None

    img = Image.open(img_input) if isinstance(img_input, str) else img_input
    arr = preprocess_image(img)

    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    pred = model.predict(arr, verbose=0)
    label = CLASSES[np.argmax(pred)]
    confidence = np.max(pred)

    print(f"  [인식 로그] char_{debug_idx}: {label} ({confidence:.2f})")
    return SYMBOL_MAP.get(label, label)


def segment_characters(img_path):
    pil_img = Image.open(img_path).convert("RGBA")
    bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    safe_pil = Image.alpha_composite(bg, pil_img).convert("L")
    img = np.array(safe_pil)

    if np.mean(img) < 127:
        img_for_cv = 255 - img
    else:
        img_for_cv = img.copy()

    # 글자끼리 안 붙게 최소한의 블러와 타이트한 이진화 적용
    blurred = cv2.GaussianBlur(img_for_cv, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 팽창 연산을 최소화(3x3)하여 '3'과 'x'가 붙는 현상 방지
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    initial_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 2 and h > 2:
            initial_boxes.append([x, y, w, h])

    if not initial_boxes:
        print("기호를 분할하지 못했습니다.")
        return []

    # 왼쪽에서 오른쪽 순서로 1차 정렬
    initial_boxes.sort(key=lambda b: b[0])

    # ★ 자석 알고리즘: 위아래 배치된 등호(=) 조각만 수직 병합
    merged_boxes = []
    while initial_boxes:
        box = initial_boxes.pop(0)
        bx, by, bw, bh = box
        bx2, by2 = bx + bw, by + bh

        i = 0
        while i < len(initial_boxes):
            nx, ny, nw, nh = initial_boxes[i]
            nx2, ny2 = nx + nw, ny + nh

            x_overlap = max(0, min(bx2, nx2) - max(bx, nx))
            x_gap = nx - bx2

            # x축이 겹치거나 가로로 매우 가까운 박스 검사
            if x_overlap > 0 or (x_gap > 0 and x_gap < 8):
                y_gap = max(0, min(by2, ny2) - max(by, ny))
                if y_gap == 0:  # 세로로 떨어져 있는 경우만 대상
                    dist = ny - by2 if ny > by2 else by - ny2
                    if dist < 25:  # 등호 위아랫줄 사이 허용 간격
                        bx = min(bx, nx)
                        by = min(by, ny)
                        bx2 = max(bx2, nx2)
                        by2 = max(by2, nh + ny)
                        bw = bx2 - bx
                        bh = by2 - by
                        initial_boxes.pop(i)
                        continue
            i += 1
        merged_boxes.append([bx, by, bw, bh])

    print(f"  [분할 성공] 최종 수식 {len(merged_boxes)}개 기호 분리 완료")

    char_imgs = []
    for bx, by, bw, bh in merged_boxes:
        pad = max(4, int(max(bw, bh) * 0.18))
        x0 = max(0, bx - pad)
        y0 = max(0, by - pad)
        x1 = min(safe_pil.width, bx + bw + pad)
        y1 = min(safe_pil.height, by + bh + pad)

        char_crop = safe_pil.crop((x0, y0, x1, y1))
        char_imgs.append(char_crop)

    return char_imgs


def verify_equation(symbols):
    clean_symbols = [s for s in symbols if s is not None]
    print(f"\n📢 [최종 기호 리스트]: {clean_symbols}")

    if "=" not in clean_symbols:
        print("❌ 등호(=)가 인식되지 않았습니다. 분할 상태를 재확인해야 합니다.")
        return

    eq_idx = clean_symbols.index("=")
    left = clean_symbols[:eq_idx]
    right = clean_symbols[eq_idx + 1 :]

    left_expr = "".join(str(s) for s in left)
    right_expr = "".join(str(s) for s in right)

    try:
        left_val = eval(left_expr)
        right_val = eval(right_expr)
        print(f"\n[수식 연산 결과]")
        print(f"  좌변: {left_expr} = {left_val}")
        print(f"  우변: {right_expr} = {right_val}")
        if abs(left_val - right_val) < 1e-7:
            print("  ✅ 수식이 완벽히 성립합니다!")
        else:
            print(f"  ❌ 틀린 수식입니다! ({left_val} ≠ {right_val})")
    except Exception as e:
        print(f"  ❌ 계산 오류: {e}")


if __name__ == "__main__":
    image_files = sys.argv[1:]
    if not image_files:
        sys.exit(1)

    chars = segment_characters(image_files[0])
    if chars:
        symbols = [predict_symbol(c, i) for i, c in enumerate(chars)]
        verify_equation(symbols)
