import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import sys
import os
import cv2

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
    "plus": "+",
    "minus": "-",
    "times": "*",
    "div": "/",
    "equal": "=",
    "decimal": ".",
}

# 새로 학습된 베스트 모델 로드
model = keras.models.load_model("math_model.keras")
print("✅ 성공: math_model.keras 파일 로드 완료")

input_shape = model.input_shape
IMG_SIZE = input_shape[1] if input_shape[1] is not None else 32


def preprocess_image(img):
    img = img.convert("L")
    arr = np.array(img).astype(np.float32)

    # 1. 모서리 기반 1차 반전 (검은 바탕 -> 흰 바탕)
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    if np.mean(corners) < 127:
        arr = 255.0 - arr

    # 2. 오츠 이진화  적용 (순수 흑백 고정)
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(arr_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh_corners = [thresh[0, 0], thresh[0, -1], thresh[-1, 0], thresh[-1, -1]]
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

    # 전체 이미지 모서리 기반 배경 검사
    corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
    if np.mean(corners) < 127:
        img_for_cv = 255 - img
    else:
        img_for_cv = img.copy()

    blurred = cv2.GaussianBlur(img_for_cv, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
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

    initial_boxes.sort(key=lambda b: b[0])

    merged_boxes = []
    while initial_boxes:
        box = initial_boxes.pop(0)
        bx, by, bw, bh = box

        i = 0
        while i < len(initial_boxes):
            nx, ny, nw, nh = initial_boxes[i]
            overlap_x = max(0, min(bx + bw, nx + nw) - max(bx, nx))
            min_w = min(bw, nw)
            center_b_x = bx + bw / 2
            center_n_x = nx + nw / 2
            center_diff_x = abs(center_b_x - center_n_x)

            if (overlap_x / min_w > 0.3) or (center_diff_x < 20):
                v_gap = ny - (by + bh) if ny > by else by - (ny + nh)
                if v_gap < 60 or v_gap < max(bw, nw) * 1.5:
                    x0 = min(bx, nx)
                    y0 = min(by, ny)
                    x1 = max(bx + bw, nx + nw)
                    y1 = max(by + bh, ny + nh)
                    bx, by, bw, bh = x0, y0, x1 - x0, y1 - y0
                    initial_boxes.pop(i)
                    continue
            i += 1
        merged_boxes.append([bx, by, bw, bh])

    merged_boxes.sort(key=lambda b: b[0])
    print(f"  [분할 성공] 최종 수식 {len(merged_boxes)}개 기호 분리 완료")
    os.makedirs("debug_chars", exist_ok=True)
    char_imgs = []
    for i, (bx, by, bw, bh) in enumerate(merged_boxes):
        char_crop = safe_pil.crop((bx, by, bx + bw, by + bh))
        pad = max(4, int(max(bw, bh) * 0.15))
        padded_char = Image.new("L", (bw + pad * 2, bh + pad * 2), 255)
        padded_char.paste(char_crop, (pad, pad))
        char_imgs.append(char_crop)

        from verify import preprocess_image

        processed_arr = preprocess_image(char_crop) * 255.0
        Image.fromarray(processed_arr.astype(np.uint8)).save(
            f"debug_chars/char_{i}_input.png"
        )

    return char_imgs


def verify_equation(symbols):
    clean_symbols = [s for s in symbols if s is not None]

    idx = 0
    while idx < len(clean_symbols) - 1:
        if clean_symbols[idx] == "=" and clean_symbols[idx + 1] == "=":
            clean_symbols.pop(idx + 1)
        else:
            idx += 1

    print(f"\n📢 [최종 기호 리스트]: {clean_symbols}")

    if "=" not in clean_symbols:
        print("❌ 등호(=)가 인식되지 않았습니다.")
        return

    eq_idx = clean_symbols.index("=")
    left = clean_symbols[:eq_idx]
    right = clean_symbols[eq_idx + 1 :]

    left_expr = "".join(str(s) for s in left)
    right_expr = "".join(str(s) for s in right)

    try:
        left_val = eval(left_expr)
        right_val = eval(right_expr)
        print("\n[수식 연산 결과]")
        print(f"  좌변: {left_expr} = {left_val}")
        print(f"  우변: {right_expr} = {right_val}")
        if abs(left_val - right_val) < 1e-7:
            print("  ✅ 수식이 완벽히 성립합니다!")
        else:
            print(f"  ❌ 틀린 수식입니다! ({left_val} ≠ {right_val})")
    except Exception as e:
        print(
            f"  ❌ 계산 오류: {e} (생성된 식 내부 구조 확인 필요: {left_expr} = {right_expr})"
        )


if __name__ == "__main__":
    image_files = sys.argv[1:]
    if not image_files:
        sys.exit(1)

    chars = segment_characters(image_files[0])
    if chars:
        symbols = [predict_symbol(c, i) for i, c in enumerate(chars)]
        verify_equation(symbols)
