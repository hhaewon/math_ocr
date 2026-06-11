import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import sys
import os
import scipy.ndimage as ndimage

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


def predict_symbol(img_input, debug_idx=None):
    """
    img_input: 파일 경로(str) 또는 PIL Image 객체
    """
    if model is None:
        print("모델 파일(math_model.h5)이 없습니다. 먼저 학습을 진행해주세요.")
        return None
    
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("L")
    else:
        img = img_input.convert("L")
        
    # 고품질 리사이징 사용
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    
    # 명암 정규화: 최소값을 0, 최대값을 255로 늘림 (콘트라스트 강화)
    a_min, a_max = arr.min(), arr.max()
    if a_max > a_min:
        arr = (arr - a_min) / (a_max - a_min) * 255.0
    
    # 디버깅용 저장
    if debug_idx is not None:
        debug_img = Image.fromarray(arr.astype(np.uint8))
        debug_img.save(f"debug_char_{debug_idx}.png")

    arr = arr / 255.0
    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    pred = model.predict(arr, verbose=0)
    label = CLASSES[np.argmax(pred)]
    confidence = np.max(pred)
    
    input_name = img_input if isinstance(img_input, str) else f"char_{debug_idx}"
    print(f"{input_name}: {label} ({confidence:.2f})")
    return SYMBOL_MAP[label]


def segment_characters(img_path):
    print(f"이미지 분할 중: {img_path}")
    img = Image.open(img_path).convert("L")
    arr = np.array(img)

    # 배경색 자동 감지 및 반전 (학습 데이터는 흰색 배경에 검은색 글씨)
    if np.mean(arr) < 127:
        print("어두운 배경 감지: 이미지를 반전합니다.")
        arr = 255 - arr

    # 이진화
    thresh = 127
    binary = (arr < thresh).astype(np.int32)

    # 연결된 객체 찾기
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return []

    objects = ndimage.find_objects(labeled)
    bboxes = []
    for obj in objects:
        y_slice, x_slice = obj
        bboxes.append(
            [
                x_slice.start,
                y_slice.start,
                x_slice.stop - x_slice.start,
                y_slice.stop - y_slice.start,
            ]
        )

    # x좌표 순으로 정렬
    bboxes.sort(key=lambda b: b[0])

    # 가로로 겹치거나 아주 가까운 박스 합치기
    merged = []
    if bboxes:
        curr = bboxes[0]
        for i in range(1, len(bboxes)):
            nxt = bboxes[i]
            overlap = min(curr[0] + curr[2], nxt[0] + nxt[2]) - max(curr[0], nxt[0])
            dist = nxt[0] - (curr[0] + curr[2])

            if overlap > -10: 
                new_x = min(curr[0], nxt[0])
                new_y = min(curr[1], nxt[1])
                new_w = max(curr[0] + curr[2], nxt[0] + nxt[2]) - new_x
                new_h = max(curr[1] + curr[3], nxt[1] + nxt[3]) - new_y
                curr = [new_x, new_y, new_w, new_h]
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)

    # 노이즈 제거
    merged = [b for b in merged if b[2] > 5 and b[3] > 5]

    char_imgs = []
    for x, y, w, h in merged:
        char_arr = arr[y : y + h, x : x + w]
        char_img = Image.fromarray(char_arr)

        # 정사각형 패딩
        size = max(w, h)
        padded = Image.new("L", (size, size), 255)
        padded.paste(char_img, ((size - w) // 2, (size - h) // 2))

        # 여백 추가 (0.3로 원상복구)
        margin = int(size * 0.3)
        final_size = size + margin * 2
        final_img = Image.new("L", (final_size, final_size), 255)
        final_img.paste(padded, (margin, margin))

        char_imgs.append(final_img)

    print(f"총 {len(char_imgs)}개의 기호를 찾았습니다.")
    return char_imgs


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

    # 기호들을 하나로 합치기
    left_expr = "".join(str(s) for s in left)
    right_expr = "".join(str(s) for s in right)

    try:
        left_val = eval(left_expr)
        right_val = eval(right_expr)
        print(f"좌변: {left_expr} = {left_val}")
        print(f"우변: {right_expr} = {right_val}")
        
        if abs(left_val - right_val) < 1e-7:
            print("✅ 맞아요!")
        else:
            print("❌ 틀렸어요!")
    except Exception as e:
        print(f"계산 오류: {e}")


# 테스트: 기호 이미지 파일들을 순서대로 넣거나, 전체 수식 이미지 하나만 넣으면 됨
# 예시 1 (개별): python verify.py 3.png plus.png 5.png equal.png 8.png
# 예시 2 (전체): python verify.py equation.jpg
if __name__ == "__main__":
    image_files = sys.argv[1:]
    if not image_files:
        print("사용법: python verify.py <이미지 파일들...>")
        sys.exit(1)

    if len(image_files) == 1:
        chars = segment_characters(image_files[0])
        if chars:
            # i를 넘겨서 디버깅용 파일명을 구분함
            symbols = [predict_symbol(c, i) for i, c in enumerate(chars)]
        else:
            print("이미지에서 기호를 찾을 수 없습니다.")
            sys.exit(1)
    else:
        symbols = [predict_symbol(f) for f in image_files]
    
    print(f"인식된 기호: {symbols}")
    verify_equation(symbols)
