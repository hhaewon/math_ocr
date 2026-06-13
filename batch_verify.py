import os
import glob
from verify import segment_characters, predict_symbol, verify_equation

# 대상 디렉토리
TARGET_DIR = "equation_images"

# 지원하는 이미지 확장자
EXTENSIONS = ["*.jpg", "*.png", "*.jpeg"]


def main():
    # 1. 이미지 파일 리스트 확보 (하위 폴더 포함)
    image_files = []
    for ext in EXTENSIONS:
        # recursive=True를 사용하여 하위 폴더까지 탐색
        image_files.extend(
            glob.glob(os.path.join(TARGET_DIR, "**", ext), recursive=True)
        )

    # 파일 이름순으로 정렬
    image_files.sort()

    if not image_files:
        print(f"'{TARGET_DIR}' 폴더에서 이미지를 찾을 수 없습니다.")
        return

    print(
        f"🔍 총 {len(image_files)}개의 이미지를 발견했습니다. 배치 처리를 시작합니다.\n"
    )

    for idx, img_path in enumerate(image_files):
        print(f"[{idx + 1}/{len(image_files)}] 📄 처리 중: {img_path}")
        print("-" * 50)

        try:
            # verify.py의 로직을 그대로 사용
            # 1. 이미지 분할
            chars = segment_characters(img_path)

            if chars:
                # 2. 각 기호 인식 (i는 디버깅 파일명 구분용)
                symbols = [predict_symbol(c, f"{idx}_{i}") for i, c in enumerate(chars)]

                # 3. 수식 검증 및 결과 출력
                verify_equation(symbols)
            else:
                print("  ❌ 이미지에서 기호를 찾을 수 없습니다.")

        except Exception as e:
            print(f"  💥 오류 발생: {e}")

        print("-" * 50)
        print("\n")

    print("✅ 모든 이미지 처리가 완료되었습니다.")


if __name__ == "__main__":
    main()
