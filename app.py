import os
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import verify

app = Flask(__name__)
CORS(app)

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # 이미지 저장
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        # 1. 원본 이미지 base64 변환 (화면 표시용)
        with open(file_path, "rb") as f:
            original_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # 2. 기호 분할
        chars = verify.segment_characters(file_path)
        if not chars:
            return jsonify({'error': 'Could not segment characters'}), 400
        
        results = []
        symbols = []
        
        # 3. 각 기호 인식
        for i, char_img in enumerate(chars):
            # 인식 결과
            symbol = verify.predict_symbol(char_img, i)
            symbols.append(symbol)
            
            # 전처리된 이미지 base64 (화면 표시용)
            processed_arr = verify.preprocess_image(char_img) * 255.0
            disp_char = Image.fromarray(processed_arr.astype(np.uint8))
            char_base64 = pil_to_base64(disp_char)
            
            results.append({
                'symbol': str(symbol),
                'image': char_base64
            })
            
        # 4. 수식 검증 (verify.py의 공통 로직 사용)
        verification = verify.get_verification_result(symbols)
        
        # UI에서 사용하기 위해 숫자 값을 문자열로 변환 (필요시)
        if verification['left_val'] is not None:
            verification['left_val'] = str(verification['left_val'])
        if verification['right_val'] is not None:
            verification['right_val'] = str(verification['right_val'])
        
        return jsonify({
            'original_image': original_base64,
            'characters': results,
            'verification': verification
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
