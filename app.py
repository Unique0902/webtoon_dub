from flask import Flask, render_template, request, jsonify
import os
import base64
from PIL import Image
import io
import pytesseract
from gtts import gTTS
import time
import pyscreenshot as ImageGrab
import cv2
import numpy as np
import pygame
import re
from io import BytesIO
import traceback

app = Flask(__name__)

# Tesseract OCR 경로 설정 (Windows 사용자는 설치 경로에 맞게 수정)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# pygame 초기화
pygame.mixer.init()

# 저장 디렉토리 생성
os.makedirs('static/audio', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# 말풍선 감지 함수
def detect_speech_bubbles(image):
    # OpenCV 형식으로 변환
    if isinstance(image, Image.Image):
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    # 원본 이미지 저장
    timestamp = int(time.time())
    original_path = f'static/images/bubble_original_{timestamp}.png'
    cv2.imwrite(original_path, img)
    
    # 이미지 크기 조정 (해상도 증가)
    height, width = img.shape[:2]
    img = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    
    # 여러 방법으로 말풍선 감지 시도
    bubbles = []
    
    # 방법 1: 적응형 이진화 (말풍선은 일반적으로 흰색)
    thresh1 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 방법 2: Otsu 이진화
    _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 방법 3: 고정 임계값 이진화 (밝은 영역 감지)
    _, thresh3 = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # 각 이진화 결과 저장
    cv2.imwrite(f'static/images/thresh1_{timestamp}.png', thresh1)
    cv2.imwrite(f'static/images/thresh2_{timestamp}.png', thresh2)
    cv2.imwrite(f'static/images/thresh3_{timestamp}.png', thresh3)
    
    # 각 이진화 방법에 대해 말풍선 감지 시도
    for i, thresh in enumerate([thresh1, thresh2, thresh3]):
        # 모폴로지 연산으로 노이즈 제거 및 말풍선 영역 강화
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 말풍선 내부를 채우기 위한 닫힘 연산
        kernel_close = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 디버깅용 이미지
        debug_img = img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f'static/images/contours{i+1}_{timestamp}.png', debug_img)
        
        # 말풍선 후보 필터링
        for contour in contours:
            # 면적이 너무 작은 윤곽선 제외
            area = cv2.contourArea(contour)
            if area < 1000:  # 최소 면적 기준
                continue
            
            # 윤곽선의 경계 상자 가져오기
            x, y, w, h = cv2.boundingRect(contour)
            
            # 너무 넓은 영역 제외 (전체 이미지의 80% 이상)
            img_area = img.shape[0] * img.shape[1]
            if area > 0.8 * img_area:
                continue
            
            # 종횡비가 극단적인 경우 제외
            aspect_ratio = w / float(h)
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue
            
            # 중복 방지 (이미 추가된 말풍선과 겹치는 경우 제외)
            is_duplicate = False
            for bx, by, bw, bh in bubbles:
                # 두 사각형의 중심점 계산
                center1_x, center1_y = x + w/2, y + h/2
                center2_x, center2_y = bx + bw/2, by + bh/2
                
                # 중심점 간의 거리 계산
                distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
                
                # 거리가 작으면 중복으로 간주
                if distance < (w + bw) / 4:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                bubbles.append((x, y, w, h))
    
    # 말풍선이 감지되지 않은 경우 추가 시도
    if not bubbles:
        print("기본 방법으로 말풍선이 감지되지 않아 추가 방법 시도")
        
        # 방법 4: 엣지 감지 기반 접근
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        cv2.imwrite(f'static/images/edges_{timestamp}.png', edges)
        cv2.imwrite(f'static/images/dilated_{timestamp}.png', dilated)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 디버깅용 이미지
        debug_img = img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f'static/images/edge_contours_{timestamp}.png', debug_img)
        
        # 말풍선 후보 필터링
        for contour in contours:
            # 면적이 너무 작은 윤곽선 제외
            area = cv2.contourArea(contour)
            if area < 1000:  # 최소 면적 기준
                continue
            
            # 윤곽선의 경계 상자 가져오기
            x, y, w, h = cv2.boundingRect(contour)
            
            # 너무 넓은 영역 제외 (전체 이미지의 80% 이상)
            img_area = img.shape[0] * img.shape[1]
            if area > 0.8 * img_area:
                continue
            
            # 종횡비가 극단적인 경우 제외
            aspect_ratio = w / float(h)
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue
            
            bubbles.append((x, y, w, h))
    
    # 말풍선이 여전히 감지되지 않은 경우 텍스트 영역 기반 접근
    if not bubbles:
        print("엣지 기반 방법으로도 말풍선이 감지되지 않아 텍스트 영역 기반 접근 시도")
        
        # MSER(Maximally Stable Extremal Regions) 사용하여 텍스트 영역 감지
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(blurred)
        
        # 텍스트 영역을 포함하는 사각형 생성
        text_boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if w > 10 and h > 10 and w < img.shape[1]/2 and h < img.shape[0]/2:
                text_boxes.append((x, y, w, h))
        
        # 텍스트 영역 병합
        if text_boxes:
            # 텍스트 영역 시각화
            text_vis = img.copy()
            for x, y, w, h in text_boxes:
                cv2.rectangle(text_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.imwrite(f'static/images/text_regions_{timestamp}.png', text_vis)
            
            # 텍스트 영역을 기반으로 말풍선 추정
            # 가까운 텍스트 영역 그룹화
            groups = []
            for box in text_boxes:
                x, y, w, h = box
                added = False
                for i, group in enumerate(groups):
                    for gx, gy, gw, gh in group:
                        # 두 박스가 가까운지 확인
                        if (abs((x + w/2) - (gx + gw/2)) < w + gw and 
                            abs((y + h/2) - (gy + gh/2)) < h + gh):
                            groups[i].append(box)
                            added = True
                            break
                    if added:
                        break
                if not added:
                    groups.append([box])
            
            # 각 그룹에 대해 경계 상자 생성
            for group in groups:
                if len(group) >= 2:  # 최소 2개 이상의 텍스트 영역이 있는 경우만 고려
                    min_x = min(x for x, _, _, _ in group)
                    min_y = min(y for _, y, _, _ in group)
                    max_x = max(x + w for x, _, w, _ in group)
                    max_y = max(y + h for _, y, _, h in group)
                    
                    # 여백 추가
                    padding = 20
                    min_x = max(0, min_x - padding)
                    min_y = max(0, min_y - padding)
                    max_x = min(img.shape[1], max_x + padding)
                    max_y = min(img.shape[0], max_y + padding)
                    
                    bubbles.append((min_x, min_y, max_x - min_x, max_y - min_y))
    
    # 말풍선이 여전히 감지되지 않은 경우 이미지를 몇 개의 영역으로 나누어 처리
    if not bubbles:
        print("모든 방법으로 말풍선이 감지되지 않아 이미지를 영역으로 분할")
        
        # 이미지를 4개의 영역으로 분할
        h, w = img.shape[:2]
        regions = [
            (0, 0, w//2, h//2),           # 좌상단
            (w//2, 0, w//2, h//2),        # 우상단
            (0, h//2, w//2, h//2),        # 좌하단
            (w//2, h//2, w//2, h//2)      # 우하단
        ]
        
        # 각 영역에 대해 텍스트 감지 시도
        for x, y, rw, rh in regions:
            # 영역 추출
            roi = blurred[y:y+rh, x:x+rw]
            
            # 영역에 대해 이진화
            _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 텍스트 영역 감지
            roi_mser = cv2.MSER_create()
            roi_regions, _ = roi_mser.detectRegions(roi)
            
            # 텍스트 영역이 있는지 확인
            if len(roi_regions) > 5:  # 일정 개수 이상의 텍스트 영역이 있으면 말풍선으로 간주
                bubbles.append((x, y, rw, rh))
    
    # 말풍선 영역 이미지 추출
    bubble_images = []
    for x, y, w, h in bubbles:
        # 약간의 여백 추가
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        # 말풍선 영역 추출
        bubble_img = img[y1:y2, x1:x2]
        
        # PIL 이미지로 변환
        bubble_pil = Image.fromarray(cv2.cvtColor(bubble_img, cv2.COLOR_BGR2RGB))
        bubble_images.append(bubble_pil)
        
        # 디버깅용 개별 말풍선 저장
        cv2.imwrite(f'static/images/bubble_region_{timestamp}_{len(bubble_images)}.png', bubble_img)
    
    # 디버깅용 이미지 생성 (말풍선 표시)
    debug_img = img.copy()
    for i, (x, y, w, h) in enumerate(bubbles):
        # 말풍선 경계 표시
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # 말풍선 번호 표시
        cv2.putText(debug_img, str(i+1), (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite(f'static/images/bubbles_detected_{timestamp}.png', debug_img)
    
    # 원본 이미지 크기에 맞게 말풍선 좌표 조정
    scale_x = width / (width*2)
    scale_y = height / (height*2)
    
    adjusted_bubbles = []
    for x, y, w, h in bubbles:
        adjusted_bubbles.append((
            int(x * scale_x),
            int(y * scale_y),
            int(w * scale_x),
            int(h * scale_y)
        ))
    
    # 말풍선이 감지되지 않은 경우 전체 이미지를 하나의 말풍선으로 처리
    if not bubble_images:
        print("말풍선이 감지되지 않아 전체 이미지를 하나의 말풍선으로 처리")
        bubble_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))]
        adjusted_bubbles = [(0, 0, width, height)]
    
    return bubble_images, adjusted_bubbles

# 이미지 전처리 함수
def preprocess_image(image):
    # PIL 이미지를 OpenCV 형식으로 변환
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 원본 이미지 저장
    timestamp = int(time.time())
    original_path = f'static/images/original_{timestamp}.png'
    cv2.imwrite(original_path, img)
    
    # 이미지 크기 조정 (해상도 증가)
    height, width = img.shape[:2]
    img = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 웹툰 특화 전처리 시작
    processed_images = []
    
    # 1. 기본 전처리 - 노이즈 제거 및 대비 향상
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    enhanced = cv2.equalizeHist(denoised)
    
    # 2. 적응형 이진화 (여러 파라미터로 시도)
    binary1 = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    binary2 = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 5
    )
    
    # 3. Otsu 이진화
    _, binary3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. 웹툰 말풍선 특화 처리 - 경계 강화
    edges = cv2.Canny(enhanced, 100, 200)
    dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    binary4 = cv2.adaptiveThreshold(
        cv2.addWeighted(enhanced, 0.8, cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)[:,:,0], 0.2, 0),
        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 5. 텍스트 영역 강화 (웹툰 말풍선 내 텍스트 특화)
    kernel_close = np.ones((3, 3), np.uint8)
    binary5 = cv2.morphologyEx(binary3, cv2.MORPH_CLOSE, kernel_close)
    
    # 6. 추가 처리 - 특정 웹툰 스타일에 맞춘 처리
    # 대비 강화 후 이진화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_clahe = clahe.apply(denoised)
    _, binary6 = cv2.threshold(enhanced_clahe, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 7. 가우시안 블러 적용 후 이진화
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    _, binary7 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 8. 반전 이미지 처리
    inverted = cv2.bitwise_not(gray)
    _, binary8 = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 모든 이진화 결과 저장
    cv2.imwrite(f'static/images/binary1_{timestamp}.png', binary1)
    cv2.imwrite(f'static/images/binary2_{timestamp}.png', binary2)
    cv2.imwrite(f'static/images/binary3_{timestamp}.png', binary3)
    cv2.imwrite(f'static/images/binary4_{timestamp}.png', binary4)
    cv2.imwrite(f'static/images/binary5_{timestamp}.png', binary5)
    cv2.imwrite(f'static/images/binary6_{timestamp}.png', binary6)
    cv2.imwrite(f'static/images/binary7_{timestamp}.png', binary7)
    cv2.imwrite(f'static/images/binary8_{timestamp}.png', binary8)
    
    # PIL 이미지로 변환
    result1 = Image.fromarray(cv2.cvtColor(binary1, cv2.COLOR_GRAY2RGB))
    result2 = Image.fromarray(cv2.cvtColor(binary2, cv2.COLOR_GRAY2RGB))
    result3 = Image.fromarray(cv2.cvtColor(binary3, cv2.COLOR_GRAY2RGB))
    result4 = Image.fromarray(cv2.cvtColor(binary4, cv2.COLOR_GRAY2RGB))
    result5 = Image.fromarray(cv2.cvtColor(binary5, cv2.COLOR_GRAY2RGB))
    result6 = Image.fromarray(cv2.cvtColor(binary6, cv2.COLOR_GRAY2RGB))
    result7 = Image.fromarray(cv2.cvtColor(binary7, cv2.COLOR_GRAY2RGB))
    result8 = Image.fromarray(cv2.cvtColor(binary8, cv2.COLOR_GRAY2RGB))
    
    # 원본 이미지도 PIL로 변환하여 추가
    original_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inverted_pil = Image.fromarray(cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB))
    
    # 결과 이미지 저장
    result1.save(f'static/images/preprocessed1_{timestamp}.png')
    result2.save(f'static/images/preprocessed2_{timestamp}.png')
    result3.save(f'static/images/preprocessed3_{timestamp}.png')
    result4.save(f'static/images/preprocessed4_{timestamp}.png')
    result5.save(f'static/images/preprocessed5_{timestamp}.png')
    result6.save(f'static/images/preprocessed6_{timestamp}.png')
    result7.save(f'static/images/preprocessed7_{timestamp}.png')
    result8.save(f'static/images/preprocessed8_{timestamp}.png')
    
    # 모든 처리된 이미지 반환
    return [result1, result2, result3, result4, result5, result6, result7, result8, original_pil, inverted_pil]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_screen', methods=['POST'])
def capture_screen():
    try:
        # 화면 캡처
        screenshot = ImageGrab.grab()
        
        # 이미지 저장
        timestamp = int(time.time())
        image_path = f'static/images/screenshot_{timestamp}.png'
        screenshot.save(image_path)
        
        return jsonify({
            'success': True,
            'image_path': image_path
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.json
        image_path = data['image_path']
        
        # 이미지 로드
        image = Image.open(image_path)
        
        # 이미지 전처리
        processed_images = preprocess_image(image)
        
        # 다양한 OCR 설정으로 텍스트 추출 시도
        texts = []
        configs = [
            '--oem 3 --psm 6 -l kor+eng',  # 기본 설정
            '--oem 3 --psm 4 -l kor+eng',  # 단일 텍스트 블록 감지
            '--oem 3 --psm 11 -l kor+eng', # 텍스트 블록 없이 단일 텍스트 라인
            '--oem 1 --psm 6 -l kor+eng',  # 신경망 LSTM OCR 엔진
            '--oem 1 --psm 4 -l kor+eng',  # 신경망 + 단일 텍스트 블록
            '--oem 0 --psm 6 -l kor+eng',  # 레거시 Tesseract OCR 엔진
            '--oem 3 --psm 6 -c preserve_interword_spaces=1 -l kor+eng', # 단어 간격 보존
            '--oem 3 --psm 3 -l kor+eng',  # 자동 페이지 세그먼테이션
            '--oem 3 --psm 7 -l kor+eng',  # 단일 텍스트 라인 (세로 정렬)
            '--oem 3 --psm 13 -l kor+eng', # 원시 라인 (세로 정렬)
        ]
        
        # 각 이미지와 설정 조합으로 OCR 시도
        for i, img in enumerate(processed_images):
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img, config=config)
                    if text.strip():
                        # 텍스트 후처리
                        cleaned = text.strip()
                        # 줄바꿈 정리
                        cleaned = re.sub(r'\s+', ' ', cleaned)
                        # 특수문자 정리 (한글, 영문, 숫자, 기본 문장부호 유지)
                        cleaned = re.sub(r'[^\w\s\.\,\?\!\"\'\:\;\-\(\)가-힣]', '', cleaned)
                        texts.append(cleaned)
                except Exception as e:
                    print(f"OCR 오류 (이미지 {i}, 설정 {config}): {str(e)}")
                    continue
        
        # 결과가 없으면 빈 배열 반환
        if not texts:
            return jsonify({'error': 'No text detected'}), 400
        
        # 가장 긴 텍스트 결과 선택 (일반적으로 더 많은 정보 포함)
        longest_text = max(texts, key=len)
        
        # 타임스탬프 생성
        timestamp = int(time.time())
        
        # 오디오 생성
        audio_paths = []
        for i, text in enumerate(texts[:3]):  # 상위 3개 결과만 사용
            try:
                tts = gTTS(text=text, lang='ko')
                audio_path = f'static/audio/speech_{timestamp}_{i}.mp3'
                tts.save(audio_path)
                audio_paths.append(audio_path)
            except Exception as e:
                print(f"TTS 오류: {str(e)}")
                continue
        
        return jsonify({
            'texts': texts,
            'audio_paths': audio_paths
        })
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process_selected_area', methods=['POST'])
def process_selected_area():
    try:
        data = request.json
        image_data = data['image_data']
        
        # base64 이미지 데이터 디코딩
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # 이미지 저장
        timestamp = int(time.time())
        image_path = f'static/images/selected_{timestamp}.png'
        
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        # 이미지 로드 및 전처리
        image = Image.open(BytesIO(image_bytes))
        processed_images = preprocess_image(image)
        
        # 다양한 OCR 설정으로 텍스트 추출 시도
        texts = []
        configs = [
            '--oem 3 --psm 6 -l kor+eng',  # 기본 설정
            '--oem 3 --psm 4 -l kor+eng',  # 단일 텍스트 블록 감지
            '--oem 3 --psm 11 -l kor+eng', # 텍스트 블록 없이 단일 텍스트 라인
            '--oem 1 --psm 6 -l kor+eng',  # 신경망 LSTM OCR 엔진
            '--oem 1 --psm 4 -l kor+eng',  # 신경망 + 단일 텍스트 블록
            '--oem 0 --psm 6 -l kor+eng',  # 레거시 Tesseract OCR 엔진
            '--oem 3 --psm 6 -c preserve_interword_spaces=1 -l kor+eng', # 단어 간격 보존
            '--oem 3 --psm 3 -l kor+eng',  # 자동 페이지 세그먼테이션
            '--oem 3 --psm 7 -l kor+eng',  # 단일 텍스트 라인 (세로 정렬)
            '--oem 3 --psm 13 -l kor+eng', # 원시 라인 (세로 정렬)
        ]
        
        # 각 이미지와 설정 조합으로 OCR 시도
        for i, img in enumerate(processed_images):
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img, config=config)
                    if text.strip():
                        # 텍스트 후처리
                        cleaned = text.strip()
                        # 줄바꿈 정리
                        cleaned = re.sub(r'\s+', ' ', cleaned)
                        # 특수문자 정리 (한글, 영문, 숫자, 기본 문장부호 유지)
                        cleaned = re.sub(r'[^\w\s\.\,\?\!\"\'\:\;\-\(\)가-힣]', '', cleaned)
                        texts.append(cleaned)
                except Exception as e:
                    print(f"OCR 오류 (이미지 {i}, 설정 {config}): {str(e)}")
                    continue
        
        # 결과가 없으면 빈 배열 반환
        if not texts:
            return jsonify({'error': 'No text detected'}), 400
        
        # 가장 긴 텍스트 결과 선택 (일반적으로 더 많은 정보 포함)
        longest_text = max(texts, key=len)
        
        # 오디오 생성
        audio_paths = []
        for i, text in enumerate(texts[:3]):  # 상위 3개 결과만 사용
            try:
                tts = gTTS(text=text, lang='ko')
                audio_path = f'static/audio/speech_{timestamp}_{i}.mp3'
                tts.save(audio_path)
                audio_paths.append(audio_path)
            except Exception as e:
                print(f"TTS 오류: {str(e)}")
                continue
        
        return jsonify({
            'texts': texts,
            'audio_paths': audio_paths
        })
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Flask 서버 실행
if __name__ == '__main__':
    app.run(debug=True)