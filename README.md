# 웹툰 더빙 프로그램

화면에 표시된 웹툰의 대사를 자동으로 추출하고 음성으로 변환하여 더빙해주는 웹 애플리케이션입니다.

## 주요 기능

- 전체 화면 캡처 또는 선택 영역 캡처
- 캡처된 이미지에서 텍스트(대사) 추출
- 추출된 텍스트를 한국어 음성으로 변환
- 생성된 음성 재생 및 다운로드

## 설치 방법

1. 필요한 패키지 설치:
```
pip install -r requirements.txt
```

2. Tesseract OCR 설치:
   - Windows: 
     1. https://github.com/UB-Mannheim/tesseract/wiki 에서 최신 버전 다운로드 (예: tesseract-ocr-w64-setup-5.3.3.20231005.exe)
     2. 설치 시 "Additional language data (download)" 옵션에서 "Korean" 선택
     3. 설치 경로를 기억해두세요 (기본값: C:\Program Files\Tesseract-OCR)
     4. 시스템 환경 변수 PATH에 Tesseract 설치 경로 추가 (제어판 > 시스템 > 고급 시스템 설정 > 환경 변수)
     5. 새로운 환경 변수 TESSDATA_PREFIX를 생성하고 값을 "C:\Program Files\Tesseract-OCR\tessdata"로 설정
   - macOS: `brew install tesseract tesseract-lang`
   - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-kor`

3. 설치 확인:
   - 명령 프롬프트(CMD)나 PowerShell에서 다음 명령 실행:
   ```
   tesseract --version
   ```
   - 버전 정보가 표시되면 설치가 성공적으로 완료된 것입니다.

4. Tesseract OCR 경로 설정:
   - Tesseract OCR이 설치되어 있지만 PATH에 추가되지 않은 경우, app.py 파일에서 다음 줄을 찾아 Tesseract 설치 경로를 올바르게 설정하세요:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```
   - 설치 경로가 다른 경우 위 경로를 실제 설치 경로로 수정하세요.

## 실행 방법

```
python app.py
```

웹 브라우저에서 `http://localhost:5000`으로 접속하여 애플리케이션을 사용할 수 있습니다.

## 사용 방법

1. "전체 화면 캡처" 버튼을 클릭하여 현재 화면을 캡처하거나, "영역 선택 캡처" 버튼을 클릭하여 특정 영역만 캡처합니다.
2. 캡처된 이미지가 표시되면 "텍스트 추출 및 더빙" 버튼을 클릭합니다.
3. 추출된 텍스트와 생성된 음성이 하단에 표시됩니다.
4. 음성 파일을 다운로드하여 저장할 수 있습니다.

## 웹툰 텍스트 인식 최적화 팁

1. **말풍선 영역만 선택하기**:
   - "영역 선택 캡처" 기능을 사용하여 말풍선 영역만 정확하게 선택하세요.
   - 배경 이미지나 다른 요소가 포함되면 텍스트 인식률이 떨어질 수 있습니다.

2. **고해상도 이미지 사용**:
   - 웹툰 이미지의 해상도가 높을수록 텍스트 인식률이 높아집니다.
   - 가능하면 웹툰 확대 후 캡처하세요.

3. **단순한 글꼴의 웹툰 선택**:
   - 장식이 많은 글꼴보다 단순한 글꼴이 더 잘 인식됩니다.
   - 특수 효과가 적용된 텍스트는 인식하기 어려울 수 있습니다.

4. **밝은 배경, 어두운 텍스트**:
   - 흰색 배경에 검은색 텍스트가 가장 잘 인식됩니다.
   - 색상 대비가 높을수록 인식률이 높아집니다.

5. **전처리된 이미지 확인**:
   - 텍스트 추출 후 `static/images` 폴더에서 `preprocessed_*.png` 파일을 확인하세요.
   - 전처리된 이미지에서 텍스트가 명확하게 보이면 인식률이 높아집니다.

## 기술 스택

- 백엔드: Flask (Python)
- 프론트엔드: HTML, CSS, JavaScript, jQuery, Bootstrap
- 이미지 처리: OpenCV, Pillow
- 텍스트 추출: Tesseract OCR
- 음성 합성: Google Text-to-Speech (gTTS)
- 음성 재생: Pygame

## 문제 해결

1. "tesseract is not installed or it's not in your PATH" 오류가 발생하는 경우:
   - Tesseract OCR을 설치하세요: https://github.com/UB-Mannheim/tesseract/wiki
   - 설치 후 컴퓨터를 재시작하세요.
   - 또는 app.py 파일에서 Tesseract 경로를 직접 설정하세요:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```
   - 설치 경로가 다른 경우 위 경로를 실제 설치 경로로 수정하세요.

2. 한국어 텍스트가 제대로 인식되지 않는 경우:
   - Tesseract OCR의 한국어 언어 데이터가 올바르게 설치되었는지 확인하세요.
   - 이미지의 품질과 해상도가 충분한지 확인하세요.
   - 텍스트가 명확하게 보이는 영역만 선택하여 캡처해 보세요.
   - 웹툰의 글꼴이 특이하거나 장식이 많은 경우 인식률이 떨어질 수 있습니다.
   - 말풍선 영역만 정확하게 선택하여 캡처하세요.

3. 음성이 재생되지 않는 경우:
   - 브라우저의 오디오 설정을 확인하세요.
   - 다른 브라우저로 시도해 보세요.

4. "ImportError: cannot import name 'url_quote' from 'werkzeug.urls'" 오류가 발생하는 경우:
   - Werkzeug 버전을 Flask 2.0.1과 호환되는 버전으로 다운그레이드하세요:
   ```
   pip install werkzeug==2.0.1
   ```
   - 또는 requirements.txt 파일에 werkzeug==2.0.1을 추가한 후 다시 설치하세요:
   ```
   pip install -r requirements.txt
   ```

## 주의사항

- 텍스트 추출 정확도는 웹툰의 글꼴, 배경 등에 따라 달라질 수 있습니다.
- 한국어 텍스트 추출을 위해 Tesseract OCR의 한국어 언어 데이터가 필요합니다.
- 음성 합성을 위해 인터넷 연결이 필요합니다 (gTTS는 Google의 온라인 서비스를 사용합니다). 