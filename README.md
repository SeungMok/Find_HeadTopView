# 필수 패키지 설치
pip install -U pip
pip install pyqt5 opencv-python numpy ultralytics labelme matplotlib
  ※주의 labelme를 사용하려면 python 3.11 이하 버전으로 해야 함

# GUI 실행
python main.py

# 실행 방법
- 프로그램이 실행되면 이미지 로딩, 회전, 뒤집기, 외곽선 보기(Contours) 기능을 사용할 수 있습니다.
- 이미지는 마우스로 드래그 앤 드롭하여 로딩 가능합니다.
- Contours 버튼을 누르면 외곽선 분석 모드로 전환되며, Flip/Rotate 버튼은 비활성화됩니다.
- Cancel을 누르면 원본 이미지로 되돌아갑니다.
- Contours 결과가 나온 이미지는 이동, 확대/축소가 가능
  - ctrl + 마우스 휠 : 확대/축소
  - ctrl + 드래그 : 이동
