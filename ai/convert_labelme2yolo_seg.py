'''
How to use:
    1. 학습시킬 Image 가져오기 ("jpg", "jpeg", "png", "bmp", "tiff")

    2. Labelme로 라벨링한 JSON 파일 가져오기

    3. 설정 Section에서 경로 및 비율 설정
        1) JSON 파일과 이미지 파일이 있는 폴더를 설정    
            - input_base_dir : Labelme JSON 파일과 이미지가 있는 폴더 경로
            - input_label_dir : Labelme JSON 파일이 있는 폴더 경로
            - input_image_dir : 이미지 파일이 있는 폴더 경로

        2) YOLO 데이터셋을 생성할 폴더 경로 설정
            - output_yolo_dataset_dir : YOLO 데이터셋을 생성할 폴더 경로

        3) train, val, test 비율 설정 (합계가 1이 되어야 합니다)

    4. 생성된 폴더와 yaml 파일 확인 후 아래의 예시를 보고 모델 학습
        yolo task=segment mode=train model=yolov8m-seg.pt data=./data.yaml epoch=100 imgsz=640 project="./runs/train" name="seg_exp1"
    
    5. YOLOv8 모델 학습 후, 생성된 best.pt 모델로 yolo_segmentor.py 의 모델 변경하여 프로그램 실행하여 테스트
'''

import os
import json
import shutil
import random
from glob import glob

####################################### 설정 Section #######################################
# 라벨링된 JSON 파일과 해당 이미지가 있는 폴더 경로 (절대 경로 또는 스크립트 실행 위치 기준 상대 경로)
input_base_dir = r"ai" # 실제 경로로 변경해주세요
input_label_dir = os.path.join(input_base_dir, "Labels\Labelme") #base_dir 안에 Labels 폴더가 있어야합니다.
input_image_dir = os.path.join(input_base_dir, "Images") #base_dir 안에 Images 폴더가 있어야합니다.

# 생성될 YOLO 데이터셋의 기본 출력 폴더 (이 폴더 안에 images, labels, dataset.yaml이 생성됩니다)
output_yolo_dataset_dir = r"ai\YOLO_Dataset" # 원하는 출력 경로로 변경해주세요

# 학습, 검증, 테스트 데이터셋 비율 (합계가 1이 되어야 합니다)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
####################################### 설정 Section #######################################
image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"] # 이미지 파일 확장자 (필요에 따라 추가/변경)

# 비율 합계 검사
if not (train_ratio + val_ratio + test_ratio == 1.0):
    raise ValueError("train, val, test 비율의 합계가 1.0이 되어야 합니다.")

if os.path.exists(output_yolo_dataset_dir):
    print(f"기존 YOLO 데이터셋 폴더 '{output_yolo_dataset_dir}'를 삭제합니다.")
    shutil.rmtree(output_yolo_dataset_dir)
    print("삭제 완료.")
    
# YOLO 데이터셋 폴더 구조 생성
yolo_images_dir = os.path.join(output_yolo_dataset_dir, "images")
yolo_labels_dir = os.path.join(output_yolo_dataset_dir, "labels")

for sub_dir in ["train", "val", "test"]:
    os.makedirs(os.path.join(yolo_images_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(yolo_labels_dir, sub_dir), exist_ok=True)

def convert_polygon_to_yolo(points, img_w, img_h):
    """
    Polygon 포인트를 YOLO 포맷으로 변환합니다.
    """
    norm_points = []
    for x, y in points:
        norm_points.append(x / img_w)
        norm_points.append(y / img_h)
    return norm_points

# 클래스 이름을 정수 ID로 매핑
label2id = {}
current_label_id = 0

# 모든 JSON 파일 목록 가져오기
json_files = glob(os.path.join(input_label_dir, "*.json"))

# 이미지 파일과 JSON 파일을 매칭
data_pairs = []
for json_path in json_files:
    base_name_json = os.path.splitext(os.path.basename(json_path))[0]
    found_image = False
    for ext in image_extensions:
        image_path = os.path.join(input_image_dir, f"{base_name_json}.{ext}")
        if os.path.exists(image_path):
            data_pairs.append((json_path, image_path))
            found_image = True
            break
    if not found_image:
        print(f"경고: {json_path}에 해당하는 이미지 파일을 찾을 수 없습니다. (지원 확장자: {', '.join(image_extensions)})")

if not data_pairs:
    print("처리할 JSON-이미지 쌍을 찾을 수 없습니다. 입력 경로를 확인해주세요.")
    exit()

# 데이터를 랜덤하게 섞고 train, val, test로 분할
random.shuffle(data_pairs)
total_data = len(data_pairs)

train_split = int(total_data * train_ratio)
val_split = int(total_data * val_ratio)

train_data = data_pairs[:train_split]
val_data = data_pairs[train_split : train_split + val_split]
test_data = data_pairs[train_split + val_split :]

print(f"총 데이터 수: {total_data}")
print(f"Train 데이터 수: {len(train_data)}")
print(f"Val 데이터 수: {len(val_data)}")
print(f"Test 데이터 수: {len(test_data)}")

# 데이터셋 처리 및 파일 복사/생성
def process_data_split(data_list, split_name):
    global current_label_id # 함수 내에서 전역 변수 변경을 위해 global 선언
    print(f"\n--- {split_name} 데이터 처리 중 ---")
    for json_path, image_path in data_list:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data["imageWidth"]
        img_h = data["imageHeight"]
        shapes = data["shapes"]

        yolo_lines = []
        for shape in shapes:
            if shape["shape_type"] != "polygon":
                continue # Polygon 형태만 처리

            label = shape["label"]
            if label not in label2id:
                label2id[label] = current_label_id
                current_label_id += 1

            class_id = label2id[label]
            norm_points = convert_polygon_to_yolo(shape["points"], img_w, img_h)

            # YOLO format: class_id x1 y1 x2 y2 ... xn yn (x,y는 정규화된 값)
            line = f"{class_id} " + " ".join([f"{p:.6f}" for p in norm_points])
            yolo_lines.append(line)

        # 이미지 파일과 .txt 라벨 파일의 기본 이름 (확장자 제외)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 이미지 파일 복사
        dest_image_path = os.path.join(yolo_images_dir, split_name, os.path.basename(image_path))
        shutil.copyfile(image_path, dest_image_path)

        # .txt 라벨 파일 저장
        txt_path = os.path.join(yolo_labels_dir, split_name, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

# 각 데이터셋 분할 처리
process_data_split(train_data, "train")
process_data_split(val_data, "val")
process_data_split(test_data, "test") # test_data가 비어 있을 수 있습니다.

# dataset.yaml 파일 생성
dataset_yaml_path = os.path.join(output_yolo_dataset_dir, "dataset.yaml")

names_dict = {id: name for name, id in sorted(label2id.items(), key=lambda item: item[1])}

# names:
#   index : name 형식으로 변환
names_yaml_block = "names:\n" + "\n".join([f"  {i}: {name}" for i, name in names_dict.items()])

dataset_yaml_content = f"""
path: {os.path.abspath(output_yolo_dataset_dir)}
train: images/train
val: images/val
test: images/test

{names_yaml_block}
"""

with open(dataset_yaml_path, "w", encoding="utf-8") as f:
    f.write(dataset_yaml_content)

print(f"\n변환 완료! YOLO 데이터셋이 여기에 저장됨:\n{output_yolo_dataset_dir}")
print(f"dataset.yaml 파일이 생성되었습니다:\n{dataset_yaml_path}")
print("\n생성된 클래스 ID 매핑:")
for label, id in sorted(label2id.items(), key=lambda item: item[1]):
    print(f"  {label}: {id}")