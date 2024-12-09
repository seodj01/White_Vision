from flask import Flask, request, render_template_string
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from io import BytesIO  # 메모리 내 이미지 처리를 위한 모듈
import base64  # Base64 인코딩을 위한 모듈
from lightweight_deeplabv3 import MobileNetV3DeepLabV3  # 모델 클래스 import
import torch.nn.functional as F  # 소프트맥스 계산

# Flask 애플리케이션 생성
app = Flask(__name__)

# 학습된 모델 불러오기
model = MobileNetV3DeepLabV3(num_classes=8)
model.load_state_dict(torch.load("./best_checkpoint/batch_256_best.pth", map_location="cpu")["model_state_dict"])
model.eval()

# 색상 매핑 (변경된 매핑)
color_map = {
    "sidewalk_braille": (255, 255, 0),     # 인도 + 점자블록 - 노란색
    "roadway_alley_bike_caution_zone": (255, 0, 0),  # 로드웨이 + 골목길 + 자전거도로 + 위험구역 - 빨간색
    "cross_walk": (135, 206, 250),         # 횡단보도 - 하늘색
}

# 클래스 묶기
class_group = {
    "sidewalk_braille": [1, 2],       # 인도, 점자블록
    "roadway_alley_bike_caution_zone": [3, 4, 5, 7],  # 로드웨이, 골목길, 자전거도로, 위험구역
    "cross_walk": [6]                # 횡단보도
}

# 이미지 전처리 함수
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 리사이즈
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
    ])
    tensor = preprocess(image).unsqueeze(0)  # 배치 차원 추가
    return tensor

# 전처리 결과를 시각화하는 함수
def tensor_to_image(tensor):
    # Normalize 해제 (정규화 역연산)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
    tensor = (tensor * std) + mean  # 역정규화
    tensor = np.clip(tensor, 0, 1)  # 0~1로 클리핑

    # 다시 PIL 이미지로 변환
    return Image.fromarray((tensor * 255).astype(np.uint8))

# generate_segmentation_image 함수 수정
def generate_segmentation_image(output, threshold=0.5):
    probabilities = F.softmax(output, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)

    max_probabilities = np.max(probabilities, axis=0)  # (H, W)
    predicted_classes = np.argmax(probabilities, axis=0)  # (H, W)

    height, width = predicted_classes.shape
    segmented_image = np.zeros((height, width, 3), dtype=np.uint8)

    for group_name, classes in class_group.items():
        color = color_map[group_name]
        mask = np.isin(predicted_classes, classes) & (max_probabilities >= threshold)
        segmented_image[mask] = color

    # 임계값을 넘지 못한 픽셀은 흰색으로 설정
    segmented_image[max_probabilities < threshold] = (255, 255, 255)

    return Image.fromarray(segmented_image)

# 메인 페이지
@app.route('/')
def index():
    return '''
        <!doctype html>
        <title>Image Segmentation</title>
        <h1>이미지 업로드</h1>
        <form method="POST" action="/segment" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="업로드">
        </form>
    '''

# 세그멘테이션 엔드포인트
@app.route('/segment', methods=['POST'])
def segment():
    # 업로드된 파일 가져오기
    file = request.files['file']
    if file:
        # 원본 이미지를 메모리에서 처리
        image = Image.open(file.stream).convert('RGB')
        input_tensor = preprocess_image(image)

        # 전처리된 이미지를 확인하기 위한 작업
        preprocessed_image = tensor_to_image(input_tensor[0])

        # 모델 추론
        with torch.no_grad():
            output = model(input_tensor)

        # 세그멘테이션 결과 이미지 생성
        segmented_image = generate_segmentation_image(output)

        # 이미지를 메모리에 저장 및 Base64로 변환 (원본 이미지)
        original_img_io = BytesIO()
        image.save(original_img_io, 'PNG')
        original_img_io.seek(0)
        original_base64 = base64.b64encode(original_img_io.getvalue()).decode('utf-8')

        # 전처리된 이미지도 Base64로 변환
        preprocessed_img_io = BytesIO()
        preprocessed_image.save(preprocessed_img_io, 'PNG')
        preprocessed_img_io.seek(0)
        preprocessed_base64 = base64.b64encode(preprocessed_img_io.getvalue()).decode('utf-8')

        # 세그멘테이션 결과 이미지도 Base64로 변환
        segmented_img_io = BytesIO()
        segmented_image.save(segmented_img_io, 'PNG')
        segmented_img_io.seek(0)
        segmented_base64 = base64.b64encode(segmented_img_io.getvalue()).decode('utf-8')

        # HTML 페이지에 이미지 출력
        html_content = f'''
            <!doctype html>
            <title>Segmentation Result</title>
            <h1>세그멘테이션 결과</h1>
            <div style="display: flex; gap: 20px;">
                <div>
                    <h3>원본 이미지</h3>
                    <img src="data:image/png;base64,{original_base64}">
                </div>
                <div>
                    <h3>전처리된 이미지</h3>
                    <img src="data:image/png;base64,{preprocessed_base64}">
                </div>
                <div>
                    <h3>세그멘테이션 결과</h3>
                    <img src="data:image/png;base64,{segmented_base64}">
                </div>
            </div>
            <br><a href="/">다시 업로드</a>
        '''
        return render_template_string(html_content)

    return "파일 업로드 실패"

# Flask 실행
if __name__ == '__main__':
    app.run(debug=True)
