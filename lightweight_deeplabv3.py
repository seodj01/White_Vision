import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large

# Squeeze-and-Excitation 모듈 정의
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        se = F.adaptive_avg_pool2d(x, 1).view(b, c)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(b, c, 1, 1)
        return x * se

# Lightweight ASPP 모듈
class LightweightASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightweightASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv_out = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))

        # Global context
        x4 = self.global_avg_pool(x)
        x4 = F.relu(self.global_conv(x4))
        x4 = F.interpolate(x4, size=x1.size()[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_out(x)
        x = self.dropout(x)
        return x

# MobileNetV3 기반 DeepLabV3 모델 정의
class MobileNetV3DeepLabV3(nn.Module):
    def __init__(self, num_classes=8):
        super(MobileNetV3DeepLabV3, self).__init__()

        # MobileNetV3 Large 백본 정의
        mobilenet_v3_model = mobilenet_v3_large(pretrained=True)
        self.backbone = mobilenet_v3_model.features

        # 다운샘플링 비율을 줄이기 위해 마지막 단계의 stride 수정
        # 마지막 레이어가 Conv2dNormActivation이므로 직접 접근하여 stride를 변경합니다.
        for module in self.backbone[::-1]:
            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (1, 1):
                    module.stride = (1, 1)
                    break

        # ASPP 모듈
        self.aspp = LightweightASPP(in_channels=960, out_channels=256)

        # Squeeze-and-Excitation 모듈 추가
        self.se = SqueezeExcitation(256)

        # 최종 출력 레이어
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x_size = x.size()[2:]

        # MobileNetV3 백본을 사용하여 특징 추출
        x = self.backbone(x)

        # ASPP 모듈 적용
        x = self.aspp(x)

        # Squeeze-and-Excitation 모듈 적용
        x = self.se(x)

        # 클래스 점수 계산
        x = self.classifier(x)

        # 원본 이미지 크기로 업샘플링
        x = F.interpolate(x, size=x_size, mode='bilinear', align_corners=False)
        return x

# 모델 생성 및 장치 설정
num_classes = 8  # 배경을 포함한 총 8개 클래스
model = MobileNetV3DeepLabV3(num_classes=num_classes)

# 모델을 GPU 또는 CPU에 할당
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
