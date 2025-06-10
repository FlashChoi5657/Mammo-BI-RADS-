import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class MammoNpyDataset(Dataset):
    def __init__(self, csv_file, root_dir, split="trian", transform=None):
        """
        Parameters:
            csv_file (str): label_map.csv 경로
            root_dir (str): .npy 파일들이 있는 폴더 경로
            transform (callable, optional): numpy 이미지에 적용할 transform 함수
        """
        df = pd.read_csv(csv_file)
        self.labels_df = df[df["split"] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        file_path = os.path.join(self.root_dir, row['filename'])
        image = np.load(file_path)  # shape: (H, W, 4)
        label = row['class'] -1

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)  

        # if isinstance(image, np.ndarray):
        #     image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return image, label

from torchvision import transforms

# ✅ 훈련 데이터용 변환기 (Data Augmentation 포함)
transform_train = transforms.Compose([
                                transforms.Resize((512, 512)),          # 모든 이미지를 512x512 크기로 맞춤 (입력 통일 목적)
                                transforms.ColorJitter(brightness=0.4), # [-40%, +40%] 범위 내 에서 무작위로 밝기 배수 선택
                                transforms.RandomApply([               # 대괄호 안의 transform 함수는 확률적으로 적용
                                transforms.RandomRotation(10),          # 이미지 ±10도 범위 내에서 회전
                                ], p=0.5),
                                transforms.ToTensor(),                  # PIL 이미지 → PyTorch 텐서 (0~255 → 0~1 정규화 포함)
                            ])

# ✅ 검증/테스트용 변환기 (데이터 증강 ❌, 입력만 정규화)
transform_test = transforms.Compose([
                                transforms.Resize((512, 512)),          # 테스트 이미지 크기 통일 (학습보다 작게 사용 가능)
                                transforms.ToTensor(),                  # 텐서로 변환
                            ])

csv_path = "../dataset/BI-Rads_cls/01.label_map.csv"
root_dir = "../dataset/BI-Rads_cls"

train_dataset = MammoNpyDataset(csv_file=csv_path, root_dir=root_dir, split="train", transform=transform_train)
val_dataset = MammoNpyDataset(csv_file=csv_path, root_dir=root_dir, split="val", transform=transform_test)


from torch.utils.data import DataLoader

batch_size = 8

loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
loader_valid = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


import torch
import random
import numpy as np
import os

seed = 26
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 멀티 GPU용

torch.backends.cudnn.deterministic = True  # 연산 순서를 고정해서 동일한 결과 유도
torch.backends.cudnn.benchmark = False     # 성능 대신 재현성을 선택
torch.backends.cudnn.enabled = False       # 완전히 끌 수도 있음 (필요 시)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# timm 모델 리스트 (전이학습용)
model_names = [
    "resnet50",
    "efficientnet_b0",
    "mobilenetv3_small_100",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "swinv2_base_window8_256",
    "swinv2_cr_base_224", # transfer X
    "convnext_base",
    "regnety_016",
    "densenet121",
    "dm_nfnet_f0", # transfer X
    "coat_lite_small"
]

def convert_to_grayscale(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None
            )
            
            with torch.no_grad():
            # 평균 값을 구해 3채널 → 1채널로 가중치 초기화
                avg_weight =  module.weight.data.mean(dim=1, keepdim=True)
                new_conv.weight.data = avg_weight.repeat(1, 4, 1, 1)
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data
            setattr(model, name, new_conv)
            print(f"✅ Replaced Conv2d in {name}")
            break
        else:
            convert_to_grayscale(module)  # 재귀적으로 탐색

# 모델 생성 및 파라미터 확인
def get_timm_model(name, num_classes=10, gray_scale=False):
    try:
        model = timm.create_model(name, pretrained=True)
    except: # transfer learning이 불가한 모델이 존재!
        model = timm.create_model(name, pretrained=False)
        print(f"❌ Failed to load {name}")
    # reset_classifier 지원 여부 체크 필요
    model.reset_classifier(num_classes)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 총 파라미터 수: {total:,}")
    print(f"🎯 학습 가능한 파라미터 수: {trainable:,}")

    if gray_scale:
        convert_to_grayscale(model)

    return model


model_name = "resnet152"
model = get_timm_model(model_name, num_classes=4, gray_scale=True)
model = nn.DataParallel(model)
model.to(device)