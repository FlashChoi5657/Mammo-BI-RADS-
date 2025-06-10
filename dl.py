import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.utils.class_weight import compute_class_weight


class MammoNpyDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, image_size=512):
        df = pd.read_csv(os.path.join(root_dir, "01.label_map.csv"))
        self.labels_df = df[df["split"] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.mode = split
        self.size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        file_path = os.path.join(self.root_dir, row['filename'])
        image = np.load(file_path)  # shape: (H, W, 4)
        label = row['class'] - 1

        if self.transform():
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label

class MammoDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, image_size=512):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.size = image_size

    def setup(self, stage=None):    
        transform_train = transforms.Compose([
                                transforms.Resize((self.size, self.size)),          # 모든 이미지를 512x512 크기로 맞춤 (입력 통일 목적)
                                transforms.ColorJitter(brightness=0.4), # [-40%, +40%] 범위 내 에서 무작위로 밝기 배수 선택
                                transforms.RandomApply([               # 대괄호 안의 transform 함수는 확률적으로 적용
                                transforms.RandomRotation(10),          # 이미지 ±10도 범위 내에서 회전
                                ], p=0.5),
                                transforms.ToTensor(),                  # PIL 이미지 → PyTorch 텐서 (0~255 → 0~1 정규화 포함)
                            ])
        
        transform_val = transforms.Compose([
                                transforms.Resize((self.size, self.size)),          # 테스트 이미지 크기 통일 (학습보다 작게 사용 가능)
                                transforms.ToTensor(),                  # 텐서로 변환
                            ])
        
        if stage == 'fit' or stage is None:
            self.train_dataset = MammoNpyDataset(self.root_dir, split='train', transform=transform_train, image_size=self.image_size)
            self.val_dataset = MammoNpyDataset(self.root_dir, split='val', transform=transform_val, image_size=self.image_size)
        if stage == 'test' or stage is None:
            self.test_dataset = MammoNpyDataset(self.root_dir, split='test', transform=transform_val, image_size=self.image_size)
        
        labels = self.train_dataset.labels_df['class'].values - 1  # 클래스가 1부터 시작한다면 조정
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

def get_dataloader(root_dir, split, batch_size, image_size=512):
    dataset = MammoNpyDataset(root_dir, split, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=4)