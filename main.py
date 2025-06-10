import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, recall_score, f1_score
from torchmetrics import Accuracy, Recall, F1Score
from collections import Counter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from utils import get_timm_model
from dl import get_dataloader, MammoNpyDataset, MammoDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# reproducibility
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    print(f"🔒 Seed 고정 완료: {seed}")


# ===== 학습 함수 =====
class MammoClassifier(pl.LightningModule):
    def __init__(self, model_name, num_classes, class_weights=None, lr=1e-4, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_timm_model(model_name, num_classes=num_classes, gray_scale=True)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

        # 🔸 Metrics 초기화
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("Lr", current_lr, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, y)
        self.val_recall.update(preds, y)
        self.val_f1.update(preds, y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
        # 🔸 자동 집계된 metric 출력
        self.log('val_acc', self.val_acc.compute(), prog_bar=True, sync_dist=True)
        self.log('val_recall', self.val_recall.compute(), prog_bar=True, sync_dist=True)
        self.log('val_f1', self.val_f1.compute(), prog_bar=True, sync_dist=True)

        # 🔸 reset 필수
        self.val_acc.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
        return [optimizer], [scheduler]

# ===== 메인 실행 =====
def main():
    torch.set_float32_matmul_precision('medium')  # 또는 'high'
    set_seed(26)

    model_name = "convnext_base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 사용자 정의 dataset 준비
    root_dir = "dataset/bi-rads/4channles_cls"
    data_module = MammoDataModule(root_dir=root_dir, batch_size=32)
    data_module.setup()

    num_classes = 4
    lightning_model = MammoClassifier(model_name=model_name, num_classes=num_classes, class_weights=data_module.class_weights.to(device))

    checkpoint_acc = ModelCheckpoint(dirpath='Project/Mammo/checkpoints/', filename=f"{model_name}-" + "{epoch}-{val_acc:.2f}",
                                    monitor='val_acc', mode='max', save_top_k=3, verbose=True)
    early_stop_cb = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=True)

    trainer = pl.Trainer(
        default_root_dir="Project/Mammo/logs",
        max_epochs=200, min_epochs=155,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto', strategy='ddp',
        callbacks=[checkpoint_acc, early_stop_cb],
        log_every_n_steps=10
    )

    trainer.fit(lightning_model, datamodule=data_module)

if __name__ == "__main__":
    main()

