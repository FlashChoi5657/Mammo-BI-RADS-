import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score
import timm

# ===== 모델 불러오기 함수 =====
def convert_to_grayscale(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(4, module.out_channels, module.kernel_size,
                                  module.stride, module.padding, bias=module.bias is not None)
            with torch.no_grad():
                avg_weight = module.weight.data.mean(dim=1, keepdim=True)
                new_conv.weight.data = avg_weight.repeat(1, 4, 1, 1)
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data
            setattr(model, name, new_conv)
            print(f"✅ Replaced Conv2d in {name}")
            break
        else:
            convert_to_grayscale(module)

def get_timm_model(name, num_classes=4, gray_scale=False):
    try:
        model = timm.create_model(name, pretrained=True)
    except:
        model = timm.create_model(name, pretrained=False)
        print(f"❌ Failed to load {name}")
    model.reset_classifier(num_classes)

    if gray_scale:
        convert_to_grayscale(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 총 파라미터 수: {total:,}")
    print(f"🎯 학습 가능한 파라미터 수: {trainable:,}")
    return model

class EarlyStoppingTopK:
    def __init__(self, patience=5, save_dir='checkpoints', top_k=3):
        """
        상위 K개 모델을 저장하고, 성능이 개선되지 않으면 조기 종료하는 유틸리티

        Parameters:
        - patience (int): 개선되지 않아도 기다릴 최대 epoch 수
        - save_dir (str): 모델 파일 저장 폴더
        - top_k (int): 저장할 최고 성능 모델 수
        """
        self.patience = patience                # 최대 허용 정체 epoch 수
        self.save_dir = save_dir                # 모델 저장 폴더
        self.top_k = top_k                      # 저장할 top-k 모델 개수
        self.counter = 0                        # 연속으로 개선되지 않은 epoch 수
        self.early_stop = False                 # 중단 여부
        self.best = []                          # [(val_loss, index)] 형태로 저장
        os.makedirs(save_dir, exist_ok=True)    # 저장 경로 생성

    def __call__(self, val_loss, model):
        """
        매 epoch마다 호출하여 val_loss를 평가하고 모델을 저장 또는 중단 판단

        Parameters:
        - val_loss (float): 현재 validation loss
        - model (nn.Module): 현재 모델 객체

        Returns:
        - early_stop (bool): 학습을 멈춰야 하는지 여부
        """
        # 조건: 아직 top_k 미만이거나, 기존 top_k 중 가장 안 좋은 loss보다 더 좋을 때
        if len(self.best) < self.top_k or val_loss < self.best[-1][0]:
            # 인덱스 결정 (새로 추가 or 가장 나쁜 것 덮어쓰기)
            idx = len(self.best) if len(self.best) < self.top_k else self.top_k - 1
            # 파일 저장
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"checkpoint_top{idx+1}.pt"))
            # 리스트에 추가하고 정렬
            self.best.append((val_loss, idx))
            self.best.sort()  # val_loss 오름차순
            self.best = self.best[:self.top_k]  # top_k만 유지
            # 이름 재정렬 (top1 ~ topK)
            for i, (loss, _) in enumerate(self.best):
                path = os.path.join(self.save_dir, f"checkpoint_top{i+1}.pt")
                torch.save(model.state_dict(), path)
            self.counter = 0  # 개선되었으므로 counter 초기화
        else:
            self.counter += 1  # 개선되지 않음 → 카운터 증가
            if self.counter >= self.patience:
                self.early_stop = True  # patience 초과 시 조기 종료

        return self.early_stop

def ensemble_predict(models, dataloader, device, weights=None, voting='soft'):
    """
    앙상블 예측 함수 (DataLoader 지원, soft/hard voting + 성능 평가 포함)

    Parameters:
        models (list): PyTorch 분류 모델 리스트
        dataloader (DataLoader): (images, labels) 튜플 제공
        device (str): 실행 디바이스 (예: 'cuda' 또는 'cpu')
        weights (list or None): soft voting에서 모델별 가중치 (None이면 평균)
        voting (str): 'soft' 또는 'hard'

    Returns:
        all_preds (Tensor): 예측된 클래스 인덱스 (N,)
        all_probs (Tensor or None): soft voting일 경우 평균 확률 분포 (N, num_classes)
        metrics (dict): 정확도, F1, Recall
    """
    for model in models:
        model.eval()
        model.to(device)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            else:
                raise ValueError("Dataloader must return (images, labels) tuple.")
            images, labels = images.to(device), labels.to(device)

            batch_probs = []
            batch_preds = []

            for model in models:
                output = model(images)              # (B, num_classes)
                prob = F.softmax(output, dim=1)     # (B, num_classes)
                batch_probs.append(prob)
                batch_preds.append(prob.argmax(dim=1))  # (B,)

            if voting == 'hard':
                votes = torch.stack(batch_preds, dim=0)  # (num_models, B)
                pred_class = torch.mode(votes, dim=0).values  # (B,)
                all_preds.append(pred_class)

            else:  # soft voting
                probs = torch.stack(batch_probs, dim=0)  # (num_models, B, num_classes)
                if weights:
                    weights_tensor = torch.tensor(weights, device=device).view(-1, 1, 1)
                    weighted_probs = probs * weights_tensor
                    avg_prob = weighted_probs.sum(dim=0) / weights_tensor.sum()
                else:
                    avg_prob = probs.mean(dim=0)  # (B, num_classes)

                pred_class = avg_prob.argmax(dim=1)  # (B,)
                all_preds.append(pred_class)
                all_probs.append(avg_prob)

            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0).cpu()

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_score': f1_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
    }
    print(metrics)

    if voting == 'soft':
        all_probs = torch.cat(all_probs, dim=0).cpu()
        return all_preds, all_probs, metrics
    else:
        return all_preds, None, metrics
