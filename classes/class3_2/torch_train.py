import random
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ===== 1. 数据集（保持与原接口兼容） =====
# 这里假设 load_example_ds 返回 (X, y)，且 X,y 均为二维 Tensor
from classes.class3_2.dataset import load_example_ds


class ExampleDataset(Dataset):
    """将现有 load_example_ds() 封装为标准 Dataset"""

    def __init__(self) -> None:
        X, y = load_example_ds()  # X:(N, in_features), y:(N, 1)
        assert X.ndim == 2 and y.ndim == 2, "Expect 2-D tensors"
        assert X.shape[0] == y.shape[0], "Mismatched sample size"
        self.X, self.y = X.float(), y.float()  # 确保 dtype = float32

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ===== 2. 线性模型 + 损失函数 + 优化器 =====
def build_model(in_features: int, lr: float = 0.03) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    model = nn.Linear(in_features, 1)  # bias=True 默认即含 b
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    return model, optimizer, criterion


# ===== 3. 训练与评估 =====
def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    epochs: int = 10,
    device: torch.device | str = "cpu",
) -> None:
    device = torch.device(device)
    model.to(device)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # ---- Evaluate ----
        model.eval()
        with torch.no_grad():
            se_sum, n_total = 0.0, 0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                se_sum += ((y_pred - y_batch) ** 2).sum().item()
                n_total += y_batch.shape[0]
            avg_mse = se_sum / n_total

        print(f"Epoch {epoch:02d}/{epochs} | Avg MSE: {avg_mse:.6f}")

    # ---- Done ----
    w, b = model.weight.data.squeeze(), model.bias.data.item()
    print("\nTraining finished. Learned parameters:")
    print(f"w: {w.tolist()}")
    print(f"b: {b:.6f}")


# ===== 4. main 入口 =====
if __name__ == "__main__":
    torch.manual_seed(42)        # 为可重复性设随机种子
    random.seed(42)

    batch_size = 32
    epochs = 10
    lr = 0.03

    dataset = ExampleDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    in_features = dataset.X.shape[1]
    model, optimizer, criterion = build_model(in_features, lr=lr)

    train(model, optimizer, criterion, dataloader, epochs=epochs)
