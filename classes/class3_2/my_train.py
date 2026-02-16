# 从数据集中遍历k个epoch，每个epoch选择b个样本（打乱后逐步b个b个选择），然后求梯度，并优化模型参数
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from classes.class3_2.dataset import load_example_ds


class AbstractDataset(ABC):
    @abstractmethod
    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回 X,y
        pass


class DatasetLoader:
    def __init__(self, batch_size: int, dataset: AbstractDataset):
        self.batch_size = batch_size
        self.X, self.y = dataset.load()
        self.__dataset_check(self.X, self.y)

    def __dataset_check(self, X: torch.Tensor, y: torch.Tensor):
        # (1) X 与 y 都应为二维数组
        assert len(X.shape) == 2 and len(y.shape) == 2, \
            f"X 和 y 必须都是二维数组，但当前形状分别为 {X.shape} 与 {y.shape}"

        # (2) 样本数检查
        assert X.shape[0] == y.shape[0], \
            f"样本数不一致：X 行={X.shape[0]}, y 行={y.shape[0]}"

    def sample(self):
        num_examples = len(self.X)
        indices = list(range(num_examples))
        # 这些样本是随机读取的，没有特定的顺序
        random.shuffle(indices)
        for i in range(0, num_examples, self.batch_size):
            batch_indices = torch.tensor(
                indices[i: min(i + self.batch_size, num_examples)])
            yield self.X[batch_indices], self.y[batch_indices]


class AbstractModel(ABC):
    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def optimize(self, x: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor, learning_rate: float):
        pass

    @abstractmethod
    def print_params(self):
        pass


class LinearModel(AbstractModel):
    def __init__(self, in_features: int):
        self.w = torch.normal(0, 0.01, size=(in_features, 1), requires_grad=False)
        self.b = torch.zeros(1, requires_grad=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def optimize(self, X: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor, learning_rate: float):
        e = y_pred - y_true  # (n,1)
        n = y_true.shape[0]
        grad_w = X.T @ e
        grad_b = e.sum()
        self.w -= learning_rate / n * grad_w
        self.b -= learning_rate / n * grad_b

    def print_params(self):
        print(f"Learned w: {self.w.squeeze().tolist()}")
        print(f"Learned b: {self.b.item()}")


@dataclass
class TrainArgs:
    epoch: int
    batch_size: int
    learning_rate: float
    dataset: AbstractDataset
    model: AbstractModel


class MyDataset(AbstractDataset):
    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        return load_example_ds()


def train_pipeline(train_args: TrainArgs):
    dataset = train_args.dataset
    dataset_loader = DatasetLoader(train_args.batch_size, dataset)
    model = train_args.model
    for epoch in range(train_args.epoch):
        print(f"\n===== Epoch {epoch + 1}/{train_args.epoch} =====")
        # --------- 训练 ---------
        for batch_x, batch_y_true in dataset_loader.sample():
            batch_y_pred = model.forward(batch_x)
            model.optimize(batch_x, batch_y_pred, batch_y_true, learning_rate=train_args.learning_rate)
        # --------- 评估 ---------
        print(f"Epoch {epoch} Train Completed! Evaluating...")
        total_se, total_n = 0.0, 0
        with torch.no_grad():
            for bx, by in dataset_loader.sample():
                y_pred = model.forward(bx)
                total_se += ((y_pred - by) ** 2).sum().item()
                total_n += by.shape[0]
        avg_l2 = total_se / total_n
        print(f"Avg L2 loss: {avg_l2:.6f}")

    print("Train completed! Printing Model params:")
    model.print_params()


if __name__ == "__main__":
    train_args = TrainArgs(
        epoch=10,
        batch_size=32,
        learning_rate=0.03,
        dataset=MyDataset(),
        model=LinearModel(in_features=2)
    )
    train_pipeline(train_args)
