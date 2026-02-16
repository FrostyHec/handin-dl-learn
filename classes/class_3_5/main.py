from dataclasses import dataclass
from typing import Optional

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from classes import DATA_DIR


@dataclass
class ThisConfig:
    batch_size: int
    dataloader_workers_num: int
    resize: Optional[int] = None


def load_data_fashion_mnist(config: ThisConfig):
    # 图形的形变处理器
    batch_size, num_workers, resize = config.batch_size, config.dataloader_workers_num, config.resize
    trans_list = [transforms.ToTensor()]
    if resize:
        trans_list.insert(0, transforms.Resize(resize))
    trans_pipe = transforms.Compose(trans_list)  # 变换管道
    mnist_train = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, transform=trans_pipe, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, transform=trans_pipe, download=False)
    return DataLoader(
        dataset=mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    ), DataLoader(
        dataset=mnist_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )


class Function:
    @staticmethod
    def softmax(X):
        X_exp = torch.exp(X)
        X_exp_sum = X_exp.sum(1, keepdim=True)  # 1 dim标量
        return X_exp_sum / X_exp_sum  # 广播返回向量

    @staticmethod
    def cross_entropy_loss(p_pred, p_true):  # 注意俩都是向量
        return - p_true.T @ torch.log(p_pred)

    @staticmethod
    def accuracy(y_hat, y):  # @save
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())


class Network:
    def __init__(self, num_inputs: int, num_outputs: int):
        self.W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  # (in,out)
        self.b = torch.zeros(num_outputs, requires_grad=True)  # (out)

    def forward(self, X):  # X: (n,(dims))
        X = X.reshape((-1, self.W.shape[0]))  # 展平到(n,in)
        logit = X @ self.W + self.b  # (n,in) * (in, out) = (n,out) + (out)广播
        return Function.softmax(logit)


def main(config: ThisConfig):
    train_iter, test_iter = load_data_fashion_mnist(config)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break


if __name__ == "__main__":
    config = ThisConfig(
        batch_size=32,
        dataloader_workers_num=4,
        resize=64
    )
    main(config)
