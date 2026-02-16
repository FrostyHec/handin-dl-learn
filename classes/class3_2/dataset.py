import numpy as np
import torch

from utils.draw_utils import DrawUtils


def synthetic_data(w, b, num_examples):  # @save
    """
    生成y=Xw+b+噪声
    其中：w (d,); b 标量; num_examples = n
    """
    X = torch.normal(0, 1, size=(num_examples, len(w)))  # 随机一组(n,d)的数据，分布为N(0,1)
    epsilon = torch.normal(0, 0.01, size=(num_examples,))
    y = torch.matmul(X, w) + b + epsilon  # (n,)
    y = y.reshape((-1, 1))  # reshape成二维数组(n,1)

    # ============ 数据一致性检查 ============
    # (1) X 与 y 都应为二维数组
    assert len(X.shape) == 2 and len(y.shape) == 2, \
        f"X 和 y 必须都是二维数组，但当前形状分别为 {X.shape} 与 {y.shape}"

    # (2) 样本数检查
    assert num_examples == X.shape[0] == y.shape[0], \
        f"样本数不一致：num_examples={num_examples}, X 行={X.shape[0]}, y 行={y.shape[0]}"

    # (3) 特征维度数检查
    assert X.shape[1] == w.shape[0], \
        f"特征维度不匹配：X 列数={X.shape[1]}，而 w 长度={w.shape[0]}"
    return X, y





def load_example_ds() -> tuple[torch.Tensor, torch.Tensor]:
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    return features, labels


if __name__ == "__main__":
    features, labels = load_example_ds()
    print(f"Total dataset size:{len(features)}")
    print(f"features: {features[:5]},\n labels:{labels[:5]}")

    for dim in range(features[0].shape[0]):
        DrawUtils.draw_dataset(features, labels, dim, fig_name=f"dataset_dim={dim}")
