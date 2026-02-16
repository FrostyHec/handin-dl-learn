import torch
from matplotlib import pyplot as plt


class DrawUtils:
    @staticmethod
    def draw_dataset(features:torch.Tensor, labels:torch.Tensor, x_dim:int, fig_name:str="dataset"):
        # 2. 准备绘图：先将张量搬到 CPU，再转 NumPy
        x = features[:, x_dim].detach().cpu().numpy()  # 选择第 2 列特征作为横轴
        y = labels.squeeze(1).detach().cpu().numpy()  # (n, 1) -> (n,)

        # 3. 创建画布（设置 figure size）
        plt.figure(figsize=(6, 4))  # 宽 6 英寸，高 4 英寸

        # 4. 画散点
        plt.scatter(x, y,
                    s=10,  # s 为每个点的大小
                    c="royalblue",  # 点的颜色
                    alpha=0.6,  # 透明度
                    edgecolors="none")  # 去掉点边框

        # 5. 添加注释
        plt.xlabel(f"Feature dim = {x_dim} (x)")
        plt.ylabel("Label (y)")
        plt.title("Synthetic data scatter plot")
        plt.grid(True, linestyle="--", alpha=0.3)  # 可选：加网格

        # 6. 显示
        plt.tight_layout()  # 防止文字被裁剪
        plt.savefig(f"{fig_name}.png")