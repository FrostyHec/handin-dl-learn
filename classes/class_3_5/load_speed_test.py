import math, torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from classes import DATA_DIR
from utils.time_utils import Timer

device    = 'cpu'                         # 仅测加载速度，通常在 CPU 上即可
REPEAT    = 3                             # 每个 batch_size 测多次求平均
BATCHES_PER_RUN = 100                     # 每次只迭代前 N 个 batch，加快实验

def get_dataloader(batch_size, num_workers=1, resize=None):
    """返回一个只做 *ToTensor* (+ 可选 Resize) 的 FashionMNIST DataLoader。"""
    tfms = [transforms.ToTensor()]
    if resize:
        tfms.insert(0, transforms.Resize(resize))
    tfms = transforms.Compose(tfms)

    ds = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=tfms)

    return DataLoader(ds, batch_size=batch_size,
                      shuffle=True, num_workers=num_workers,
                      pin_memory=(device=='cuda'))

def measure_throughput(batch_size, num_workers):
    """返回 samples/sec 的平均值 (REPEAT 次取平均)。"""
    loader = get_dataloader(batch_size, num_workers=num_workers)
    timer = Timer()
    speeds = []
    for _ in range(REPEAT):
        n_samples = 0
        for i, (X, y) in enumerate(loader):
            if i >= BATCHES_PER_RUN:
                break
            n_samples += X.size(0)
        secs = timer.stop()
        speeds.append(n_samples / secs)
    return sum(speeds) / len(speeds)

if __name__ == '__main__':
    batch_sizes = [2 ** k for k in range(0, 11 + 4)]   # 1 … 16384
    worker_list = [1, 2, 4, 8, 16, 32, 128]

    plt.figure(figsize=(8, 5))

    for nw in worker_list:
        results = []
        for bs in batch_sizes:
            spd = measure_throughput(bs, num_workers=nw)
            results.append(spd)
            print(f"[workers={nw:3d}] bs={bs:5d}  speed={spd:8.1f} samples/s")
        log_bs = [math.log2(b) for b in batch_sizes]
        plt.plot(log_bs, results, marker='o', label=f'workers={nw}')

    plt.xticks(log_bs, batch_sizes, rotation=30)   # ↖︎ 倾斜 30° 避免重叠
    plt.xlabel('batch_size')
    plt.ylabel('Samples / second')
    plt.title('DataLoader Speed vs. Batch Size (different num_workers)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig('loader_speed_workers.png', dpi=150)
    print("Figure saved to loader_speed_workers.png")
