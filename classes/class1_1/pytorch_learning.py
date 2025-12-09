import torch

if __name__ == '__main__':
    x = torch.arange(12)
    print(x)
    X = x.reshape(3, 4)
    print(X.size())
    z = torch.empty(3)
    print(z)