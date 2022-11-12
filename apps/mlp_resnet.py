import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim))
    model = nn.Sequential(nn.Residual(fn), nn.ReLU())
    return model


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    model = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)]),
        nn.Linear(hidden_dim, num_classes))
    return model


def epoch(dataloader, model, opt=None):
    np.random.seed(4)

    model.train()
    if opt is None:
        model.eval()

    total_acc = total_loss = 0
    for _, (X, y) in enumerate(dataloader):
        pred = model(X)
        total_acc += np.sum(pred.numpy().argmax(1) == y.numpy())

        loss = nn.SoftmaxLoss()(pred, y)
        total_loss += loss.numpy() * y.shape[0]

        if opt is not None:
            loss.backward()
            opt.step()
            opt.reset_grad()

    avg_acc = total_acc / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader.dataset)
    return 1 - avg_acc, avg_loss


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)

    train_dataset = ndl.data.MNISTDataset(
        "%s/train-images-idx3-ubyte.gz" % (data_dir),
        "%s/train-labels-idx1-ubyte.gz" % (data_dir))
    test_dataset = ndl.data.MNISTDataset(
        "%s/t10k-images-idx3-ubyte.gz" % (data_dir),
        "%s/t10k-labels-idx1-ubyte.gz" % (data_dir))
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
        test_err, test_loss = epoch(test_dataloader, model)
        print(
            "train_err=%f\ttrain_loss=%f\ttest_err=%f\ttest_loss=%f" %
            (train_err, train_loss, test_err, test_loss))

    return train_err, train_loss, test_err, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
