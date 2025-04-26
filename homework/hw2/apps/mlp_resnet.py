import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    m = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(
        nn.Residual(m),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    m = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob)
            for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes),
    )
    return m
    ### END YOUR SOLUTION


def epoch(dataloader, model: nn.Module, opt: ndl.optim.Optimizer =None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    loss = nn.SoftmaxLoss()
    error_rate = 0.
    losses = 0.
    for i, batch in enumerate(dataloader):
        X, y = batch
        if opt is not None:
            opt.reset_grad()
        y_hat: ndl.Tensor = model(X)
        predictions = np.argmax(y_hat.realize_cached_data(), axis=1)
        error_rate += np.sum(predictions != y.numpy())
        loss_value: ndl.Tensor = loss(y_hat, y)
        losses += loss_value.numpy() * X.shape[0]
        if opt is not None:
            loss_value.backward()
            opt.step()
    
    return error_rate / len(dataloader.dataset), losses / len(dataloader.dataset)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
    )
    train_dataloader = ndl.data.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True
    )
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    )
    test_dataloader = ndl.data.DataLoader(
        test_dataset,
        batch_size,
        shuffle=False
    )
    model = MLPResNet(
        dim=28*28,
        hidden_dim=hidden_dim,
        num_blocks=3,
        num_classes=10,
        norm=nn.BatchNorm1d,
        drop_prob=0.1,
    )
    opt = optimizer(
        model.parameters(),
        lr = lr,
        weight_decay=weight_decay,
    ) if optimizer is not None else None

    train_error, train_loss = 0., 0.
    test_error, test_loss = 0., 0.
    for _ in range(epochs):
        train_error, train_loss = epoch(
            train_dataloader,
            model,
            opt,
        )
        test_error, test_loss = epoch(
            test_dataloader,
            model,
            None,
        )
    return train_error, train_loss, test_error, test_loss
    ## END YOUR SOLUTION

if __name__ == "__main__":
    train_mnist(data_dir="../data")
