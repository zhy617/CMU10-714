"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Project root:", project_root)
sys.path.append(project_root)

# notice that we can not import python.needle
# because it can cause fucking unknown error
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        # print(magic, num, rows, cols)
        images = np.frombuffer(f.read(), dtype=np.uint8)
        # print(images.shape)
        images = images.reshape(num, rows * cols).astype(np.float32)
        images = images / 255.0  # Normalize to [0, 1]

    with gzip.open(label_filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        labels = labels.reshape(num)
    
    return images, labels

    # return NotImplementedError("Please implement parse_mnist")
    ### END YOUR SOLUTION


def softmax_loss(Z:ndl.Tensor, y_one_hot:ndl.Tensor):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    loss = ndl.summation(ndl.log(ndl.summation(ndl.exp(Z), axes=1)) - ndl.summation(Z * y_one_hot, axes=1))
    # print("losssssssssssssssssssssssssss", loss.shape)
    loss /= Z.shape[0]
    return loss
    # return (ndl.log(ndl.exp(Z).sum((1,))).sum() - (y_one_hot * Z).sum()) / Z.shape[0]
    return NotImplementedError("Please implement softmax_loss")
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    # print(X.shape[0])
    for i in range(0, X.shape[0], batch):
        X_batch:ndl.Tensor = ndl.Tensor(X[i:min(i + batch, X.shape[0])])
        y_batch:ndl.NDArray = y[i:min(i + batch, X.shape[0])]
        y_one_hot = ndl.array_api.zeros((X_batch.shape[0], W2.shape[1]))
        y_one_hot[ndl.array_api.arange(y_batch.shape[0]), y_batch] = 1
        y_one_hot = ndl.Tensor(y_one_hot)
        # print(y)
        # print(y_one_hot)
        Z = ndl.relu(X_batch @ W1) @ W2
        loss:ndl.Tensor = softmax_loss(Z, y_one_hot)
        # print("loss shapeeeee", loss.shape)
        # print("W1 shape", W1.shape)
        # print("W2 shape", W2.shape)
        loss.backward()
        # print(W1.realize_cached_data())
        # print(W1.grad.realize_cached_data())
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
        # Z1 = np.maximum(X_batch @ W1, 0)
        # G2 = np.exp(Z1 @ W2) / np.sum(np.exp(Z1 @ W2), axis=1, keepdims=True) - np.eye(W2.shape[1])[y_batch]
        # G1 = (G2 @ W2.T) * (Z1 > 0)
        # W2 -= lr * (Z1.T @ G2) / X_batch.shape[0]
        # W1 -= lr * (X_batch.T @ G1) / X_batch.shape[0]
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
