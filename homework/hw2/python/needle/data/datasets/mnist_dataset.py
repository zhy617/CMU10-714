from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
        flag = False,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as f:
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
        
        self.images:np.ndarray = images

        # TestOnly
        # if flag:
        #     print(self.images.shape)

        self.labels:np.ndarray = labels
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image = self.images[index]
        if self.transforms is not None:
            for fn in self.transforms:
                image = np.reshape(image, (28, 28, -1))
                image = fn(image)
                image = np.reshape(image, (-1, 28 * 28))
        return image, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION