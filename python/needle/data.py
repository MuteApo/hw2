import struct
import gzip
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """

        flip_img = np.random.rand() < self.p
        return img[:, ::-1, :] if flip_img else img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """

        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding + 1, size=2)
        h, w, c = img.shape
        output = np.zeros((h + self.padding * 2, w + self.padding * 2, c))
        output[self.padding:h + self.padding, self.padding:w + self.padding, :] = img[:, :, :]
        h0, w0 = self.padding + shift_x, self.padding + shift_y
        return output[h0:h0 + h, w0:w0 + w, :]


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """

    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        self.batch_id = 0
        if self.shuffle:
            indices = list(range(len(self.dataset)))
            np.random.shuffle(indices)
            batches = range(self.batch_size, len(self.dataset), self.batch_size)
            self.ordering = np.array_split(indices, batches)
        return self

    def __next__(self):
        if self.batch_id >= len(self.ordering):
            raise StopIteration
        mini_batch = []
        for i in range(len(self.dataset[0])):
            item = [self.dataset[_][i] for _ in self.ordering[self.batch_id]]
            mini_batch.append(Tensor(item))
        self.batch_id += 1
        return mini_batch


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.images, image_num = self.load_image(image_filename)
        self.labels, label_num = self.load_label(label_filename)
        assert image_num == label_num, 'image number mismatches label number'
        self.len = image_num
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        if isinstance(self.labels[index], Iterable):
            img = self.images[index].reshape(-1, 28, 28, 1)
            img = np.array([self.apply_transforms(_).reshape(-1,) for _ in img])
        else:
            img = self.apply_transforms(self.images[index].reshape(28, 28, 1)).reshape(-1)
        return img, self.labels[index]

    def __len__(self) -> int:
        return self.len

    def load_image(self, image_filename):
        with gzip.open(image_filename, 'rb') as image_file:
            _, image_num, _, _ = struct.unpack('>IIII', image_file.read(16))
            images = np.frombuffer(image_file.read(), dtype=np.uint8)
            images = images.reshape(image_num, 784).astype('float32')
            image_max, image_min = np.max(images), np.min(images)
            images = (images - image_min) / (image_max - image_min)
        return images, image_num

    def load_label(self, label_filename):
        with gzip.open(label_filename, 'rb') as label_file:
            _, label_num = struct.unpack('>II', label_file.read(8))
            labels = np.frombuffer(label_file.read(), dtype=np.uint8)
        return labels, label_num


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
