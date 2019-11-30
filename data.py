import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset


def load_mnist(n_ex):
    from tensorflow.keras.datasets import mnist as mnist_keras

    x_test, y_test = mnist_keras.load_data()[1]
    x_test = x_test.astype(np.float64) / 255.0
    x_test = x_test[:, None, :, :]

    return x_test[:n_ex], y_test[:n_ex]


def load_cifar10(n_ex):
    from madry_cifar10.cifar10_input import CIFAR10Data
    cifar = CIFAR10Data('madry_cifar10/cifar10_data')
    x_test, y_test = cifar.eval_data.xs.astype(np.float32), cifar.eval_data.ys
    x_test = np.transpose(x_test, axes=[0, 3, 1, 2])
    return x_test[:n_ex], y_test[:n_ex]


def load_imagenet(n_ex, size=224):
    IMAGENET_SL = size
    IMAGENET_PATH = "/scratch/maksym/imagenet/val_orig"
    imagenet = ImageFolder(IMAGENET_PATH,
                           transforms.Compose([
                               transforms.Resize(IMAGENET_SL),
                               transforms.CenterCrop(IMAGENET_SL),
                               transforms.ToTensor()
                           ]))
    torch.manual_seed(0)

    imagenet_loader = DataLoader(imagenet, batch_size=n_ex, shuffle=True, num_workers=1)
    x_test, y_test = next(iter(imagenet_loader))

    return np.array(x_test, dtype=np.float32), np.array(y_test)


datasets_dict = {'mnist': load_mnist,
                 'cifar10': load_cifar10,
                 'imagenet': load_imagenet,
}
bs_dict = {'mnist': 10000,
           'cifar10': 4096,  # 4096 is the maximum that fits
           'imagenet': 100,
}
