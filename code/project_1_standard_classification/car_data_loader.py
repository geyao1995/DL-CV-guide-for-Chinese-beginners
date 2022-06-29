# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#imagefolder
from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import CarBrandClsConfig


class CarDataset(ImageFolder):
    def __init__(self, root: Path, train: bool, transform=None):
        if train:
            dir_images = root.joinpath('train')
        else:
            dir_images = root.joinpath('test')

        super(CarDataset, self).__init__(root=dir_images, transform=transform)


def get_loader():
    root_images = CarBrandClsConfig.root_images
    size_input = CarBrandClsConfig.size_input
    size_train_batch = CarBrandClsConfig.size_train_batch
    size_test_batch = CarBrandClsConfig.size_test_batch
    size_padding = [i // 8 for i in size_input]

    transform_train = transforms.Compose([
        transforms.Resize(size_input),
        transforms.RandomCrop(size_input, padding=size_padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size_input),
        transforms.ToTensor(),
    ])

    train_dataset = CarDataset(root=root_images, train=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=size_train_batch, shuffle=True,
                              num_workers=2, pin_memory=True)

    test_dataset = CarDataset(root=root_images, train=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=size_test_batch, shuffle=False,
                             num_workers=2, pin_memory=True)

    return train_loader, test_loader


if __name__ == '__main__':
    # train_loader, test_loader = get_loader(is_mix_train=True)
    # imgs, labels = next(iter(train_loader))
    # print(imgs.size(), imgs.mean(), labels.size())

    root_images = CarBrandClsConfig.root_images
    size_input = CarBrandClsConfig.size_input
    size_padding = [i // 8 for i in size_input]

    transform_train = transforms.Compose([
        transforms.Resize(size_input),
        transforms.RandomCrop(size_input, padding=size_padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    car_mix_dataset = CarDataset(root=root_images, train=False, transform=transform_train)

    for i in range(10):
        image, label = car_mix_dataset[i]
        plt.imshow(image.permute(1, 2, 0))
        plt.title(car_mix_dataset.classes[label])
        plt.show()
