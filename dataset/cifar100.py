import torch
import torch.utils.data
import torchvision
from .transform_classify import *


class Cifar100:
    def __init__(self, batch_size, num_workers=4):
        self.dataset_root = '/home/wxrui/DATA/cifar100'
        self.NUM_CLASSES = 100
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_loader(self):
        train_set = torchvision.datasets.cifar.CIFAR100(
            self.dataset_root, train=True, transform=transform_train, download=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )
        return train_loader

    def get_eval_loader(self):
        eval_set = torchvision.datasets.cifar.CIFAR100(
            self.dataset_root, train=False, transform=transform_eval, download=True
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_set, num_workers=self.num_workers, pin_memory=True
        )
        return eval_loader
