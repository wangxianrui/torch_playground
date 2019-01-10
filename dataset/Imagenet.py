import torch
import torch.utils.data
import torchvision
import os
from .transform_classify import *


class Imagenet:
    def __init__(self, batch_size, num_workers=4):
        self.dataset_root = '/home/wxrui/DATA/cifar10/imagenet'
        self.NUM_CLASSES = 1000
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_loader(self):
        train_set = torchvision.datasets.ImageFolder(
            os.path.join(self.dataset_root, 'train'), transform=transform_train
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )
        return train_loader

    def get_eval_loaer(self):
        eval_set = torchvision.datasets.ImageFolder(
            os.path.join(self.dataset_root, 'val'), transform=transform_eval
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_set, num_workers=self.num_workers, pin_memory=True
        )
        return eval_loader
