import torch
from dataset.cifar10 import Cifar10

RESUME = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MobileNetV2Cifar10:
    def __init__(self):
        self.image_shape = 224
        self.batch_size = 64
        self.num_workers = 8
        self.lr_init = 0.1
        self.lr_decay = 0.1
        self.lr_epoch = [50, 90, 120, 140, 150]
        self.max_epoch = 150
        self.weight_decay = 1e-4
        self.log_step = 20
        self.save_step = 500
        self.save_path = 'checkpoint'

    def train(self):
        dataset = Cifar10(self.batch_size, self.num_workers)
        

    def eval(self):
        pass

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.lr_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
