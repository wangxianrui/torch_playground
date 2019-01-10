import os
import time
import torch
import torch.utils.data
import torchvision
from tensorboardX import SummaryWriter
from dataset.get_dataset import get_dataset
from model.get_model import get_model

writer = SummaryWriter('log')


class CONFIG:
    # dataset:  cifar10, cifar100, imagenet
    dataset = 'imagenet'
    dataset_root = os.path.expanduser('~/DATA/' + dataset)
    batch_size = 64
    num_workers = 8

    # model
    model_name = 'resnet50'
    pretrained = True
    resume_path = ''

    # optimizer
    lr = 1e-2
    momentum = 0.9
    weight_decay = 1e-4
    lr_step = [50, 90, 120, 140, 150]

    # train
    epochs = lr_step[-1]
    print_freq = 10
    save_dir = 'checkpoint'


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


def adjust_learning_rate(optimizer, epoch):
    if epoch in CONFIG.lr_step:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def accuracy(output, target, topk=(1,)):
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


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % CONFIG.print_freq == 0:
            index = epoch * len(train_loader) + i
            writer.add_scalar('train/loss', loss.item(), index)
            writer.add_scalar('train/prec1', prec1.item(), index)
            writer.add_scalar('train/prec5', prec5.item(), index)
            print('Epoch: [{}][{}/{}]\t'
                  'Lr: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def eval(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % CONFIG.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

    return top1.avg


def main():
    # dataset
    train_set, val_set = get_dataset(CONFIG.dataset, CONFIG.dataset_root)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(val_set)

    # model
    num_classes_map = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000
    }
    model = get_model(CONFIG.model_name, num_classes_map[CONFIG.dataset], CONFIG.resume_path)
    print(model)

    # ceriterion, optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        CONFIG.lr,
        momentum=CONFIG.momentum,
        weight_decay=CONFIG.weight_decay
    )

    for epoch in range(CONFIG.epochs):
        # adjust learning_rate
        adjust_learning_rate(optimizer, epoch)

        # train
        train(train_loader, model, criterion, optimizer, epoch)

        # eval
        eval_prec = eval(val_loader, model, criterion)
        writer.add_scalar('eval/prec1', eval_prec, epoch)

        # save
        torch.save(model.state_dict(), '%s/%s_%d.pth' % (CONFIG.save_dir, CONFIG.model_name, epoch))
    writer.close()


if __name__ == '__main__':
    main()
