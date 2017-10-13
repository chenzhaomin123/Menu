from config import *
from utility import  Variable
import torch
import torch.optim as optim
import MyResNet as resent
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
from CenterLoss import CenterLoss
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

def main():
    net = resent.resnet50(False, num_classes=num_classes)
    center_loss = CenterLoss(num_classes, 2048, loss_weight=0.7)
    cross_loss = nn.CrossEntropyLoss()
    if USE_CUDA:
        net.cuda()
        center_loss.cuda()
        cross_loss.cuda()

    optimizer4nn = optim.SGD(net.parameters(), lr=cross_lr, momentum=momentum, weight_decay=weight_decay)
    sheduler = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.8)

    optimzer4center = optim.SGD(center_loss.parameters(), lr=center_lr)


    normalize = transforms.Normalize(mean=mean, std=std)

    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             transforms.Scale(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize
                                         ]))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle,pin_memory=True)

    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([
                                           transforms.Scale(224),
                                           transforms.ToTensor(),
                                           normalize
                                       ]))
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)


    for e in range(epoch):
        sheduler.step()
        train(train_loader, net, [optimizer4nn, optimzer4center], e, [cross_loss,center_loss])

        prec1 = validate(val_loader, net, [cross_loss,center_loss])

        if (e+1) % 5 == 0:
            torch.save(net.state_dict(),'save/weight_%d' % (e+1))



def train(train_loader,net,optimizer,epoch,criterion):
    # classes_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    net.train()

    for i, (data,target) in enumerate(train_loader):
        data = Variable(data)
        target = Variable(target)

        output,pred = net(data)

        loss = criterion[0](pred, target)

        # loss = classes_loss + criterion[1](target, output)

        prec1, prec3 = accuracy(pred.data, target.data, topk=(1, 3))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top3.update(prec3[0], data.size(0))
        # classes_losses.update(classes_loss.data[0], data.size(0))

        optimizer[0].zero_grad()
        # optimizer[1].zero_grad()
        # classes_loss.backward()
        loss.backward()
        optimizer[0].step()
        # optimizer[1].step()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(train_loader), loss=losses, top1=top1, top3=top3))


def validate(val_loader, net, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    net.eval()

    for i, (data, target) in enumerate(val_loader):
        data = Variable(data)
        target = Variable(target)

        output, pred = net(data)
        loss = criterion[0](pred, target)
        # loss = criterion[0](pred, target) + criterion[1](target, output)

        prec1, prec3 = accuracy(pred.data, target.data, topk=(1, 3))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top3.update(prec3[0], data.size(0))

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader), loss=losses, top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))
    return top1.avg



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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    import config
    lr = config.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()