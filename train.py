#!/usr/bin/env python
import os
import torch
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import argparse
from data import detection_collate, VOCroot, VOC_CLASSES, VOC
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time
from subprocess import check_output
from datetime import datetime

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()

def save_checkpoint(net, args, iter, filename):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save({
        'iter': iter,
        'state_dict' : net.module.state_dict()
    }, filename)

def load_checkpoint(net, args):
    if args.resume:
        chkpnt = torch.load(args.resume)
        args.start_iter = chkpnt['iter']
        net.load_state_dict(chkpnt['state_dict'])
    else:
        if not os.path.exists(os.path.join(args.save_folder, args.basenet)):
            url = 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth'
            check_output(['wget', url, '-O', os.path.join(args.save_folder, args.basenet)])

        vgg_weights = torch.load(os.path.join(args.save_folder, args.basenet))
        net.vgg.load_state_dict(vgg_weights)
        net.extras.apply(init_weights)
        net.loc.apply(init_weights)
        net.conf.apply(init_weights)

def train(net, args):
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0

    dataset = VOC(args.voc_root, transform=SSDAugmentation(args.ssd_size, args.means))
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    N = len(data_loader)

    mk_var = lambda x: Variable(x.cuda() if args.cuda else x)

    for epoch in range(args.start_iter, args.epochs):
        for i, (images, targets) in enumerate(data_loader):
            if epoch in args.stepvalues:
                adjust_learning_rate(optimizer, args.gamma, epoch)

            images = mk_var(images)
            targets = [mk_var(anno) for anno in targets]

            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()


            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]

            print('%d: [%d/%d] || Loss: %.4f' % (epoch, i, N, loss.data[0]))

        save_checkpoint(net, args, epoch, os.path.join(args.checkpoint_dir, 'epoch_%d.pth' % epoch))

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
    parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
    parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--epochs', default=100, type=int, help='Maximum training epochs')
    parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
    parser.add_argument('--ssd_size', default=300, type=int, help='Input dimensions for SSD')
    args = parser.parse_args()

    args.checkpoint_dir = os.path.join(args.save_folder, 'ssd_%s' % datetime.now().isoformat())
    args.means = (104, 117, 123)  # only support voc now
    args.num_classes = len(VOC_CLASSES) + 1
    args.stepvalues = (20, 50, 70)
    args.start_iter = 0

    os.makedirs(args.save_folder, exist_ok = True)

    default_type = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor'
    torch.set_default_tensor_type(default_type)

    net = build_ssd('train', args.ssd_size, args.num_classes)

    if args.cuda:
        net = net.cuda()

    load_checkpoint(net, args)
    train(net, args)













