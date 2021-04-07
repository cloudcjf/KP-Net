import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# We organize the parameters of this project using wandb
# If do not want to use this, please delete the code about wandb
import wandb

import argparse
from tqdm import tqdm
import os

from data.kittiloader import KittiDataset
from models.models import KPNet
from models.losses import ChamferLoss, Point2PointLoss, XXLoss

def parse_args():
    parser = argparse.ArgumentParser('RSKDD')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--data_dir', type=str, default='',help='dir of dataset')
    parser.add_argument('--seq', type=str, default='00', help='training sequence of Kitti dataset')
    parser.add_argument('--npoints', type=int, default=16384, help='number of input points')
    parser.add_argument('--k',type=int, default=128)
    parser.add_argument('--nsample', type=int, default=512)
    parser.add_argument('--desc_dim', type=int, default=32)
    parser.add_argument('--dilation_ratio', type=float, default=2.0)
    parser.add_argument('--pretrain_detector', type=str, default='./ckpt/best_detector.pth', \
        help='path to pretrain model of detector')
    parser.add_argument('--alpha', type=float, default=1.0, \
        help='ratio between chamfer loss and point to point loss')
    parser.add_argument('--beta', type=float, default=1.0, \
        help='ratio between chamfer loss and point to matching loss')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--sigma_max', type=float, default=3.0, \
        help='predefined sigma upper bound')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt', help='path to save model')
    parser.add_argument('--train_type', type=str, default='det', help='det/desc')
    parser.add_argument('--use_wandb', action='store_true')
    return parser.parse_args()

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    trainset = Dataset(args.data_dir, srgs.seq_name, args.npoints)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, \
        shuffle=True, num_workers=args.num_workers, drop_last=True) # 此时点云已经加载进来
    model = KPNet(args)
    model = nn.DataParallel(model)  # 支持多gpu训练
    model = model.cuda()

    if args.use_wandb:
        wandb.watch(model)
    chamfer = ChamferLoss()
    point = Point2PointLoss()
    xxloss = XXLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_epoch_loss = float("inf")

    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0
        epoch_chamfer_loss = 0
        epoch_point_loss = 0
        epoch_xxloss = 0
        count = 0
        pbar = tqdm(enumerate(trainloader))
        for i 

if __name__ == '__main__':
    args = parse_args()
    if args.use_wandb:
        wandb.init(config=args, project='KP-Net')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>start training<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>train done<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
