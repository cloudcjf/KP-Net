import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
import numpy as np
import argparse
from tqdm import tqdm, trange
import os
from tensorboardX import SummaryWriter
from torch.backends import cudnn
# from kittiloader import KittiDataset
from oxfordloader import Oxford_train_base
from oxfordloader import *
from model.models import KPNet
from model.losses import ChamferLoss, Point2PointLoss
import model.losses as loss
import evaluate

def parse_args():
    parser = argparse.ArgumentParser('RSKDD')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.00005)
    # parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--data_dir', type=str, default='',help='dir of dataset')
    # parser.add_argument('--seq', type=str, default='00', help='training sequence of Kitti dataset')
    parser.add_argument('--graph_k',type=int, default=20)
    parser.add_argument('--k',type=int, default=64)    # 后面考虑减小，因为目前的总点数才4096
    parser.add_argument('--nsample', type=int, default=128)    # 后面考虑减小，因为目前的总点数才4096 
    parser.add_argument('--desc_dim', type=int, default=32)
    parser.add_argument('--dilation_ratio', type=float, default=2.0)
    # parser.add_argument('--alpha', type=float, default=1.0, \
    #     help='ratio between chamfer loss and point to point loss')
    # parser.add_argument('--beta', type=float, default=1.0, \
    #     help='ratio between chamfer loss and point to matching loss')
    # parser.add_argument('--temperature', type=float, default=0.1)
    # parser.add_argument('--sigma_max', type=float, default=3.0, \
    #     help='predefined sigma upper bound')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt/', help='path to save model')

    # args below are from PointNetVLAD
    parser.add_argument('--positives_per_query', type=int, default=2,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--negatives_per_query', type=int, default=8,
                        help='Number of definite negatives in each training tuple [default: 18]')
    parser.add_argument('--hard_neg_per_query', type=int, default=1,
                        help='Number of definite negatives in each training tuple [default: 10]')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='test Batch Size during training [default: 6]')
    parser.add_argument('--eval_positives_per_query', type=int, default=2,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--eval_negatives_per_query', type=int, default=18,
                        help='Number of definite negatives in each training tuple [default: 18]')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num_points [default: 4096]')

    # loss function params start
    parser.add_argument('--margin_1', type=float, default=1,
                        help='Margin for hinge loss [default: 0.5]')
    parser.add_argument('--margin_2', type=float, default=0.2,
                        help='Margin for hinge loss [default: 0.2]')
    parser.add_argument('--loss_ignore_zero_batch',default=False,
                        help='If present, mean only batches with loss > 0.0')
    parser.add_argument('--loss_function', default='quadruplet', choices=['triplet', 'quadruplet'], help='triplet or quadruplet [default: quadruplet]')
    parser.add_argument('--loss_lazy',default=True,help='If present, do not use lazy variant of loss')
    parser.add_argument('--triplet_use_best_positives',default=False,help='If present, use best positives, otherwise use hardest positives')
    # loss function params end

    parser.add_argument('--resume', action='store_true',
                        help='If present, restore checkpoint and resume training')
    parser.add_argument('--dataset_folder', default='/mnt/Airdrop/benchmark_datasets/',
                        help='PointNetVlad Dataset Folder')
    parser.add_argument('--emb_dims', type=int, default=1024)   # dgcnn param
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')    # dgcnn param
    parser.add_argument('--log_dir', default='logs/', help='Log dir [default: log]')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                            help='random seed (default: 1)')
    parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
    return parser.parse_args()


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.enabled = True
    # trainset = KittiDataset(args.data_dir, srgs.seq_name, args.npoints)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, \
    #     shuffle=True, num_workers=args.num_workers, drop_last=True) # 此时点云已经加载进来
    train_writer = SummaryWriter(args.log_dir)
    trainset = Oxford_train_base(args)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, \
        shuffle=True, num_workers=args.num_workers, drop_last=True) # 此时点云已经加载进来
    model = KPNet(args)
    model = nn.DataParallel(model)  # 支持多gpu训练
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # recall 停止上升时,降低学习率
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=2, verbose=True, threshold = 0.1,min_lr=0.00001)
    # chamfer = ChamferLoss()
    # point = Point2PointLoss()

    best_epoch_loss = float("inf")
    epochs = trange(args.max_epoch, leave=True, desc="Epoch")  # 进度条
    for epoch in epochs:
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0
        # epoch_chamfer_loss = 0
        # epoch_point_loss = 0
        count = 0
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc="Batches")
        for i, data in pbar:
            queries, positives, negatives, other_neg = data # other_neg = 1 now
            # print(".................................................................queries here: ",queries.shape,positives.shape,negatives.shape,other_neg.shape)
            feed_tensor = torch.cat((queries, positives, negatives, other_neg), 1)  #  [batch_size, (queries_num + positive_num + negative_num + other_neg_num), 4096, 3]
            feed_tensor = feed_tensor.view((-1, args.num_points, 3))    #  [batch_size * (queries_num + positive_num + negative_num + other_neg_num), 4096, 3]
            feed_tensor = feed_tensor.cuda()
            # print('feed_tensor copied onto cuda')
            global_descriptor = model(feed_tensor)  # [args.batch_size*(queries_num + positive_num + negative_num + other_neg_num), 256]
            output_queries, output_positives, output_negatives, output_other_neg = torch.split(global_descriptor, [args.batch_size*1, args.batch_size*args.positives_per_query, args.batch_size*args.negatives_per_query, args.batch_size*1], dim=0)
            # 临时增加batch维度
            output_queries = output_queries.view(args.batch_size,1,256)
            output_positives = output_positives.view(args.batch_size,args.positives_per_query,256)
            output_negatives = output_negatives.view(args.batch_size,args.negatives_per_query,256)
            output_other_neg = output_other_neg.view(args.batch_size,1,256)
            loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, args.margin_1,\
                                args.margin_2, use_min=args.triplet_use_best_positives, lazy=args.loss_lazy,\
                                ignore_zero_loss=args.loss_ignore_zero_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_writer.add_scalar("Batch loss", loss.cpu().item(), int(epoch)*len(trainloader)*int(args.batch_size)+count)
            train_writer.add_scalar("learn rate", optimizer.param_groups[0]['lr'], int(epoch)*len(trainloader)*int(args.batch_size)+count)
            epoch_loss = epoch_loss + float(loss)
            count += 1
            

        epoch_loss = epoch_loss / count
        print('Epoch {} finished. Loss: {:.3f}'.format(epoch+1, epoch_loss))

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss    # update best epoch loss
            torch.save(model.state_dict(), args.ckpt_dir+'best_'+str(epoch)+'.pth')   # 后面考虑optimizer的保存

        eval_recall = evaluate.evaluate_model(model,args)
        scheduler.step(eval_recall)
        train_writer.add_scalar("Value of Recall: ",eval_recall, epoch)


if __name__ == '__main__':
    args = parse_args()
    if args.loss_function == 'quadruplet':
        # 有了第二项约束，类内间距离应该比类内距离大
        print("use quadruplet_loss")
        loss_function = loss.quadruplet_loss
    else:
        print("use triplet_loss_wrapper")
        loss_function = loss.triplet_loss_wrapper
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>start training<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>train done<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
