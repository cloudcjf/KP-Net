#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import random_dilation_encoding
class KPNet(nn.Module):
    '''
    Input:
        x: point cloud [B,N,3+C]
    Output:
        keypoints: [B,3,M]
        sigmas: [B,M]
        desc: [B,C_d,M]
    '''
    def __init__(self, args):
        super(KPNet, self).__init__()
        self.ninput = args.npoints      # num of input points
        self.nsample = args.nsample     # num of key points & decs
        self.k = args.k     # KNN
        self.dilation_ratio = args.dilation_ratio   # 膨胀率

        self.C1 = 64
        self.C2 = 128
        self.C3 = 256
        self.in_channel = 8  # 这里就是点云的 3+C, 坐标,强度,法向量,曲率通道数
        # nn.Conv2d: 参数1:输入张量的通道数; 参数2:输出的通道数,即卷积核个数; 参数3:卷积核尺寸
        # nn.BatchNorm2d: 批量归一化,为了使数据不过大,分布更合理, 参数num_features: 输入数据的通道数或特征数
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channel, self.C1, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.C1, self.C2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C2),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(self.C2, self.C3, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C3),
                                   nn.ReLU())
        # nn.Conv1d: 卷积核大小: kernel_size * in_channel
        self.mlp1 = nn.Sequential(nn.Conv1d(self.C3, self.C3, kernel_size=1),
                                  nn.BatchNorm1d(self.C3),
                                  nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(self.C3, self.C3, kernel_size=1),
                                  nn.BatchNorm1d(self.C3),
                                  nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(self.C3, 1, kernel_size=1))
        '''
        softplus: 激活函数,ReLU的变种
        softplus(x) = 1/β * log(1 + exp(β*x))
        '''
        self.softplus = nn.Softplus()


    # cjf node embedding by dgcnn
    def dgcnn_conv_pass(self, x):
        self.k = self.args.K
        xyz = x[:,:3,:] # Bx3xN
        sem = x[:,3:,:]   # BxfxN
        # dgcnn on spatial space
        xyz = dgcnn.get_graph_feature(xyz, self.k)    #Bx6xNxk
        xyz = self.dgcnn_s_conv1(xyz)
        xyz1 = xyz.max(dim=-1, keepdim=False)[0]
        xyz = dgcnn.get_graph_feature(xyz1, k=self.k)
        xyz = self.dgcnn_s_conv2(xyz)
        xyz2 = xyz.max(dim=-1, keepdim=False)[0]
        xyz = dgcnn.get_graph_feature(xyz2, k=self.k)
        xyz = self.dgcnn_s_conv3(xyz)
        xyz3 = xyz.max(dim=-1, keepdim=False)[0]
        # dgcnn on feature space
        # by cjf
        # 这里的feature对应的网络层参数需要修改
        # by cjf
        sem = dgcnn.get_graph_feature(sem, self.k)  # Bx2fxNxk
        sem = self.dgcnn_f_conv1(sem)
        sem1 = sem.max(dim=-1, keepdim=False)[0]
        sem = dgcnn.get_graph_feature(sem1, k=self.k)
        sem = self.dgcnn_f_conv2(sem)
        sem2 = sem.max(dim=-1, keepdim=False)[0]
        sem = dgcnn.get_graph_feature(sem2, k=self.k)
        sem = self.dgcnn_f_conv3(sem)
        sem3 = sem.max(dim=-1, keepdim=False)[0]

        x = torch.cat((xyz3, sem3), dim=1)
        # x = self.dgcnn_conv_all(x)
        x = self.dgcnn_conv_end(x)
        # print(x.shape)

        x = x.permute(0, 2, 1)  # [node_num, 32]
        return x

    # 定义前向函数
    def forward(self, point_cloud):
        '''
        Input:
            point_cloud: [B,N,3+C]
        '''
        # random sample, B,3+C不变,点数N变小
        randIdx = torch.randperm(self.ninput)[:self.nsample]
        x_sample = point_cloud[:,randIdx,:]

        # random dilation cluster
        '''
        Input: 
            x: [B,N,3+C]; x_sample: [B,S,3+C]
        Output:
            random_cluster: [B, 4+C, S, K]
            random_xyz:     [B, S, K, 3]
        '''
        random_cluster, random_xyz = random_dilation_encoding(x_sample, point_cloud, self.k, self.dilation_ratio)
        # random_cluster 作为keypoint的 local feature送入GCN



        # Attentive points aggregation
        #  MLP
        #  embedding: [B, 256, S, K]
        embedding = self.conv3(self.conv2(self.conv1(random_cluster)))
        #  Maxpool
        #  x1: [B, 1, S, K]
        x1 = torch.max(embedding, dim=1, keepdim=True)[0]
        #  x1: [B, S, K]
        x1 = x1.squeeze(dim=1)
        #  attentive_weights: [B, S, K]
        attentive_weights = F.softmax(x1, dim=-1)
        #  score_xyz: [B, 3, S, K]
        score_xyz = attentive_weights.unsqueeze(1).repeat(1,3,1,1)
        #  random_xyz.permute: [B, 3, S, K]
        xyz_scored = torch.mul(random_xyz.permute(0,3,1,2),score_xyz)
        #  keypoints: [B, 3, S]
        keypoints = torch.sum(xyz_scored, dim=-1, keepdim=False)
        #  score_feature: [B, 256, S, K]
        score_feature = attentive_weights.unsqueeze(1).repeat(1,self.C3,1,1)
        attentive_feature_map = torch.mul(embedding, score_feature)
        #  keypoints: [B, 256, S]
        global_cluster_feature = torch.sum(attentive_feature_map, dim=-1, keepdim=False)
        saliency_uncertainty = self.mlp3(self.mlp2(self.mlp1(global_cluster_feature)))
        #  why add 0.001 --> to avoid zero
        saliency_uncertainty = self.softplus(saliency_uncertainty) + 0.001
        saliency_uncertainty = saliency_uncertainty.squeeze(dim=1)
        print('2222222222222222222222222222222222222222222222')
        # GCN embedding



        # 

        # return keypoints, saliency_uncertainty, random_cluster, attentive_feature_map









# subclass of torch.nn class
class Detector(nn.Module):
    '''
    Input:
        x: [B, N, 3+C]
    Output:
        keypoints: [B,3,M] 
        saliency_uncertainty: [B,M] 
        random_cluster: [B,4+C,M,k] 
        attentive_feature_map: [B, C_a, M,k]
    '''
    def __init__(self, args):
        super(Detector, self).__init__()
        self.ninput = args.npoints      # num of input points
        self.nsample = args.nsample     # num of key points & decs
        self.k = args.k     # KNN
        self.dilation_ratio = args.dilation_ratio   # 膨胀率

        self.C1 = 64
        self.C2 = 128
        self.C3 = 256
        self.in_channel = 8  # 这里就是点云的 3+C, 坐标,强度,法向量,曲率通道数
        # nn.Conv2d: 参数1:输入张量的通道数; 参数2:输出的通道数,即卷积核个数; 参数3:卷积核尺寸
        # nn.BatchNorm2d: 批量归一化,为了使数据不过大,分布更合理, 参数num_features: 输入数据的通道数或特征数
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channel, self.C1, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.C1, self.C2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C2),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(self.C2, self.C3, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C3),
                                   nn.ReLU())
        # nn.Conv1d: 卷积核大小: kernel_size * in_channel
        self.mlp1 = nn.Sequential(nn.Conv1d(self.C3, self.C3, kernel_size=1),
                                  nn.BatchNorm1d(self.C3),
                                  nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(self.C3, self.C3, kernel_size=1),
                                  nn.BatchNorm1d(self.C3),
                                  nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(self.C3, 1, kernel_size=1))
        '''
        softplus: 激活函数,ReLU的变种
        softplus(x) = 1/β * log(1 + exp(β*x))
        '''
        self.softplus = nn.Softplus()

    def forward(self, x):
        '''
        Input:
            x: [B,N,3+C]
        '''
        # random sample, B,3+C不变,点数N变小
        randIdx = torch.randperm(self.ninput)[:self.nsample]
        x_sample = x[:,randIdx,:]

        # random dilation cluster
        '''
        Input: 
            x: [B,N,3+C]; x_sample: [B,S,3+C]
        Output:
            random_cluster: [B, 4+C, S, K]
            random_xyz:     [B, S, K, 3]
        '''
        random_cluster, random_xyz = random_dilation_encoding(x_sample, x, self.k, self.dilation_ratio)

        # Attentive points aggregation
        #  MLP
        #  embedding: [B, 256, S, K]
        embedding = self.conv3(self.conv2(self.conv1(random_cluster)))
        #  Maxpool
        #  x1: [B, 1, S, K]
        x1 = torch.max(embedding, dim=1, keepdim=True)[0]
        #  x1: [B, S, K]
        x1 = x1.squeeze(dim=1)
        #  attentive_weights: [B, S, K]
        attentive_weights = F.softmax(x1, dim=-1)
        #  score_xyz: [B, 3, S, K]
        score_xyz = attentive_weights.unsqueeze(1).repeat(1,3,1,1)
        #  random_xyz.permute: [B, 3, S, K]
        xyz_scored = torch.mul(random_xyz.permute(0,3,1,2),score_xyz)
        #  keypoints: [B, 3, S]
        keypoints = torch.sum(xyz_scored, dim=-1, keepdim=False)
        #  score_feature: [B, 256, S, K]
        score_feature = attentive_weights.unsqueeze(1).repeat(1,self.C3,1,1)
        attentive_feature_map = torch.mul(embedding, score_feature)
        #  keypoints: [B, 256, S]
        global_cluster_feature = torch.sum(attentive_feature_map, dim=-1, keepdim=False)
        saliency_uncertainty = self.mlp3(self.mlp2(self.mlp1(global_cluster_feature)))
        #  why add 0.001 --> to avoid zero
        saliency_uncertainty = self.softplus(saliency_uncertainty) + 0.001
        saliency_uncertainty = saliency_uncertainty.squeeze(dim=1)
        print('2222222222222222222222222222222222222222222222')
        return keypoints, saliency_uncertainty, random_cluster, attentive_feature_map
# subclass of torch.nn class
class Descriptor(nn.Module):
    '''
    Input:
        random_cluster: [B,4+C,M,k] 
        attentive_feature_map: [B, C_a, M,k]
    Output:
        desc: [B,C_f,M]
    '''
    def __init__(self, args):
        super(Descriptor, self).__init__()

        self.C1 = 64
        self.C2 = 128
        self.C3 = 128
        self.C_detector = 256

        self.desc_dim = args.desc_dim
        self.in_channel = 8
        self.k = args.k

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channel, self.C1, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.C1, self.C2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C2),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(self.C2, self.C3, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C3),
                                   nn.ReLU())
        
        self.conv4 = nn.Sequential(nn.Conv2d(2*self.C3+self.C_detector, self.C2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C2),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(self.C2, self.desc_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.desc_dim),
                                   nn.ReLU())
        
    def forward(self, random_cluster, attentive_feature_map):
        # MLP
        x1 = self.conv3(self.conv2(self.conv1(random_cluster)))
        # Maxpool
        x2 = torch.max(x1, dim=3, keepdim=True)[0]
        # 1*Cf K个
        x2 = x2.repeat(1,1,1,self.k)
        # K*Cf
        x2 = torch.cat((x2, x1),dim=1) # [B,2*C3,N,k]
        # K*Ca
        x2 = torch.cat((x2, attentive_feature_map), dim=1)
        x2 = self.conv5(self.conv4(x2))
        desc = torch.max(x2, dim=3, keepdim=False)[0]
        return desc
# subclass of torch.nn class
class RSKDD(nn.Module):
    '''
    Input:
        x: point cloud [B,N,3+C]
    Output:
        keypoints: [B,3,M]
        sigmas: [B,M]
        desc: [B,C_d,M]
    '''
    def __init__(self, args):
        super(RSKDD, self).__init__()
        # RSKDD网络里还有两个子网络
        self.detector = Detector(args)
        self.descriptor = Descriptor(args)
    # 定义前向函数
    def forward(self, x):
        keypoints, sigmas, random_cluster, attentive_feature_map = self.detector(x)  # detector的输出
        desc = self.descriptor(random_cluster, attentive_feature_map)  # descriptor的输出

        return keypoints, sigmas, desc