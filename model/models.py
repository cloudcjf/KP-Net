#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import random_dilation_encoding

def knn(x, k):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
        return idx

def get_graph_feature(x, k, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k)   # (batch_size, num_points, k)
        device = torch.device('cuda')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
        return feature


class KPNet(nn.Module):
    def __init__(self, args):
        super(KPNet, self).__init__()
        self.ninput = args.num_points      # num of input points
        self.nsample = args.nsample     # num of key points & decs
        self.k = args.k     # KNN for keypoints
        self.graph_k = args.graph_k # KNN for DGCNN
        self.dilation_ratio = args.dilation_ratio
        
        # below is detector network structure
        self.in_channel = 4 # 点云输入：x, y, z, d, 后面可以增加其他特征维度C
        self.C1 = 64
        self.C2 = 128
        self.C3 = 256
        self.desc_dim = args.desc_dim


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

        # below is descriptor network structure
        self.C1_ = 64
        self.C2_ = 128
        self.C3_ = 128
        self.C_detector = 256   # 根据前面detector的输出维度决定
        self.in_channel_ = 4    # relative_x, relative_y, relative_z, d, 后面可以增加其他特征维度C
        self.conv1_ = nn.Sequential(nn.Conv2d(self.in_channel_, self.C1_, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C1_),
                                   nn.ReLU())
        self.conv2_ = nn.Sequential(nn.Conv2d(self.C1_, self.C2_, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C2_),
                                   nn.ReLU())
        self.conv3_ = nn.Sequential(nn.Conv2d(self.C2_, self.C3_, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C3_),
                                   nn.ReLU())
        
        self.conv4 = nn.Sequential(nn.Conv2d(2*self.C3_+self.C_detector, self.C2_, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C2_),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(self.C2_, self.desc_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.desc_dim),
                                   nn.ReLU())

        # below is dgcnn network structure
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)  # SG_PR设为32
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.output_channels = 128  # output global desc dimension
        self.filters_1 = 64
        self.filters_2 = 64
        self.filters_3 = 32
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.output_channels*2, self.output_channels*2))
        torch.nn.init.xavier_uniform_(self.weight_matrix)   # 均匀分布

        self.dgcnn_s_conv1 = nn.Sequential(nn.Conv2d(3*2, self.filters_1, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        # descriptor dimension = 32                           
        self.dgcnn_f_conv1 = nn.Sequential(nn.Conv2d(32*2, self.filters_1, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv2 = nn.Sequential(nn.Conv2d(self.filters_1*2, self.filters_2, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.dgcnn_conv3 = nn.Sequential(nn.Conv2d(self.filters_2*2, self.filters_3, kernel_size=1, bias=False),
        #                            self.bn3,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, self.output_channels)

    def dgcnn(self,x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.graph_k)    # [B, 6, S, 10] || [B, 64, S, 10]
        if x.size(1) == 6:  # 3*2
            x = self.dgcnn_s_conv1(x)   # [B,64,S,10]
        else:   # x.size(1) == 32*2
            x = self.dgcnn_f_conv1(x)   # [B,64,S,10]
        x1 = x.max(dim=-1, keepdim=False)[0]    # [B,64,S]

        x = get_graph_feature(x1, k=self.graph_k)
        x = self.dgcnn_conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]    # [B,64,S]

        x = get_graph_feature(x2, k=self.graph_k)
        x = self.dgcnn_conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]    # [B,128,S]

        x = get_graph_feature(x3, k=self.graph_k)
        x = self.dgcnn_conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]    # [B,256,S]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # [B,512,S]

        x = self.dgcnn_conv5(x)                 # [B,1024,S]
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)   # [B,1024]
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)   # [B,1024]
        x = torch.cat((x1, x2), 1)  # [B,2048]

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x) # [B,128]
        # print("-----------------------------------------",x_)
        # print("-----------------------------------------",x)
        return x


    def detector(self, x):
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
            random_cluster: [B, 4+C, S, K], 里面knn的坐标是相对坐标，可作为初始特征
            random_xyz:     [B, S, K, 3], 里面knn的坐标是绝对坐标
        '''
        random_cluster, random_xyz = random_dilation_encoding(x_sample, x, self.k, self.dilation_ratio)
        # print("random_cluster shape: ", random_cluster.shape)   [22,4,S,K]
        # Attentive points aggregation
        #  MLP
        #  embedding: [B, 256, S, K]
        embedding = self.conv3(self.conv2(self.conv1(random_cluster)))
        #  Maxpool
        #  x1: [B, 1, S, K]
        x1 = torch.max(embedding, dim=1, keepdim=True)[0]
        #  x1: [B, S, K]
        x1 = x1.squeeze(dim=1)
        # Softmax,归一化到0-1之间
        #  attentive_weights: [B, S, K]
        attentive_weights = F.softmax(x1, dim=-1)
        #  把权重在xyz三个维度上扩展，score_xyz: [B, 3, S, K]
        score_xyz = attentive_weights.unsqueeze(1).repeat(1,3,1,1)
        #  random_xyz.permute以后的维度顺序: [B, 3, S, K]
        xyz_scored = torch.mul(random_xyz.permute(0,3,1,2),score_xyz)   # torch.mul对应元素相乘
        #  keypoints: [B, 3, S]
        # 总结一下关键点的构建：
        #           1. 在原始点云中随机选取512个关键点的候选点
        #           2. 利用一系列操作得到这些候选点的KNN邻点
        #           3. 获得KNN与候选点的相对坐标，相对距离，以及KNN点自身的法向量，曲率特征
        #           4. 上述信息利用网络算出每个KNN点的权重
        #           5. 对这些KNN点进行加权求和，就得到最终的关键点坐标
        keypoints = torch.sum(xyz_scored, dim=-1, keepdim=False)
        #  把权重在特征的C3个维度上扩展，score_feature: [B, 256, S, K]
        score_feature = attentive_weights.unsqueeze(1).repeat(1,self.C3,1,1)
        attentive_feature_map = torch.mul(embedding, score_feature)
        #  加权求和，global_cluster_feature: [B, 256, S]
        global_cluster_feature = torch.sum(attentive_feature_map, dim=-1, keepdim=False)
        # MLP
        saliency_uncertainty = self.mlp3(self.mlp2(self.mlp1(global_cluster_feature)))
        #  why add 0.001 --> to avoid zero
        saliency_uncertainty = self.softplus(saliency_uncertainty) + 0.001
        saliency_uncertainty = saliency_uncertainty.squeeze(dim=1)

        # TODO 根据上面求出的显著性提取出最重要的S个关键点及对应的random_cluster

        return keypoints, saliency_uncertainty, random_cluster, attentive_feature_map


    def descriptor(self, random_cluster, attentive_feature_map):
            # MLP 这里x1不直接用embedding是因为desc和det的MLP参数是不一样的
            #   random_cluster: [B, 4+C, S, K]
            x1 = self.conv3_(self.conv2_(self.conv1_(random_cluster)))
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


    def forward(self, pointcloud):
        keypoints, sigmas, random_cluster, attentive_feature_map = self.detector(pointcloud)
        # print('................................................keypoints shape: ', keypoints.shape)   #  [B,3,512]
        desc = self.descriptor(random_cluster, attentive_feature_map)
        # print('................................................descs shape: ', desc.shape)      #  [B,32,512]
        keypoints_feature = self.dgcnn(keypoints)   # [B, 128]
        desc_feature = self.dgcnn(desc)             # [B, 128]
        embedding = torch.cat((keypoints_feature,desc_feature), dim=1)      # [B, 256]
        embedding_norm = torch.norm(embedding, dim=1,keepdim=True)
        embedding = embedding / embedding_norm
        # batch_size = embedding.shape[0]
        # graph上每个node都乘上一个权重，然后再取平均值，成为这个graph的全局描述
        # global_context = torch.matmul(embedding, self.weight_matrix) # Bx256
        # transformed_global = torch.tanh(global_context) # 论文中的c Bx256
        # sigmoid_scores = torch.sigmoid(torch.matmul(embedding,transformed_global.view(batch_size,-1, 1)))   #weights      BxNx1
        # print('................................................sigmoid_scores shape: ', sigmoid_scores.shape)
        # representation = torch.matmul(embedding.permute(0,2,1),sigmoid_scores)    #论文中的e Bx32x1
        # print('................................................representation shape: ', representation.shape)
        # # return representation, sigmoid_scores
        # return representation
        return embedding 



