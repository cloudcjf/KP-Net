#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # matmul: tensor multiply; -1: auto calculate how much should be
    dist = -2 * torch.matmul(src, dst.permute(0,2,1))
    # torch.view: resize
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input point cloud [B, N, 3+C] tensor
        idx: point index [B, S, K] tensor
    output:
        indexed points: [B, S, K, 3+C] tensor
    """
    device = points.device # copy these tensors to GPU
    B = points.shape[0]
    view_shape = list(idx.shape)  # view_shape: [B, S, K]
    view_shape[1:] = [1]*(len(view_shape)-1)  # [1]*2 --> [1, 1]  view_shape now: [B, 1, 1]
    repeat_shape = list(idx.shape)  # repeat_shape: [B, S, K]
    repeat_shape[0] = 1  # repeat_shape now: [1, S, K]
    '''
    batch_indices: [B, S, K]
        第一个维度B: 有B帧点云
        第二个维度S: 每帧点云采样的S个点
        第三个维度K: 每个点的K个最近邻
        这里求new_points非常有trick,points的第一维遍历batch_indices确定点云帧数,第二维遍历idx确定该帧点云的某个采样点,
        这样遍历完得到B*S*K个点,再用':'获得点的坐标及特征
    new_points: [B, S, K, 3+C]
    '''
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # [B, S]
    new_points = points[batch_indices, idx, :]
    return new_points

def query_knn_point(k, query, pc):
    """
    Input:
        k: number of neighbor points
        query: query points [B, S, 3]
        pc: point cloud [B, N, 3]
      
    Output:
        normed_knn_points: [B, S, k, 3]
        knn_ids: [B, S, k] index of knn points
    """
    # permute: exchange dim order
    '''
    before: query: [B, S, 3] pc: [B, N, 3]
    after:  query: [B, 3, S, 1] database: [B, 3, 1, N]
    '''
    query = query.permute(0,2,1).unsqueeze(3)
    database = pc.permute(0,2,1).unsqueeze(2)
    # after broadcast, query-database: [B, 3, S, N], 3这个维度上现在不再是点的坐标,而是两点坐标分量差值
    # norm: [B, S, N],此时每个元素都是两点之间的归一化距离, dim=1: 第二维被缩减
    norm = torch.norm(query-database, dim=1, keepdim=False)
    # order function k-nearest
    '''
    knn_ids: [B, S, k] tensor,S个采样点,每个点有K个最近邻点
    '''
    knn_d, knn_ids = torch.topk(norm, k=k, dim=2, largest=False, sorted=True)
    """
        Input:
            pc: input point cloud [B, N, 3+C] tensor
            knn_ids: point index [B, S, K] tensor
        output:
            knn_points: [B, S, K, 3+C] tensor
    """
    knn_points = index_points(pc, knn_ids)
    #  centroids: [B, S, 3+C] B帧点云,每帧S个点,对应S个centroids
    centroids = torch.mean(knn_points, dim=2)
    # 使centroids和knn_points维度一致, 求knn各个点相对于中心的相对坐标
    centroids = centroids.unsqueeze(2).repeat(1,1,k,1)
    normed_knn_points = knn_points - centroids
    return normed_knn_points, knn_ids

def random_dilation_encoding(x_sample, x, k, n):
    '''
     Input:
         x_sample: [B, nsample, 3+C]
         x: [B, N, 3+C]
         k: number of neighbors
         n: dilation ratio
     Output:
         dilation_group: random dilation cluster [B, 4+C, nsample, k]
         dilation_xyz: xyz of random dilation cluster [B, 3, nsample, k]
     '''
    xyz_sample = x_sample[:,:,:3]  # 在采样点中取每点的前三个值
    xyz = x[:,:,:3]  # 在原始输入点云中取每点的前三个值
    feature = x[:,:,3:]  # 在原始输入点云中取3+C的前三个以后的值,即C
    '''
    获得knn个点的相对于中心的坐标值和索引,注意这里的knn是2k个,是膨胀过后的
    knn_idx: [B, S, 2K]
    '''
    _, knn_idx = query_knn_point(int(k*n), xyz_sample, xyz)
    # return a random array ,data range from 0 ~ k*n-1
    # 在0~k*n-1 中随机提取k个索引
    rand_idx = torch.randperm(int(k*n))[:k]
    dilation_idx = knn_idx[:,:,rand_idx]  # [B, S, K]
    # 提取原始坐标和特征: [B, S, K, 3] || [B, S, K, C]
    dilation_xyz = index_points(xyz, dilation_idx) # 每个KNN邻点的坐标
    dilation_feature = index_points(feature, dilation_idx) # 每个KNN邻点的特征
    # [B, S, 3] --> [B, S, K, 3]
    xyz_expand = xyz_sample.unsqueeze(2).repeat(1,1,k,1) # 每个采样点的坐标，与xyz_sample的区别仅仅是维度转换
    dilation_xyz_resi = dilation_xyz - xyz_expand  # [B, S, K, 3]
    dilation_xyz_dis = torch.norm(dilation_xyz_resi,dim=-1,keepdim=True)  # [B, S, K, 1]
    dilation_group = torch.cat((dilation_xyz_dis, dilation_xyz_resi),dim=-1)  # [B, S, K, 4]
    dilation_group = torch.cat((dilation_group, dilation_feature), dim=-1)  # [B, S, K, 4+C]
    dilation_group = dilation_group.permute(0,3,1,2)  # [B, 4+C, S, K]
    return dilation_group, dilation_xyz