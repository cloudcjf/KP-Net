import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChamferLoss(nn.Module):
    """
    Calculate probabilistic chamfer distance between keypoints1 and keypoints2
    Input:
        keypoints1: [B,3,M]
        keypoints2: [B,3,M]
        sigma1: [B,M]
        sigma2: [B,M]
    """
    def __init__(self):
        super(ChamferLoss, self).__init__()
    
    def forward(self, keypoints1, keypoints2, sigma1, sigma2):
        B, M = keypoints1.size()[0], keypoints1.size()[2]
        #  normally N = M
        N = keypoints2.size()[2]
        N = keypoints2.size()[2]
        '''
        algrithm for calculating distances between different number of points
        set1: expand 4th dimension to set2's number
        set2: expand 3th dimension to set1's number
        '''
        keypoints1_expanded = keypoints1.unsqueeze(3).expand(B,3,M,N)
        keypoints2_expanded = keypoints2.unsqueeze(2).expand(B,3,M,N)

        # diff: [B, M, M]
        diff = torch.norm(keypoints1_expanded-keypoints2_expanded, dim=1, keepdim=False)

        if sigma1 is None or sigma2 is None:
            min_dist_forward, _ = torch.min(diff, dim=2, keepdim=False)
            forward_loss = min_dist_forward.mean()

            min_dist_backward, _ = torch.min(diff, dim=1, keepdim=False)
            backward_loss = min_dist_backward.mean()

            loss = forward_loss + backward_loss
        
        else:
            min_dist_forward, min_dist_forward_I = torch.min(diff, dim=2, keepdim=False)
            selected_sigma_2 = torch.gather(sigma2, dim=1, index=min_dist_forward_I)
            sigma_forward = (sigma1 + selected_sigma_2)/2
            forward_loss = (torch.log(sigma_forward)+min_dist_forward/sigma_forward).mean()

            min_dist_backward, min_dist_backward_I = torch.min(diff, dim=1, keepdim=False)
            selected_sigma_1 = torch.gather(sigma1, dim=1, index=min_dist_backward_I)
            sigma_backward = (sigma2 + selected_sigma_1)/2
            backward_loss = (torch.log(sigma_backward)+min_dist_backward/sigma_backward).mean()

            loss = forward_loss + backward_loss
        return loss

class Point2PointLoss(nn.Module):
    '''
    Calculate point-to-point loss between keypoints and pc
    Input:
        keypoints: [B,3,M]
        pc: [B,3,N]
    '''
    def __init__(self):
        super(Point2PointLoss, self).__init__()
    
    def forward(self, keypoints, pc):
        B, M = keypoints.size()[0], keypoints.size()[2]
        N = pc.size()[2]
        keypoints_expanded = keypoints.unsqueeze(3).expand(B,3,M,N)
        pc_expanded = pc.unsqueeze(2).expand(B,3,M,N)
        diff = torch.norm(keypoints_expanded-pc_expanded, dim=1, keepdim=False)
        min_dist, _ = torch.min(diff, dim=2, keepdim=False)
        return torch.mean(min_dist)

# loss which affects place recognition performance
def best_pos_distance(query, pos_vecs):
    # print("query shape: ",query.shape)
    num_pos = pos_vecs.shape[1]
    query_copies = query.repeat(1, int(num_pos), 1)
    # print("query_copies shape: ",query_copies.shape)
    diff = ((pos_vecs - query_copies) ** 2).sum(2)
    min_pos, _ = diff.min(1)
    max_pos, _ = diff.max(1)
    return min_pos, max_pos


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    loss = loss.clamp(min=0.0)
    # print("........................................................clamp loss: ",loss)
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()
    
    # print("........................................................triplet loss: ",triplet_loss)
    return triplet_loss


def triplet_loss_wrapper(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    return triplet_loss(q_vec, pos_vecs, neg_vecs, m1, use_min, lazy, ignore_zero_loss)


def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:#获得正样本最小的距离还是最大
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = m1 + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    loss = loss.clamp(min=0.0)
    # 是否只看max
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    # 是否忽略为0的loss
    if ignore_zero_loss:
        # gt 若大于1e-16则为1
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()

    other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies) ** 2).sum(2)
    second_loss = second_loss.clamp(min=0.0)
    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.sum(1)
    # 是否忽略为0的loss
    if ignore_zero_loss:
        # gt 若大于1e-16则为1
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss
    return total_loss
