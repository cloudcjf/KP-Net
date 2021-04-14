#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import pickle
import random
from time import time
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree
import torch
from tqdm import tqdm


load_fast=True
FEATURE_OUTPUT_DIM = 256
DATASET_FOLDER = '/mnt/Airdrop/benchmark_datasets/'

'''
    loading pointclouds functions
'''

def get_queries_dict(filename):
    # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries


def get_sets_dict(filename):
    #[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Sets Loaded.")
        return trajectories


def load_pc_file(filename):
    # returns Nx3 matrix
    pc = np.fromfile(os.path.join(DATASET_FOLDER, filename), dtype=np.float64)

    if(pc.shape[0] != 4096*3):
        print("Error in pointcloud shape")
        return np.array([])

    pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return pc


def load_pc_files(filenames):
    pcs = []
    for filename in filenames:
        # log_string(filename)
        pc = load_pc_file(filename)
        if(pc.shape[0] != 4096):
            continue
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        rotation_angle = (np.random.uniform()*np.pi) - np.pi/2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.005, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    # get query tuple for dictionary entry
    # return list [query,positives,negatives]
    start = time()
    query = load_pc_file(dict_value["query"])  # Nx3
    random.shuffle(dict_value["positives"])
    pos_files = []

    # 不必考虑正样本是否充足，因为之前判断过
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    positives = load_pc_files(pos_files)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        # 如果hard不够，再进行补充
        while(len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_pc_files(neg_files)

    # log_string("load time: ",time()-start)
    # 是否需要额外的neg（Quadruplet loss需要）
    if other_neg is False:
        return [query, positives, negatives]
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        # 减去与neighbors公共有的部分，剩下既不进也不远的那些部分
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)
        if(len(possible_negs) == 0):
            return [query, positives, negatives, np.array([])]
        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"]) # 就一个

        return [query, positives, negatives, neg2]


def get_rotated_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    query = load_pc_file(dict_value["query"])  # Nx3
    q_rot = rotate_point_cloud(np.expand_dims(query, axis=0))
    q_rot = np.squeeze(q_rot)

    random.shuffle(dict_value["positives"])
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    positives = load_pc_files(pos_files)
    p_rot = rotate_point_cloud(positives)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while(len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_pc_files(neg_files)
    n_rot = rotate_point_cloud(negatives)

    if other_neg is False:
        return [q_rot, p_rot, n_rot]

    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
        n2_rot = rotate_point_cloud(np.expand_dims(neg2, axis=0))
        n2_rot = np.squeeze(n2_rot)

        return [q_rot, p_rot, n_rot, n2_rot]


def get_jittered_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    query = load_pc_file(dict_value["query"])  # Nx3
    #q_rot= rotate_point_cloud(np.expand_dims(query, axis=0))
    q_jit = jitter_point_cloud(np.expand_dims(query, axis=0))
    q_jit = np.squeeze(q_jit)

    random.shuffle(dict_value["positives"])
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    positives = load_pc_files(pos_files)
    p_jit = jitter_point_cloud(positives)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while(len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_pc_files(neg_files)
    n_jit = jitter_point_cloud(negatives)

    if other_neg is False:
        return [q_jit, p_jit, n_jit]

    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
        n2_jit = jitter_point_cloud(np.expand_dims(neg2, axis=0))
        n2_jit = np.squeeze(n2_jit)

        return [q_jit, p_jit, n_jit, n2_jit]

'''
    loading pointclouds functions
'''
TRAIN_FILE = 'generating_queries/training_queries_baseline.pickle'
TEST_FILE = 'generating_queries/test_queries_baseline.pickle'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load dictionary of training queries
# if not para.args.eval:
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
TEST_QUERIES = get_queries_dict(TEST_FILE)

# else:
#     TRAINING_QUERIES = []
#     TEST_QUERIES = []
HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []
TRAINING_POINT_CLOUD = []


# 这里最好能跟数据生成同步
# TRAINING_POINT_CLOUD.shape: [21711, 4096, 3]
if load_fast:
    print("start load fast")
    DIR="./generating_queries/"
    if os.path.exists(DIR+"TRAINING_POINT_CLOUD.npy"):
        TRAINING_POINT_CLOUD = np.load(DIR+"TRAINING_POINT_CLOUD.npy")
        print("load npy size : ",TRAINING_POINT_CLOUD.shape)
    else:
        for i in tqdm(range(len(TRAINING_QUERIES))):
            filename = TRAINING_QUERIES[i]["query"]
            pc = load_pc_file(filename)
            TRAINING_POINT_CLOUD.append(pc)
        TRAINING_POINT_CLOUD = np.asarray(TRAINING_POINT_CLOUD).reshape(-1,4096,3)
        np.save(DIR+"TRAINING_POINT_CLOUD.npy", TRAINING_POINT_CLOUD)
        print("save npy size: ", TRAINING_POINT_CLOUD.shape)
else:
    TRAINING_POINT_CLOUD = []
    print("load_fast "+str(load_fast))





def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

def get_query_tuple_fast(item, dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    # get query tuple for dictionary entry
    # return list [query,positives,negatives]
    start = time()
    query = TRAINING_POINT_CLOUD[item] # 就一个点云
    random.shuffle(dict_value["positives"])
    # 不必考虑正样本是否充足，因为之前判断过
    positives = TRAINING_POINT_CLOUD[(dict_value["positives"][:num_pos])]

    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        neg_indices=dict_value["negatives"][:num_neg]
    else:
        neg_indices.append(hard_neg)
        j = 0
        # 如果hard不够，再进行补充
        neg_indices = list(flat(neg_indices))
        while(len(neg_indices) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    neg_indices = list(flat(neg_indices))
    negatives = TRAINING_POINT_CLOUD[neg_indices]

    # log_string("load time: ",time()-start)
    # 是否需要额外的neg（Quadruplet loss需要）
    if other_neg is False:
        return [query, positives, negatives]
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        # 减去与neighbors公共有的部分，剩下既不进也不远的那些部分
        neighbors = list(flat(neighbors))
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)
        if(len(possible_negs) == 0):
            return [query, positives, negatives, np.array([])]
        neg2 = TRAINING_POINT_CLOUD[possible_negs[0]] # 就一个

        return [query, positives, negatives, neg2]

def get_random_hard_negatives(query_vec, random_negs, hard_neg_num):
    global TRAINING_LATENT_VECTORS

    latent_vecs = []
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])

    latent_vecs = np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]), k=hard_neg_num)
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs = hard_negs.tolist()
    return hard_negs

def get_feature_representation(filename, model):
    model.eval()
    queries = load_pc_files([filename])
    queries = np.expand_dims(queries, axis=1)
    # if(BATCH_NUM_QUERIES-1>0):
    #    fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
    #    q=np.vstack((queries,fake_queries))
    # else:
    #    q=queries
    with torch.no_grad():
        q = torch.from_numpy(queries).float()
        q = q.to(device)
        output = model(q)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    model.train()
    return output

# 设置成随机抽取的
class Oxford_train_base(Dataset):
    def __init__(self, args):
        self.num_points = args.num_points
        self.positives_per_query=args.positives_per_query
        self.negatives_per_query=args.negatives_per_query
        self.train_len=len(TRAINING_QUERIES.keys())
        # self.train_file_items = np.random.permutation(np.arange(0, self.train_len))
        # self.train_file_items = np.arange(0, self.train_len)
        print('Load Oxford Dataset')
        # self.data, self.label = []
        self.last = []
        print(self.train_len)   # 21711
    def __getitem__(self, item):
        # print('len(TRAINING_QUERIES[item][positives]: ',len(TRAINING_QUERIES[item]["positives"])," and item: ",item)
        if (len(TRAINING_QUERIES[item]["positives"]) < self.positives_per_query):
            if self.last==[]:
                print("wrong")
            else:
                print('right')
                return self.last[0], self.last[1], self.last[2], self.last[3]
        # no cached feature vectors
        if load_fast:
            q_tuples=get_query_tuple_fast(item, TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                TRAINING_QUERIES, hard_neg=[], other_neg=True)
        else:
            q_tuples = get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True)
        # 对点云进行增强，旋转或者加噪声
        # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[item],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
        # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[item],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))

        # 这里默认使用了quadruplet loss，所以必须找到other_neg
        if (q_tuples[3].shape[0] != self.num_points):
            print('----' + 'FAULTY other_neg' + '-----')
            if self.last==[]:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]

        queries = np.expand_dims(np.array(q_tuples[0], dtype=np.float32), axis=0)
        other_neg = np.expand_dims(np.array(q_tuples[3], dtype=np.float32), axis=0)
        positives = np.array(q_tuples[1], dtype=np.float32)
        negatives = np.array(q_tuples[2], dtype=np.float32)

        if (len(queries.shape) != 3):
            print('----' + 'FAULTY QUERY' + '-----')
            if self.last==[]:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]
        self.last = [queries, positives, negatives, other_neg]
        return queries.astype('float32'), positives.astype('float32'), negatives.astype('float32'), other_neg.astype('float32')

    def __len__(self):
        return self.train_len

# class Oxford_train_advance(Dataset):
#     def __init__(self, args):
#         self.num_points = args.num_points
#         self.positives_per_query = args.positives_per_query
#         self.negatives_per_query = args.negatives_per_query
#         self.train_len = len(TRAINING_QUERIES.keys())
#         # self.train_file_items = np.random.permutation(np.arange(0, self.train_len))
#         # self.train_file_items = np.arange(0, self.train_len)
#         print('Load Oxford Dataset')
#         # self.data, self.label = []
#         self.sampled_neg = 4000
#         self.hard_neg_num = args.hard_neg_per_query
#         if self.hard_neg_num > args.negatives_per_query:
#             print("self.hard_neg_num >  args.negatives_per_query")
#         self.last=[]
#     def __getitem__(self, item):
#         if (len(TRAINING_QUERIES[item]["positives"]) < self.positives_per_query):
#             # log_string("lack positive")
#             if self.last==[]:
#                 print("wrong")
#             else:
#                 return self.last[0], self.last[1], self.last[2], self.last[3]
#         if (len(HARD_NEGATIVES.keys()) == 0):
#             from time import time
#             start = time()
#             query = get_feature_representation(TRAINING_QUERIES[item]['query'], para.model)
#             # log_string("data: ",time()-start)
#             random.shuffle(TRAINING_QUERIES[item]['negatives'])
#             # log_string("data: ",time()-start)
#             negatives = TRAINING_QUERIES[item]['negatives'][0:self.sampled_neg]
#             # log_string("data: ",time()-start)
#             # 找到离当前query最近的neg KDtree比较耗时
#             hard_negs = get_random_hard_negatives(query, negatives, self.hard_neg_num)
#             # log_string("data: ",time()-start)
#             # log_string(hard_negs)
#             if load_fast:
#                 q_tuples=get_query_tuple_fast(item, TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
#                                 TRAINING_QUERIES, hard_neg=hard_negs, other_neg=True)
#             else:
#                 q_tuples=get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
#                                 TRAINING_QUERIES, hard_neg = hard_negs, other_neg=True)
#             # log_string("data: ",time()-start)
#         #     如果指定了一些HARD_NEGATIVES，實際沒有
#         else:
#             query = get_feature_representation(
#                 TRAINING_QUERIES[item]['query'], para.model)
#             random.shuffle(TRAINING_QUERIES[item]['negatives'])
#             negatives = TRAINING_QUERIES[item
#                         ]['negatives'][0:self.sampled_neg]
#             hard_negs = get_random_hard_negatives(
#                 query, negatives, self.hard_neg_num)
#             hard_negs = list(set().union(
#                 HARD_NEGATIVES[item], hard_negs))
#             # log_string('hard', hard_negs)
#             if load_fast:
#                 q_tuples=get_query_tuple_fast(item, TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
#                                 TRAINING_QUERIES, hard_neg=hard_negs, other_neg=True)
#             else:
#                 q_tuples=get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
#                                 TRAINING_QUERIES, hard_negs, other_neg=True)

#         # 这里默认使用了quadruplet loss，所以必须找到other_neg
#         if (q_tuples[3].shape[0] != self.num_points):
#             print('----' + 'FAULTY other_neg' + '-----')
#             if self.last==[]:
#                 print("wrong")
#             else:
#                 return self.last[0], self.last[1], self.last[2], self.last[3]

#         queries = np.expand_dims(np.array(q_tuples[0], dtype=np.float32), axis=0)
#         other_neg = np.expand_dims(np.array(q_tuples[3], dtype=np.float32), axis=0)
#         positives = np.array(q_tuples[1], dtype=np.float32)
#         negatives = np.array(q_tuples[2], dtype=np.float32)

#         if (len(queries.shape) != 3):
#             print('----' + 'FAULTY QUERY' + '-----')
#             if self.last==[]:
#                 print("wrong")
#             else:
#                 return self.last[0], self.last[1], self.last[2], self.last[3]
#         self.last = [queries, positives, negatives, other_neg]
#         return queries.astype('float32'), positives.astype('float32'), negatives.astype('float32'), other_neg.astype('float32')

#     def __len__(self):
#         return self.train_len


def update_vectors(args, model, tqdm_flag=True):
    global TRAINING_LATENT_VECTORS
    global TRAINING_QUERIES

    torch.cuda.empty_cache()

    if tqdm_flag:
        fun_tqdm = tqdm
    else:
        fun_tqdm = list

    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))

    batch_num = args.eval_batch_size * (1 + args.positives_per_query + args.negatives_per_query)
    # log_string("\n args: ",args.batch_num_queries,args.positives_per_query,args.negatives_per_query)
    q_output = []

    model.eval()

    for q_index in fun_tqdm(range(len(train_file_idxs) // batch_num)):
    # for q_index in tqdm(range(batch_num*2 // batch_num)):
        if load_fast:
            file_indices = np.arange(q_index * batch_num, (q_index + 1) * (batch_num))
            queries = TRAINING_POINT_CLOUD[file_indices]
        else:
            file_indices = train_file_idxs[q_index * batch_num:(q_index + 1) * (batch_num)]
            file_names = []
            for index in file_indices:
                file_names.append(TRAINING_QUERIES[index]["query"])
            queries = load_pc_files(file_names)

        feed_tensor = torch.from_numpy(queries).float()
        feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out = model(feed_tensor)
        out = out.detach().cpu().numpy()
        out = np.squeeze(out)
        q_output.append(out)

    q_output = np.array(q_output)
    if (len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    del feed_tensor

    # handle edge case
    for q_index in fun_tqdm(range((len(train_file_idxs) // batch_num * batch_num), len(TRAINING_QUERIES.keys()))):
        if load_fast:
            queries = TRAINING_POINT_CLOUD[train_file_idxs[q_index]]
            queries = np.expand_dims(np.expand_dims(queries, axis=0), axis=0)
        else:
            index = train_file_idxs[q_index]
            queries = load_pc_files([TRAINING_QUERIES[index]["query"]])

        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            queries_tensor = queries_tensor.to(device)
            output = model(queries_tensor)

        output = output.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output.reshape(-1,FEATURE_OUTPUT_DIM)))
        else:
            q_output = output
    del queries_tensor

    model.train()

    TRAINING_LATENT_VECTORS = q_output
    # log_string("Updated cached feature vectors")
    torch.cuda.empty_cache()

    if tqdm_flag:
        print("update all vectors.")
    else:
        print("update all vectors.", print_flag=False)