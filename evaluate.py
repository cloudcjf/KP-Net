import argparse
import math
import numpy as np
import socket
import importlib
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from model.models import KPNet
from oxfordloader import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from tensorboardX import SummaryWriter


DATASET_FOLDER = '/mnt/Airdrop/benchmark_datasets/'
# 将oxford替换为redidential/university/business,测试另外三个数据集
EVAL_DATABASE_FILE = 'generating_queries/oxford_evaluation_database.pickle'
EVAL_QUERY_FILE = 'generating_queries/oxford_evaluation_query.pickle'
RESULTS_FOLDER = "results/"
OUTPUT_FILE = "results/results.txt"

cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(args):
    model = KPNet(args)
    model = model.to(device)

    filename = args.ckpt_dir
    print("Loading model from: ", filename)
    checkpoint = torch.load(filename)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)
    model = nn.DataParallel(model)

    print(evaluate_model(model,args))


def evaluate_model(model,args):
    DATABASE_SETS = get_sets_dict(EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(EVAL_QUERY_FILE)

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    for i in range(len(DATABASE_SETS)):
        DATABASE_VECTORS.append(get_latent_vectors(model, args, DATABASE_SETS[i]))

    for j in range(len(QUERY_SETS)):
        QUERY_VECTORS.append(get_latent_vectors(model, args, QUERY_SETS[j]))

    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if (m == n):
                continue
            pair_recall, pair_similarity, pair_opr = get_recall(
                m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    print()
    ave_recall = recall / count
    # print(ave_recall)

    # print(similarity)
    average_similarity = np.mean(similarity)
    # print(average_similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)
    # print(ave_one_percent_recall)

    with open(OUTPUT_FILE, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(str(ave_recall))
        output.write("\n\n")
        output.write("Average Similarity:\n")
        output.write(str(average_similarity))
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))

    return ave_one_percent_recall


def get_latent_vectors(model, args, dict_to_process):

    model.eval()
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = args.eval_batch_size * \
        (1 + args.eval_positives_per_query + args.eval_negatives_per_query)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        #out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1 = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    # print(q_output.shape)
    return q_output


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    # print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        if(len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]),k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--positives_per_query', type=int, default=2,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--negatives_per_query', type=int, default=18,
                        help='Number of definite negatives in each training tuple [default: 18]')
    parser.add_argument('--hard_neg_per_query', type=int, default=1,
                        help='Number of definite negatives in each training tuple [default: 10]')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='test Batch Size during training [default: 1]')
    parser.add_argument('--eval_positives_per_query', type=int, default=2,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--eval_negatives_per_query', type=int, default=18,
                        help='Number of definite negatives in each training tuple [default: 18]')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num_points [default: 4096]')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt', help='path to save model')

    args = parser.parse_args()

    evaluate(args)
