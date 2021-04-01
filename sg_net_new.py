import time
import torch
import random
import numpy as np
from tqdm import tqdm, trange
# from torch_geometric.nn import GCNConv
from layers_batch import AttentionModule, TenorNetworkModule
from utils import *
from tensorboardX import SummaryWriter
# from warmup_scheduler import GradualWarmupScheduler
import os
import dgcnn as dgcnn
import torch.nn as nn
from collections import OrderedDict
from sklearn import metrics


class SG(torch.nn.Module):      
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        print("init SimGNN")
        super(SG, self).__init__()
        self.args = args
        # cjf commented
        self.number_labels = 128
        # cjf
        self.setup_layers()


    # cjf what usage
    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args.tensor_neurons

    # cjf setup layers of node embedding
    def setup_layers(self):
        """
        Creating the layers.
        """
        print("setup layers of node embedding")
        self.calculate_bottleneck_features() # self.feature_count = self.args.tensor_neurons = 16
        # initialize AttentionModule & TenorNetworkModule
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        bias_bool = False # TODO
        self.dgcnn_s_conv1 = nn.Sequential(
            nn.Conv2d(3*2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        # by cjf 参数要修改
        self.dgcnn_f_conv1 = nn.Sequential(
            nn.Conv2d(self.number_labels * 2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_s_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1*2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        # by cjf 参数要修改
        self.dgcnn_f_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1 * 2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_s_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2*2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        # by cjf 参数要修改
        self.dgcnn_f_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2 * 2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv_end = nn.Sequential(nn.Conv1d(self.args.filters_3 * 2,
                                                      self.args.filters_3, kernel_size=1, bias=bias_bool),
                                            nn.BatchNorm1d(self.args.filters_3), nn.LeakyReLU(negative_slope=0.2))

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

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        features_1 = data["features_1"].cuda(self.args.gpu)
        features_2 = data["features_2"].cuda(self.args.gpu)

        # features B x (3+label_num) x node_num
        abstract_features_1 = self.dgcnn_conv_pass(features_1) # node_num x feature_size(filters-3)
        abstract_features_2 = self.dgcnn_conv_pass(features_2)  #BXNXF
        print("abstract feature: ", abstract_features_1.shape)
        pooled_features_1, attention_scores_1 = self.attention(abstract_features_1) # bxfx1
        pooled_features_2, attention_scores_2 = self.attention(abstract_features_2)
        # print("pooled_features_1: ", pooled_features_1)
        # print("pooled_features_2: ", pooled_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)  # scores: 16维
        # print("scores: ", scores.shape)
        scores = scores.permute(0,2,1) # bx1xf
        # print("scores: ", scores.shape)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        # print("scores: ", scores.shape)
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)  # score: 1维
        return score, attention_scores_1, attention_scores_2


class SGTrainer(object):
    """
    SG model trainer.
    """

    def __init__(self, args, train=True):
        """
        :param args: Arguments object.
        """
        self.args = args
        print('init SGTrainer')
        self.model_pth = self.args.model # 模型文件
        # eval: train=False; train: train=True
        # cjf commented
        # 将点云进行配对
        self.initial_label_enumeration(train)
        # cjf
        self.setup_model(train)
        self.writer = SummaryWriter(logdir=self.args.logdir) # 输出训练结果日志


    def setup_model(self,train=True):
        print('setup_model')
        self.model = SG(self.args)
        # test mode
        if (not train) and self.model_pth != "": # there is a pretrained model and not train this time
            print("loading model: ", self.model_pth)
            # load the module from file, remap to cuda:0
            state_dict = torch.load(self.model_pth, map_location='cuda:0')
            # create new dict that does not contain 'module'
            # 按照放入顺序排序的字典类型
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module'
                new_state_dict[name] = v
            # load params
            self.model.load_state_dict(new_state_dict)

        # continue_train mode
        if (self.args.continue_train) and self.model_pth != "": # continue train with the exist model

            state_dict = torch.load(self.model_pth, map_location='cuda:0')
            # create new dict that does not contain 'module'
            # 按照放入顺序排序的字典类型
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module'
                new_state_dict[name] = v
            # load params
            self.model.load_state_dict(new_state_dict)
        
        # 根据具体gpu硬件，将模型数据按照batch进行并行化，用于后续在gpu上的计算
        self.model = torch.nn.DataParallel(self.model, device_ids=[self.args.gpu])
        # 将模型送入cuda，默认为当前映射的显卡，也可以修改
        self.model.cuda(self.args.gpu)

    def initial_label_enumeration(self,train=True):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        if train:
            self.training_graphs = []
            self.testing_graphs = []
            self.evaling_graphs = []
            train_sequences = self.args.train_sequences
            eval_sequences = self.args.eval_sequences
            print("Train sequences: ", train_sequences)
            print("evaling sequences: ", eval_sequences)
            graph_pairs_dir = self.args.graph_pairs_dir
            for sq in train_sequences:
                # 加载预处理好的graphs pairs
                train_graphs=load_paires(os.path.join(self.args.pair_list_dir, sq+".txt"),graph_pairs_dir)
                self.training_graphs.extend(train_graphs)  # 用列表扩展已有列表
            for sq in eval_sequences:
                self.evaling_graphs=load_paires(os.path.join(self.args.pair_list_dir, sq+".txt"),graph_pairs_dir)
            self.testing_graphs = self.evaling_graphs
            assert len(self.evaling_graphs) != 0
            assert len(self.training_graphs) != 0
        # TODO 根据新特征的维度修改，即n个特征替换12个语义类
        # cjf commented
        # self.global_labels = [i for i in range(12)]
        # self.global_labels = {val: index for index, val in enumerate(self.global_labels)}
        # self.number_of_labels = len(self.global_labels)
        # self.keepnode = self.args.keep_node

        # print(self.global_labels)
        # print(self.number_of_labels)
        # cjf commented 

    def create_batches(self, split="train"):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        if split == "train":
            random.shuffle(self.training_graphs)
            batches = [self.training_graphs[graph:graph + self.args.batch_size] for graph in
                       range(0, len(self.training_graphs), self.args.batch_size)]
        else:
            random.shuffle(self.evaling_graphs)
            batches = [self.evaling_graphs[graph:graph + self.args.batch_size] for graph in
                       range(0, len(self.evaling_graphs), self.args.batch_size)]
        return batches

    # for better training
    def augment_data(self,batch_xyz_1):
        # batch_xyz_1 = flip_point_cloud(batch_xyz_1)
        batch_xyz_1 = rotate_point_cloud(batch_xyz_1)
        batch_xyz_1 = jitter_point_cloud(batch_xyz_1)
        batch_xyz_1 = random_scale_point_cloud(batch_xyz_1)
        batch_xyz_1 = rotate_perturbation_point_cloud(batch_xyz_1)
        batch_xyz_1 = shift_point_cloud(batch_xyz_1)
        return batch_xyz_1

    def pc_normalize(self, pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def transfer_to_torch(self, data, training=True):
        """
        Transferring the data to torch and creating a hash table with the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        # data_ori = data.copy()
        # print("data[edge1]: ", data["edges_1"])  # debug

        node_num_1 = len(data["nodes_1"])
        node_num_2 = len(data["nodes_2"])
        # cjf rskdd_nodes == node_num

        if node_num_1 > self.args.node_num:
            #  np.random.choice： 从node_num_1中随机抽取node_num个，不允许重复
            # by cjf
            # 对于RSKDD的输入点，应根据saliency的大小进行采样
            # by cjf
            sampled_index_1 = np.random.choice(node_num_1, self.args.node_num, replace=False)
            sampled_index_1.sort()
            data["nodes_1"] = np.array(data["nodes_1"])[sampled_index_1].tolist()  # 转成列表类型
            data["centers_1"] = np.array(data["centers_1"])[sampled_index_1]

        elif node_num_1 < self.args.node_num:
            #  nodes补-1， centers坐标补(0,0,0)
            data["nodes_1"] = np.concatenate(
                (np.array(data["nodes_1"]), -np.ones(self.args.node_num - node_num_1))).tolist()  # padding -1
            data["centers_1"] = np.concatenate(
                (np.array(data["centers_1"]), np.zeros((self.args.node_num - node_num_1,3))))  # padding 0

        if node_num_2 > self.args.node_num:
            sampled_index_2 = np.random.choice(node_num_2, self.args.node_num, replace=False)
            sampled_index_2.sort()
            data["nodes_2"] = np.array(data["nodes_2"])[sampled_index_2].tolist()
            data["centers_2"] = np.array(data["centers_2"])[sampled_index_2]  # node_num x 3
        elif node_num_2 < self.args.node_num:
            data["nodes_2"] = np.concatenate((np.array(data["nodes_2"]), -np.ones(self.args.node_num - node_num_2))).tolist()
            data["centers_2"] = np.concatenate(
                (np.array(data["centers_2"]), np.zeros((self.args.node_num - node_num_2, 3))))  # padding 0

        # create a dictionary
        new_data = dict()
        '''
         feature space
         one-hot encoder, number_of_labels is used here
        '''
        # cjf commented
        # features_1 = np.expand_dims(np.array(
        #     [np.zeros(self.number_of_labels).tolist() if node == -1 else [
        #         1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()]
        #      for node in data["nodes_1"]]), axis=0)
        # features_2 = np.expand_dims(np.array(
        #     [np.zeros(self.number_of_labels).tolist() if node == -1 else [
        #         1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()]
        #      for node in data["nodes_2"]]), axis=0)
        
        features_1 = np.expand_dims(np.array(data["nodes_1"]),axis=0)
        features_2 = np.expand_dims(np.array(data["nodes_2"]),axis=0)

        # cjf commented

        # 1xnode_numx3
        '''
         spatial space
        '''
        batch_xyz_1 = np.expand_dims(data["centers_1"], axis=0)
        batch_xyz_2 = np.expand_dims(data["centers_2"], axis=0)
        if training:
            # random flip data
            # 随机对某些centers的x,y坐标值取相反数
            if random.random() > 0.5:
                batch_xyz_1[:,:,0] = -batch_xyz_1[:,:,0]
                batch_xyz_2[:, :, 0] = -batch_xyz_2[:, :, 0]
            # 对点云做各种变换
            batch_xyz_1 = self.augment_data(batch_xyz_1)
            batch_xyz_2 = self.augment_data(batch_xyz_2)
        #  B x num_node x (3+num_label) -> B x (3+num_label) x num_node
        xyz_feature_1 = np.concatenate((batch_xyz_1, features_1), axis=2).transpose(0,2,1)
        xyz_feature_2 = np.concatenate((batch_xyz_2, features_2), axis=2).transpose(0,2,1)
        new_data["features_1"] = np.squeeze(xyz_feature_1)
        new_data["features_2"] = np.squeeze(xyz_feature_2)


        if data["distance"] <= self.args.p_thresh:  # TODO
            new_data["target"] = 1.0
        elif data["distance"] >= 20:
            new_data["target"] = 0.0
        else:
            new_data["target"] = -100.0
            print("distance error: ", data["distance"])
            exit(-1)
        return new_data

    def process_batch(self, batch, training=True):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()  # set gradients to zero before backpropragation
        losses = 0
        batch_target = []
        batch_feature_1 = []
        batch_feature_2 = []
        for graph_pair in batch:
            data = process_pair(graph_pair)  # process_pair() belongs to utils.py, param is a path, not a pair itself
            # 经过transfer_to_torch后，原始输入的坐标和node被扩维，用于cuda的并行化计算；
            # node进行了one-hot编码，是data['features_1/_2']的数据来源，为了拉开正负样本之间的距离，只保留距离小于阈值的和距离大于20m的样本 
            data = self.transfer_to_torch(data, training)
            batch_feature_1.append(data["features_1"])
            batch_feature_2.append(data["features_2"])
            batch_feature_1.append(data["features_2"])
            batch_feature_2.append(data["features_1"])
            target = data["target"]
            batch_target.append(target)
            batch_target.append(target)
        data = dict()
        data["features_1"] = torch.FloatTensor(np.array(batch_feature_1))
        data["features_2"] = torch.FloatTensor(np.array(batch_feature_2))
        data["target"] = torch.FloatTensor(np.array(batch_target))
        """

        standard training process:
            step 1: preprocess input data
            step 2: prediction = model(input)
            step 3: calculate loss
            step 4: compute gradients
            step 5: update params using gradients
            step 6: reset gradients to zero

        """
        prediction, _,_ = self.model(data)
        losses = torch.mean(torch.nn.functional.binary_cross_entropy(prediction, data["target"].cuda(self.args.gpu)))
        if training:
            losses.backward(retain_graph=True)  # compute gradients
            self.optimizer.step()   # update params
        loss = losses.item()
        pred_batch = prediction.cpu().detach().numpy().reshape(-1)
        gt_batch = data["target"].cpu().detach().numpy().reshape(-1)
        return loss, pred_batch, gt_batch

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        f1_max_his = 0
        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")  # 进度条
        for epoch in epochs:
            batches = self.create_batches()  # 根据数据总量和同时处理的batch_size确定batches的数量
            self.model.train()  
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                a = time.time()
                loss_score,_,_ = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum / main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
                self.writer.add_scalar('Train_sum', loss, int(epoch)*len(batches)*int(self.args.batch_size) + main_index)
                self.writer.add_scalar('Train loss', loss_score, int(epoch) * len(batches)*int(self.args.batch_size) + main_index)

            if epoch % 2 == 0:
                print("\nModel saving.\n")
                loss, f1_max = self.score("eval")
                self.writer.add_scalar("eval_loss", loss, int(epoch)*len(batches)*int(self.args.batch_size))
                self.writer.add_scalar("f1_max_score", f1_max, int(epoch) * len(batches) * int(self.args.batch_size))
                dict_name = self.args.logdir + "/" + str(epoch)+'.pth'
                torch.save(self.model.state_dict(), dict_name)
                if f1_max_his <= f1_max:
                    f1_max_his = f1_max
                    dict_name = self.args.logdir + "/" + str(epoch)+"_best" + '.pth'
                    torch.save(self.model.state_dict(), dict_name)
                    print("\n best model saved ", dict_name)
                print("------------------------------")

    def score(self, split = 'test'):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []

        if split == "test":
            splits = self.testing_graphs
        elif split == "eval":
            splits = self.evaling_graphs
        else:
            print("Check split: ", split)
            splits = []
            exit(-1)

        losses = 0
        pred_db = []
        gt_db = []
        batches = self.create_batches(split="eval")
        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Eval Batches"):
            loss_score,pred_b,gt_b = self.process_batch(batch, False)
            losses += loss_score
            pred_db.extend(pred_b)
            gt_db.extend(gt_b)

        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)
        # calc F1-score
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        F1_max_score = np.max(F1_score)
        print("\nModel " + split + " F1_max_score: " + str(F1_max_score) + ".")
        model_loss = losses / len(batches)
        print("\nModel " + split + " loss: " + str(model_loss) + ".")
        return model_loss, F1_max_score

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n - norm_ged_mean) ** 2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " + str(round(base_error, 5)) + ".")
        print("\nModel test error: " + str(round(model_error, 5)) + ".")
    
    # useless
    def eval_pair(self, pair_file):
        data =   (pair_file)
        data = self.transfer_to_torch(data, False)
        target = data["target"]

        batch_target = []
        batch_feature_1 = []
        batch_feature_2 = []
        batch_feature_1.append(data["features_1"])
        batch_feature_2.append(data["features_2"])
        batch_target.append(target)

        data_torch = dict()
        data_torch["features_1"] = torch.FloatTensor(np.array(batch_feature_1))
        data_torch["features_2"] = torch.FloatTensor(np.array(batch_feature_2))
        data_torch["target"] = torch.FloatTensor(np.array(batch_target))
        self.model.eval()
        result_1, result_2,result_3 = self.model(data_torch)
        prediction = result_1.cpu().detach().numpy().reshape(-1)
        att_weights_1 = result_2.cpu().detach().numpy().reshape(-1)
        att_weights_2 = result_3.cpu().detach().numpy().reshape(-1)

        # print("prediction shape: ", prediction.shape)
        return prediction, att_weights_1, att_weights_2

    #  useless, 和下面重复了
    # def eval_batch_pair(self, batch):
        self.model.eval()
        batch_target = []
        batch_feature_1 = []
        batch_feature_2 = []
        for graph_pair in batch:
            # commented by cjf
            # data = process_pair(graph_pair)
            # created by cjf
            data = process_rskdd_pair(graph_pair)
            
            data = self.transfer_to_torch(data, False)
            batch_feature_1.append(data["features_1"])
            batch_feature_2.append(data["features_2"])
            target = data["target"]
            batch_target.append(target)
        data = dict()
        data["features_1"] = torch.FloatTensor(np.array(batch_feature_1))
        data["features_2"] = torch.FloatTensor(np.array(batch_feature_2))
        data["target"] = torch.FloatTensor(np.array(batch_target))
        prediction, _, _ = self.model(data)
        prediction = prediction.cpu().detach().numpy().reshape(-1)
        gt = np.array(batch_target).reshape(-1)
        return prediction, gt

    def eval_batch_pair_data(self, batch):
        self.model.eval()

        batch_target = []
        batch_feature_1 = []
        batch_feature_2 = []
        for graph_pair in batch:
            data = self.transfer_to_torch(graph_pair, False)
            batch_feature_1.append(data["features_1"])
            batch_feature_2.append(data["features_2"])
            target = data["target"]
            batch_target.append(target)
        data = dict()
        data["features_1"] = torch.FloatTensor(np.array(batch_feature_1))
        data["features_2"] = torch.FloatTensor(np.array(batch_feature_2))
        data["target"] = torch.FloatTensor(np.array(batch_target))
        forward_t = time.time()
        prediction, _, _ = self.model(data)
        print("forward time: ", time.time() - forward_t)
        prediction = prediction.cpu().detach().numpy().reshape(-1)
        gt = np.array(batch_target).reshape(-1)
        return prediction, gt

    def eval_batch_pair(self, batch):
        self.model.eval() # if not add this, params will be changed even not train
        batch_target = []
        batch_feature_1 = []
        batch_feature_2 = []
        #  graph_pair: ['***.json', 'xxx.json']
        for graph_pair in batch:
            data = process_pair(graph_pair)
            data = self.transfer_to_torch(data, False)
            batch_feature_1.append(data["features_1"])
            batch_feature_2.append(data["features_2"])
            target = data["target"]
            batch_target.append(target)
        data = dict()
        data["features_1"] = torch.FloatTensor(np.array(batch_feature_1))
        data["features_2"] = torch.FloatTensor(np.array(batch_feature_2))
        data["target"] = torch.FloatTensor(np.array(batch_target))
        print('evaling batch pair')
        # forward_t = time.time()
        prediction, _, _ = self.model(data) # model --> SG.forward
        # print("forward time: ", time.time() - forward_t)
        print('eval completed')
        prediction = prediction.cpu().detach().numpy().reshape(-1)
        gt = np.array(batch_target).reshape(-1)
        return prediction, gt

    # useless
    def write_soft_label(self, data_dir):
        eval_graphs = []
        listDir(data_dir, eval_graphs)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        thresh = 0.5
        for i in range(len(eval_graphs)):
            pair_file = eval_graphs[i]
            data = json.load(open(pair_file))
            pred, _, _ = self.eval_pair(data)

            if pred <= thresh:
                if data["distance"] <=10:
                    TN += 1
                else:
                    FN += 1
                data["distance"] = 100
            else:
                if data["distance"] <=10:
                    TP += 1
                else:
                    FP += 1
                data["distance"] = 0

            file_name = os.path.join("/media/work/data/kitti/odometry/semantic-kitti/DGCNN_graph_pairs_3_20/pred_label/05",
                                     pair_file.split('/')[-1])
            print("write pred label: ", file_name)
            with open(file_name, "w", encoding="utf-8") as file:
                json.dump(data, file)

        precesion = TP / (TP + FP)
        recall = TP / (TP + FN)
        print("thresh: ", thresh)
        print("precision: ", precesion)
        print("recall:", recall)
