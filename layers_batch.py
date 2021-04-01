#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


#embedding graphs and calculate the similarity score
class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3)) 
        
    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)   # 均匀分布

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector. 
        """
        # embedding: 论文中的u, shape: BxNx32
        batch_size = embedding.shape[0]
        # graph上每个node都乘上一个权重，然后再去平均值，成为这个graph的全局描述
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=1) # Bx32
        transformed_global = torch.tanh(global_context) # 论文中的c
        sigmoid_scores = torch.sigmoid(torch.matmul(embedding,transformed_global.view(batch_size,-1, 1)))   #weights      BxNx1
        representation = torch.matmul(embedding.permute(0,2,1),sigmoid_scores)    #论文中的e Bx32x1
        return representation, sigmoid_scores

class TenorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self,args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3, self.args.tensor_neurons))   # 32x32x16
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2*self.args.filters_3))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):    # input shape: Bx32x1
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.   
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = embedding_1.shape[0]
                                # Bx1x32           matmul             32x512           -------> Bx1x512        --------> Bx32x16
        scoring = torch.matmul(embedding_1.permute(0,2,1), self.weight_matrix.view(self.args.filters_3,-1)).view(batch_size, self.args.filters_3, self.args.tensor_neurons)
        scoring = torch.matmul(scoring.permute(0,2,1), embedding_2) # Bx16x32   matmul   Bx32x1    -------------> Bx16x1
        combined_representation = torch.cat((embedding_1, embedding_2), dim=1)  # Bx(32+32)x1
        block_scoring = torch.matmul(self.weight_matrix_block, combined_representation) # 16x64  matmul  Bx64x1    --------------->  Bx16x1
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores
