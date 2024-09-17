""" Componets of the model
"""
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
import math


class GraphLearningLayer(nn.Module):
    def __init__(self, in_dim: int, learning_dim: int, gamma: float, eta: float):
        super().__init__()
        self.projection = nn.Linear(in_dim, learning_dim, bias=False)
        self.learn_w = nn.Parameter(torch.empty(learning_dim))
        self.gamma = gamma
        self.eta = eta
        self.inint_parameters()

    def inint_parameters(self):
        nn.init.uniform_(self.learn_w, a=0, b=1)

    def forward(self, x: Tensor, adj: Tensor, box_num: Tensor = None):

        N, D = x.shape
        x_hat = self.projection(x)
        _, learning_dim = x_hat.shape

        x_i = x_hat.unsqueeze(1).expand(N, N, learning_dim)
        x_j = x_hat.unsqueeze(0).expand(N, N, learning_dim)
        m1 = x_i.norm(p=2, dim=-1, keepdim=True)
        m2 = x_j.norm(p=2, dim=-1, keepdim=True)
        distance = 1 - torch.mul(x_i, x_j) / (m1 * m2)
        print(distance.shape)

        distance = torch.einsum('ijd, d->ij', distance, self.learn_w)

        out = F.leaky_relu(distance)

        max_out_v, _ = out.max(dim=-1, keepdim=True)
        out = out - max_out_v

        soft_adj = torch.exp(out)
        soft_adj = adj * soft_adj

        sum_out = soft_adj.sum(dim=-1, keepdim=True)
        soft_adj = soft_adj / sum_out + 1e-10

        gl_loss = None
        if self.training:
            gl_loss = self._graph_learning_loss(x_hat, soft_adj, box_num)

        return soft_adj, gl_loss

    def _graph_learning_loss(self, x_hat: Tensor, adj: Tensor):

        N, D = x_hat.shape

        x_i = x_hat.unsqueeze(1).expand(N, N, D)
        x_j = x_hat.unsqueeze(0).expand(N, N, D)
        m1 = x_i.norm(p=2, dim=-1, keepdim=True)
        m2 = x_j.norm(p=2, dim=-1, keepdim=True)

        dist_loss = adj + self.eta * torch.norm(1 - torch.mul(x_i, x_j) / (m1 * m2), dim=2)
        dist_loss = torch.exp(dist_loss)

        dist_loss = torch.sum(dist_loss, dim=(0, 1)) / (N * N)

        f_norm = torch.norm(adj, dim=(0, 1)) # remove square operation duo to it can cause nan loss.

        gl_loss = dist_loss + self.gamma * f_norm
        return gl_loss


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming_uniform
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.layer1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.layer2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.layer3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])

        self.dropout = dropout

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer3(x, adj)
        x = F.leaky_relu(x, 0.25)

        return x

class Graphencoder(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Graphencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, num_view, gcns_output, adj):
        AH = []
        for i in range(num_view):
            ah = torch.mm(adj[i], gcns_output[i])
            AH.append(ah)
        input = torch.stack(AH, dim=0)
        output = self.encoder(input)

        return output


class Graphdecoder(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Graphdecoder, self).__init__()
        # self.decoder = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))
        self.decoder = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, num_view, gcns_output, z, adj):
        gcn_hat = []
        for i in range(num_view):
            w = self.decoder(z)
            gcn_hat.append(torch.mm(adj[i], w))
        gcn = torch.stack(gcns_output, dim=0)
        gcn_hat = torch.stack(gcn_hat, dim=0)
        square_loss = torch.mean(torch.square(torch.sub(gcn, gcn_hat)))

        return square_loss


class query_key_value(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(query_key_value, self).__init__()
        self.liner_q = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.liner_k = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.graphencoder = Graphencoder(in_dim, out_dim)

    def forward(self, num_view, gcns_output, adj):
        gcns_output_stack = torch.stack(gcns_output, dim=0)
        Q_view = torch.mean(gcns_output_stack, dim=0)
        Q_view = self.liner_q(Q_view)
        K_view = self.liner_k(gcns_output_stack)
        V_view = self.graphencoder(num_view, gcns_output, adj)

        return Q_view, K_view, V_view

class Attention(nn.Module):

    def __init__(self, attention_dropout=0.0):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, q, k, v, scale):
        attention = torch.mul(q, k)
        attention = attention / scale

        attention = self.softmax(attention)

        attention = self.dropout(attention)

        context = torch.mul(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.q_k_v = query_key_value(model_dim, model_dim)

        self.dot_product_attention = Attention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, num_view, gcns_output, adj):

        query, key, value = self.q_k_v(num_view, gcns_output, adj)
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(1)

        residual = query


        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # scaled dot product attention
        scale = torch.sqrt(torch.tensor(dim_per_head))
        context, attention = self.dot_product_attention(query, key, value, scale)
        # concat heads
        context = torch.sum(context.view(-1, context.shape[1], dim_per_head * num_heads), dim=1)
        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))

    def forward(self, x):
        x = self.clf(x)
        return x


def init_model_dict(num_view, num_class, dim_list, dim_he_list, gcn_dropout=0.25, gama=0.3, eta=0.7, learning_dim=100):
    model_dim = dim_he_list[-1]
    # model_dim = dim_he_list[-1] * 3
    model_dict = {}
    for i in range(num_view):
        input_dim = dim_list[i]
        model_dict["GL{:}".format(i + 1)] = GraphLearningLayer(input_dim, learning_dim, gama, eta)
        model_dict["E{:}".format(i+1)] = GCN_E(input_dim, dim_he_list, gcn_dropout)
    if num_view >= 2:
        model_dict["H"] = MultiHeadAttention(model_dim, num_heads=1, dropout=0.0)
        model_dict["C"] = Classifier(model_dim, num_class)
        model_dict["D"] = Graphdecoder(model_dim, model_dim)
    else:
        model_dict["C"] = Classifier(model_dim, num_class)
    return model_dict


def init_optim(num_view, model_dict, lr_c=1e-5, lr_e=1e-3):
    optim_dict = {}
    gcn_e_parameters = []
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)] = torch.optim.Adam(list(model_dict["GL{:}".format(i + 1)].parameters()) +
                                                             list(model_dict["E{:}".format(i + 1)].parameters()),
                                                             lr=lr_e)

    if num_view >= 2:
        optim_dict["H"] = torch.optim.Adam(list(model_dict["H"].parameters()) +
                                            list(model_dict["C"].parameters()) +
                                            list(model_dict["D"].parameters()), lr=lr_c)

    return optim_dict

def init_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2500)
    return scheduler