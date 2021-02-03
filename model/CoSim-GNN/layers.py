from config import FLAGS
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv , ChebConv
import numpy as np
from utils_our import get_branch_names, get_flag
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv

class NodeEmbedding(nn.Module):
    def __init__(self, type, in_dim, out_dim, act, bn):
        super(NodeEmbedding, self).__init__()
        self.type = type
        self.out_dim = out_dim
        if type == 'gcn':
            self.conv = GCNConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        elif type == 'gin':
            mlps = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                create_act(act, out_dim),
                nn.Linear(out_dim, out_dim))
            self.conv = GINConv(mlps)
            self.act = create_act(act, out_dim)
        elif type == 'gat':
            self.conv = GATConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        elif type == 'dgcnn':
            nnl = nn.Sequential(nn.Linear(2 * in_dim, out_dim), nn.ReLU(),
                                nn.Linear(out_dim, out_dim))
            self.conv = EdgeConv(nnl, aggr='max')
            self.act = create_act(act, out_dim)
        elif type == 'chebconv':
            self.conv = ChebConv(in_dim,out_dim,10)
            self.act = create_act(act, out_dim)
        else:
            raise ValueError(
                'Unknown node embedding layer type {}'.format(type))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(out_dim)

    def forward(self, ins, batch_data, model):
        x = ins
        edge_index = batch_data.merge_data['merge'].edge_index

        x = self.conv(x, edge_index)
        x = self.act(x)
        if self.bn:
            x = self.bn(x)
        model.store_layer_output(self, x)

        return x

class NodeEmbeddingCombinator(nn.Module):
    def __init__(self, dim_size, from_layers, layers, num_node_feat, style):
        super(NodeEmbeddingCombinator, self).__init__()
        self.from_layers = from_layers
        self.style = style
        if style == 'concat':
            dims_list = []
            for i in self.from_layers:
                if i == 0:
                    dims_list.append(num_node_feat)
                elif i >= 1:
                    dims_list.append(layers[i - 1].out_dim)
                else:
                    raise ValueError('Wrong from_layer id {}'.format(i))
            self.out_dim = np.sum(dims_list)
        elif style == 'mlp':
            MLPs = []
            weights = []
            for i in self.from_layers:
                if i == 0:
                    MLPs.append(nn.Linear(num_node_feat, dim_size))
                    weights.append(nn.Parameter(torch.Tensor(1).fill_(1.0)).to(FLAGS.device))
                elif i >= 1:
                    MLPs.append(nn.Linear(layers[i - 1].out_dim, dim_size))
                    weights.append(nn.Parameter(torch.Tensor(1).fill_(1.0)).to(FLAGS.device))
                else:
                    raise ValueError('Wrong from_layer id {}'.format(i))
            self.MLPs = nn.ModuleList(MLPs)
            self.weights = weights
            self.out_dim = dim_size
        else:
            raise ValueError(
                'Unknown style {}'.format(style))

    def forward(self, ins, batch_data, model):
        if self.style == 'concat':
            embs_list = []
            for i in self.from_layers:
                if i == 0:
                    e = batch_data.merge_data['merge'].x
                elif i >= 1:
                    e = model.get_layer_output(model.layers[i - 1])
                else:
                    raise ValueError('Wrong from_layer id {}'.format(i))
                embs_list.append(e)
            x = torch.cat(embs_list, dim=1)
        elif self.style == 'mlp':
            assert self.from_layers[0] == 0
            for j, i in enumerate(self.from_layers):
                if i == 0:
                    e = batch_data.merge_data['merge'].x.to(FLAGS.device)
                    out = self.weights[j] * (self.MLPs[j])(e).to(FLAGS.device)
                elif i >= 1:
                    e = model.get_layer_output(model.layers[i - 1]).to(
                        FLAGS.device)  # 1-based to 0-based
                    out += self.weights[j] * (self.MLPs[j])(e)
                else:
                    raise ValueError('Wrong from_layer id {}'.format(i))
            x = out
        else:
            raise NotImplementedError()

        return x

class NodeEmbeddingInteraction(nn.Module):
    def __init__(self, type, in_dim):
        super(NodeEmbeddingInteraction, self).__init__()
        self.type = type
        self.in_dim = in_dim
        if type == 'dot':
            pass
        else:
            raise ValueError('Unknown node embedding interaction '
                             'layer type {}'.format(type))

    def forward(self, ins, batch_data, model, n_outputs=FLAGS.n_outputs):
        pair_list = batch_data.split_into_pair_list(ins, 'x')
        x1x2t_li = []
        for pair in pair_list:
            x1_all, x2_all = pair.g1.x, pair.g2.x
            _, D = x1_all.shape
            D_split = D // n_outputs
            mnes = []
            for i in range(n_outputs):
                D_pre, D_post = i * D_split, (i + 1) * D_split
                x1 = x1_all[:, D_pre:D_post] if i < (n_outputs - 1) else x1_all[:, D_pre:]
                x2 = x2_all[:, D_pre:D_post] if i < (n_outputs - 1) else x2_all[:, D_pre:]
                if self.type == 'dot':
                    assert x1.shape[1] == x2.shape[1]
                    mne = torch.mm(x1, x2.t())
                else:
                    assert False
                mnes.append(mne)
                x1x2t_li.append(mne)
            pair.assign_y_pred_list(mnes, format='torch_{}'.format(FLAGS.device))
            pair.z_pred = (None, None)
        return x1x2t_li

class MatchingMatrixComp(nn.Module):
    def __init__(self, opt):
        super(MatchingMatrixComp, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.opt = opt

    def forward(self, ins, batch_data, model):
        pair_list = batch_data.pair_list
        for i, pair in enumerate(pair_list):
            g1, g2 = pair.g1.get_nxgraph(), pair.g2.get_nxgraph()
            N, M = g1.number_of_nodes(), g2.number_of_nodes()

            # construct node feature mask
            mask = torch.matmul(torch.tensor(g1.init_x), torch.tensor(g2.init_x.T)).type(
                torch.FloatTensor).to(FLAGS.device)


            ### compute Y and Z
            # for-loop implementation
            y_preds = []
            z_preds_g1 = []
            z_preds_g2 = []
            Xs = pair.get_y_pred_list_mat_view(
                format='torch_{}'.format(FLAGS.device))  # a list of tensors

            if 'sinkhorn_softmax' in self.opt:
                eps = 1e-6
                _, _, iters = self.opt.split('_')
                iters = int(iters)
                assert iters >= 1
                for X in Xs:
                    for iter in range(iters):
                        X = F.softmax(X, dim=1)
                        X = F.softmax(X, dim=0)
                    y_preds.append(X)
            elif self.opt == 'sigmoid':
                for X in Xs:
                    y_preds.append(self.sigmoid(X))
            elif self.opt == 'matching_matrix':
                for X in Xs:
                    # convert into probabilities
                    X = X[:N, :M]
                    z_pred_g1 = self.sigmoid(torch.sum(X, dim=1) / M)
                    z_pred_g2 = self.sigmoid(torch.sum(X, dim=0) / N)
                    if FLAGS.no_probability:
                        y_pred = mask * X
                    else:
                        y_pred_g1 = torch.t(self.softmax(torch.t(X)) * z_pred_g1)
                        y_pred_g2 = self.softmax(X) * z_pred_g2
                        y_pred = (y_pred_g1 + y_pred_g2) / 2
                        y_pred = mask * y_pred
                    y_preds.append(y_pred)
                    z_preds_g1.append(z_pred_g1)
                    z_preds_g2.append(z_pred_g2)
                pair.z_pred = (z_preds_g1, z_preds_g2)  # IMPORTANT: used by OurLossFunction
            else:
                assert False

            pair.assign_y_pred_list(  # used by evaluation
                [y_pred for y_pred in y_preds],
                format='torch_{}'.format(FLAGS.device))  # multiple predictions

class Sequence(nn.Module):
    def __init__(self, type, in_dim):
        super(Sequence, self).__init__()
        self.in_dim = in_dim
        self.rnn_in_dim = 2 * in_dim
        if type == 'lstm':
            self.rnn = nn.LSTM(self.rnn_in_dim, 1)
        elif type == 'gru':
            self.rnn = nn.GRU(self.rnn_in_dim, 1)
        else:
            raise ValueError('Unknown type {}'.format(type))

    def forward(self, ins, batch_data, model):
        assert type(get_prev_layer(self, model)) is NodeEmbedding
        pair_list = batch_data.split_into_pair_list(ins, 'x')
        y_pred_mat_list = []
        for pair in pair_list:
            x1, x2 = pair.g1.x, pair.g2.x
            assert x1.shape[1] == x2.shape[1]
            m, n = x1.shape[0], x2.shape[0]
            x1x2_seq = torch.zeros(m * n, self.rnn_in_dim, device=FLAGS.device)
            row = 0
            for i, ei in enumerate(x1):
                for j, ej in enumerate(x2):
                    ecat = torch.cat((ei, ej))
                    assert len(ecat) == self.rnn_in_dim
                    assert row == i * n + j, '{} {} {} {}'.format(row, i, n, j)
                    x1x2_seq[row] = ecat
                    row += 1
            x1x2_seq = x1x2_seq.view(len(x1x2_seq), 1, -1)
            out, hidden = self.rnn(x1x2_seq)
            assert out.shape == (m * n, 1, 1)
            y_pred_mat = out.view(m, n)
            pair.assign_y_pred_list(
                [y_pred_mat])
            y_pred_mat_list.append(y_pred_mat)
        return y_pred_mat_list

class Fancy(nn.Module):
    def __init__(self, in_dim):
        super(Fancy, self).__init__()
        self.in_dim = in_dim

    def forward(self, ins, batch_data, model):
        assert type(get_prev_layer(self, model)) is NodeEmbedding
        pair_list = batch_data.split_into_pair_list(ins, 'x')
        loss = 0.0
        for pair in pair_list:
            pass
        return loss

class Loss(nn.Module):
    def __init__(self, type):
        super(Loss, self).__init__()
        self.type = type
        if type == 'BCELoss':
            self.loss = nn.BCELoss()
        elif type == 'BCEWithLogitsLoss':  # contains a sigmoid
            self.loss = nn.BCEWithLogitsLoss()  # TODO: softmax?
        else:
            raise ValueError('Unknown loss layer type {}'.format(type))

    def forward(self, ins, batch_data, _):
        loss = 0
        pair_list = batch_data.pair_list
        for i, pair in enumerate(pair_list):
            print("------------------------")
            g1, g2 = pair.g1.ge7t_nxgraph(), pair.g2.get_nxgraph()
            N, M = g1.number_of_nodes(), g2.number_of_nodes()

            mask = torch.matmul(torch.tensor(g1.init_x), torch.tensor(g2.init_x.T)).type(
                torch.FloatTensor).to(FLAGS.device)

            y_preds = pair.get_y_pred_list_mat_view(format='torch_{}'.format(FLAGS.device))

            y_true = torch.zeros((N, M), device=FLAGS.device)

            y_true_dict_list = pair.get_y_true_list_dict_view()
            
            assert len(y_true_dict_list) >= 1
            y_true_dict = y_true_dict_list[0]
            for nid1 in y_true_dict.keys():
                nid2 = y_true_dict[nid1]
                y_true[nid1, nid2] = 1

            cand_losses = []
            for j in range(len(y_preds)):
                y_pred = mask * y_preds[j]
                cand_loss = self.loss(y_pred, y_true)
                cand_losses.append(cand_loss)

            loss += min(cand_losses)
        # Normalize by batch_size
        loss /= len(pair_list)
        return loss

def create_act(act, num_parameters=None):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'prelu':
        # print("?",num_parameters)
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError('Unknown activation function {}'.format(act))

def get_prev_layer(this_layer, model):
    for i, layer in enumerate(model.layers):
        j = i + 1
        if j < len(model.layers) and this_layer == model.layers[j]:
            return layer
    bnames = get_branch_names()
    if not bnames:
        return None
    for bname in bnames:
        blayers = getattr(model, bname)
        if this_layer == blayers[0]:
            return model.layers[get_flag('{}_start'.format(bname)) - 1]
        for i, layer in enumerate(blayers):
            j = i + 1
            if j < len(blayers) and this_layer == blayers[j]:
                return layer