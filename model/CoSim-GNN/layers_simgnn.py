from config import FLAGS
from layers import create_act, NodeEmbedding, get_prev_layer
from utils_our import debug_tensor, pad_extra_rows
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import Timer
import sys
from torch_geometric.data import Data as PyGSingleGraphData
from node_feat import encode_node_features

class ANPM(nn.Module):
    """ Attention_NTN_Padding_MNE layer. """

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight,
                 feature_map_dim, bias, ntn_inneract, apply_u,
                 mne_inneract, mne_method, branch_style,
                 reduce_factor, criterion):
        super(ANPM, self).__init__()

        if 'an' in branch_style:
            self.att_layer = Attention(input_dim=input_dim,
                                       att_times=att_times,
                                       att_num=att_num,
                                       att_style=att_style,
                                       att_weight=att_weight)

            self.ntn_layer = NTN(
                input_dim=input_dim * att_num,
                feature_map_dim=feature_map_dim,
                inneract=ntn_inneract,
                apply_u=apply_u,
                bias=bias)
        else:
            raise ValueError('Must have the Attention-NTN branch')

        self.num_bins = 0
        if 'pm' in branch_style:
            self.mne_layer = MNE(
                input_dim=input_dim,
                inneract=mne_inneract)

            if 'hist_' in mne_method:
                self.num_bins = self._get_mne_output_dim(mne_method)
            else:
                assert False

        self.mne_method = mne_method
        self.branch = branch_style

        proj_layers = []
        D = feature_map_dim
        while D > 1:
            next_D = D // reduce_factor
            if next_D < 1:
                next_D = 1
            linear_layer = nn.Linear(D, next_D, bias=False)
            nn.init.xavier_normal_(linear_layer.weight)
            proj_layers.append(linear_layer)
            proj_layers.append(nn.Sigmoid())
            D = next_D
        self.proj_layers = nn.ModuleList(proj_layers)

        if criterion == 'MSELoss':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError()

    def forward(self, ins, batch_data, model):
        assert type(get_prev_layer(self, model)) is NodeEmbedding
        pair_list = batch_data.split_into_pair_list(ins, 'x')

        pairwise_embeddings = []

        true = torch.zeros(len(pair_list), 1, device=FLAGS.device)
        for i, pair in enumerate(pair_list):
            x1, x2 = pair.g1.x, pair.g2.x
            x = self._call_one_pair([x1, x2])
            for proj_layer in self.proj_layers:
                x = proj_layer(x)

            pairwise_embeddings.append(x)
            pair.assign_ds_pred(x)
            true[i] = pair.get_ds_true(
                FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred, FLAGS.ds_kernel)

        pairwise_embeddings = (torch.cat(pairwise_embeddings, 0))
        pairwise_scores = pairwise_embeddings
        if FLAGS.dos_true == 'sim':
            true_score = torch.tensor(true).long()
            neg = -1 * torch.ones(true_score.size()).long()
            true_score = torch.where(true_score>0,true_score,neg).float()
            true_score.requires_grad = True
            loss = torch.sum(torch.relu(0.01+true_score*(0.5-pairwise_scores)))
            pos_index = torch.where(pairwise_scores > 0.51)
            neg_index = torch.where(pairwise_scores < 0.49)
            classification = torch.zeros(true_score.size()).long()
            classification[neg_index] = -1
            classification[pos_index] = 1
            auc = classification * true_score.long() - 1
            index = auc.nonzero()
            batch_data.auc_classification = true_score.size()[0] - index.size()[0]
            batch_data.sample_num = true_score.size()[0]
        else:
            loss = self.criterion(pairwise_scores, true)
        return loss

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]
        assert (x_1.shape[1] == x_2.shape[1])
        sim_scores, mne_features = self._get_ntn_scores_mne_features(x_1, x_2)

        # Merge.
        if self.branch == 'an':
            output = sim_scores
        elif self.branch == 'pm':
            output = mne_features
        elif self.branch == 'anpm':
            to_concat = (sim_scores, mne_features)
            output = torch.cat(to_concat, dim=1)
            # output = tf.nn.l2_normalize(output)
        else:
            raise RuntimeError('Unknown branching style {}'.format(self.branch))

        return output

    def _get_ntn_scores_mne_features(self, x_1, x_2):
        sim_scores, mne_features = None, None
        t = Timer()
        # Branch 1: Attention + NTN.
        if "an" in self.branch:
            # print("z1",x_1)
            x_1_gemb = (self.att_layer(x_1))
            x_2_gemb = (self.att_layer(x_2))
            # print("x_1_gemb", x_1_gemb)
            # print("x_2_gemb", x_2_gemb)
            self.embeddings = [x_1_gemb, x_2_gemb]
            self.att = self.att_layer.att
            sim_scores = self.ntn_layer([x_1_gemb, x_2_gemb])
            # sim_scores = torch.cosine_similarity(x_1_gemb, x_2_gemb)
            # print("sim_score",sim_scores)
            # print("an layer",t.time_msec_and_clear())
        # Branch 2: Padding + MNE.
        if 'pm' in self.branch:
            t = Timer()
            mne_features = self._get_mne_features(x_1, x_2)
            # print("mne",mne_features)
            # print("pm layer",t.time_msec_and_clear())
        return sim_scores, mne_features

    def _get_mne_features(self, x_1, x_2):
        # print("in mne")
        # print(x_1)
        # print(x_2)
        x_1_pad, x_2_pad = pad_extra_rows(x_1, x_2)
        # print("pad")
        # print(x_1_pad)
        # print(x_2_pad)
        mne_mat = self.mne_layer([x_1_pad, x_2_pad])
        # print("mat")
        # print(mne_mat)
        # mne_mat = torch.sigmoid(mne_mat)
        # print(mne_mat)
        if 'hist_' in self.mne_method:
            # mne_mat = mne_mat[:max_dim, :max_dim]
            with torch.no_grad():
                
                x = torch.histc(mne_mat.to('cpu'), bins=self.num_bins, min=0, max=0). \
                    view((1, -1)).to(FLAGS.device)  # TODO: check gradient
                #print(x)
                # TODO: https://github.com/pytorch/pytorch/issues/1382; first to cpu then back
            x /= torch.sum(x)
        else:
            raise ValueError('Unknown mne method {}'.format(self.mne_method))
        xx = x.detach().numpy()
        if True in np.isnan(xx):
            print("xx is fucked")
        return x

    def _get_mne_output_dim(self, mne_method):
        if 'hist_' in mne_method:  # e.g. hist_16_norm, hist_32_raw
            ss = mne_method.split('_')
            assert (ss[0] == 'hist')
            rtn = int(ss[1])
        else:
            raise NotImplementedError()
        return rtn


class ANPM_Pool(nn.Module):
    """ Attention_NTN_Padding_MNE layer. """

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight,
                 feature_map_dim, bias, ntn_inneract, apply_u,
                 mne_inneract, mne_method, branch_style,
                 reduce_factor, criterion):
        super(ANPM_Pool, self).__init__()

        if 'an' in branch_style:
            self.att_layer = Attention(input_dim=input_dim,
                                       att_times=att_times,
                                       att_num=att_num,
                                       att_style=att_style,
                                       att_weight=att_weight)

            self.ntn_layer = NTN(
                input_dim=input_dim * att_num,
                feature_map_dim=feature_map_dim,
                inneract=ntn_inneract,
                apply_u=apply_u,
                bias=bias)
        else:
            raise ValueError('Must have the Attention-NTN branch')

        self.num_bins = 0
        if 'pm' in branch_style:
            self.mne_layer = MNE(
                input_dim=input_dim,
                inneract=mne_inneract)

            if 'hist_' in mne_method:  # e.g. hist_16_norm, hist_32_raw
                self.num_bins = self._get_mne_output_dim(mne_method)
            else:
                assert False

        self.mne_method = mne_method
        self.branch = branch_style

        proj_layers = []
        D = feature_map_dim  # + self.num_bins
        while D > 1:
            next_D = D // reduce_factor
            if next_D < 1:
                next_D = 1
            # linear_layer = nn.Linear(D, next_D, bias=True)
            linear_layer = nn.Linear(D, next_D, bias=False)
            nn.init.xavier_normal_(linear_layer.weight)
            proj_layers.append(linear_layer)
            proj_layers.append(nn.Sigmoid())
            D = next_D
        self.proj_layers = nn.ModuleList(proj_layers)

        if criterion == 'MSELoss':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError()

    def forward(self, ins, batch_data, model):
        node_set_input = batch_data.split_into_batches(ins)
        single_graph_list = []
        metadata_list = []

        for i in range(0, len(node_set_input)):
            node_set = node_set_input[i]
            edge_index = batch_data.single_graph_list[i].edge_index
            new_x = self.att_layer(node_set)
            new_edge_index = torch.zeros(2,1).long()
            pooled_data = PyGSingleGraphData(x=new_x, edge_index=new_edge_index, edge_attr=None, y=None)
            pooled_data, _ = encode_node_features(pyg_single_g=pooled_data)
            single_graph_list.extend([pooled_data])

        batch_data.update_data(single_graph_list, metadata_list)
        new_ins = batch_data.merge_data['merge'].x

        return new_ins

class Attention(nn.Module):
    """ Attention layer."""

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight):
        super(Attention, self).__init__()
        self.emb_dim = input_dim  # same dimension D as input embeddings
        self.att_times = att_times
        self.att_num = att_num
        self.att_style = att_style
        self.att_weight = att_weight
        assert (self.att_times >= 1)
        assert (self.att_num >= 1)
        assert (self.att_style == 'dot' or self.att_style == 'slm' or
                'ntn_' in self.att_style)

        self.W_0 = glorot([self.emb_dim, self.emb_dim])


    def forward(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_mat(input))
            return rtn
        else:
            return self._call_one_mat(inputs)

    def _call_one_mat(self, inputs):
        outputs = []
        for i in range(self.att_num):
            acts = [inputs]
            assert (self.att_times >= 1)
            output = None
            for _ in range(self.att_times):
                x = acts[-1]  # x is N*D

                temp = torch.mean(x, 0).view((1, -1))  # (1, D)
                h_avg = torch.tanh(torch.mm(temp, self.W_0)) if \
                    self.att_weight else temp
                self.att = self._gen_att(x, h_avg, i)
                output = torch.mm(self.att.view(1, -1), x)
                x_new = torch.mul(x, self.att)
                acts.append(x_new)
            outputs.append(output)
        return torch.cat(outputs, 1)

    def _gen_att(self, x, h_avg, i):
        if self.att_style == 'dot':
            return interact_two_sets_of_vectors(
                x, h_avg, 1,  # interact only once
                W=[torch.eye(self.emb_dim, device=FLAGS.device)],
                act=torch.sigmoid)


class NTN(nn.Module):
    """ NTN layer.
    (Socher, Richard, et al.
    "Reasoning with neural tensor networks for knowledge base completion."
    NIPS. 2013.). """

    def __init__(self, input_dim, feature_map_dim, apply_u,
                 inneract, bias):
        super(NTN, self).__init__()

        self.feature_map_dim = feature_map_dim
        self.apply_u = apply_u
        self.bias = bias
        self.inneract = create_act(inneract)

        self.vars = {}
        self.V = glorot([feature_map_dim, input_dim * 2])
        self.W = glorot([feature_map_dim, input_dim, input_dim])
        if self.bias:
            self.b = nn.Parameter(torch.randn(feature_map_dim).to(FLAGS.device))
        if self.apply_u:
            self.U = glorot([feature_map_dim, 1])

    def forward(self, inputs):
        assert len(inputs) == 2
        return self._call_one_pair(inputs)

    def __call__(self, inputs):
        assert len(inputs) == 2
        return self._call_one_pair(inputs)

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]
        p = Timer()
        r =  interact_two_sets_of_vectors(
            x_1, x_2, self.feature_map_dim,
            V=self.V,
            W=self.W,
            b=self.b if self.bias else None,
            act=self.inneract,
            U=self.U if self.apply_u else None)
        return r


def interact_two_sets_of_vectors(x_1, x_2, interaction_dim, V=None,
                                 W=None, b=None, act=None, U=None):
    """
    Calculates the interaction scores between each row of x_1 (a marix)
        and x_2 ( a vector).
    :param x_1: an (N, D) matrix (if N == 1, a (1, D) vector)
    :param x_2: a (1, D) vector
    :param interaction_dim: number of interactions
    :param V:
    :param W:
    :param b:
    :param act:
    :param U:
    :return: if U is not None, interaction results of (N, interaction_dim)
             if U is None, interaction results of (N, 1)
    """
    feature_map = []
    ii = Timer()
    for i in range(interaction_dim):
        # print("i:",i)
        middle = 0.0
        if V is not None:
            # In case x_2 is a vector but x_1 is a matrix, tile x_2.
            tiled_x_2 = torch.mul(torch.ones_like(x_1, device=FLAGS.device), x_2)
            concat = torch.cat((x_1, tiled_x_2), 1)
            v_weight = V[i].view(-1, 1)
            V_out = torch.mm(concat, v_weight)
            middle += V_out
        if W is not None:
            temp = torch.mm(x_1, W[i])
            h = torch.mm(temp, x_2.t())  # h = K.sum(temp*e2,axis=1)
            middle += h
        if b is not None:
            middle += b[i]
        feature_map.append(middle)

    output = torch.cat(feature_map, 1)
    if act is not None:
        output = act(output)
    if U is not None:
        output = torch.mm(output, U)

    return output


class MNE(nn.Module):
    """ MNE layer. """

    def __init__(self, input_dim, inneract):
        super(MNE, self).__init__()

        self.inneract = create_act(inneract)#activate
        self.input_dim = input_dim

    def forward(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            # Double list.
            rtn = []
            for input in inputs:
                assert (len(input) == 2)
                rtn.append(self._call_one_pair(input))
            return rtn
        else:
            assert (len(inputs) == 2)
            return self._call_one_pair(inputs)

    def _call_one_pair(self, input):
        # Assume x_1 & x_2 are of dimension N * D
        x_1 = input[0]
        x_2 = input[1]

        t = Timer()
        output = torch.mm(x_1, x_2.t())
        t.time_msec_and_clear()

        return self.inneract(output)


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    rtn = nn.Parameter(torch.Tensor(*shape).to(FLAGS.device))
    nn.init.xavier_normal_(rtn)
    return rtn