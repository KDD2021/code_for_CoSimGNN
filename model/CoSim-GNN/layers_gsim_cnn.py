from layers import create_act
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from utils_our import pad_extra_rows
from config import FLAGS

class GraphConvolutionCollector(nn.Module):
    def __init__(self, gcn_num, fix_size, mode, padding_value, align_corners, **kwargs):
        super().__init__()
        self.gcn_num = gcn_num
        self.MNEResize = MNEResize(inneract=False, fix_size=fix_size,
                                    mode=mode, padding_value=padding_value,
                                    align_corners=align_corners, **kwargs)

    def forward(self, inputs, batch_data, model):
        assert (type(inputs) is list and inputs)
        assert (len(inputs) == self.gcn_num)

        gcn_ins = []  #each item is a list of similarity matricies for a given gcn output
        for gcn_layer_out in inputs:
            gcn_ins.append(torch.stack(self.MNEResize(gcn_layer_out,batch_data, model)))

        return torch.stack(gcn_ins) #gcn_num by Num_pairs by Nmax by Nmax similarity matricies len(gcn_ins)==gcn_num

class MNEResize(nn.Module):
    def __init__(self, inneract, fix_size, mode, padding_value,
                 align_corners, **kwargs):
        super().__init__()
        self.inneract = inneract
        self.fix_size = fix_size
        self.align_corners = align_corners
        self.padding_value = padding_value
        modes =  ["bilinear", "nearest", "bicubic", "area"]
        if mode < 0 or mode > len(modes):
            raise RuntimeError('Unknown MNE resize mode {}'.format(self.mode))
        self.mode = modes[mode]

    def forward(self, ins, batch_data, model):
        x = ins  # x has shape N(gs) by D
        ind_list = batch_data.merge_data["ind_list"]
        sim_mat_l = [] #list of similarity matricies (should be len(ind_list)/2 items)
        for i in range(0,len(ind_list), 2):
            g1_ind = i
            g2_ind = i + 1
            g1x = x[ind_list[g1_ind][0]: ind_list[g1_ind][1]]
            g2x = x[ind_list[g2_ind][0]: ind_list[g2_ind][1]]
            sim_mat_l.append(self._call_one_pair(g1x, g2x))
        return sim_mat_l



    def _call_one_pair(self, g1x, g2x):
        x1_pad, x2_pad = pad_extra_rows(g1x, g2x, self.padding_value)
        sim_mat_temp = torch.matmul(x1_pad, torch.t(x2_pad))

        sim_mat = sim_mat_temp.unsqueeze(0).unsqueeze(0) if not self.inneract else \
            self.inneract(sim_mat_temp).unsqueeze(0).unsqueeze(0) #need for dims for bilinear interpolation
        sim_mat_resize = F.interpolate(sim_mat,
                                       size=[self.fix_size, self.fix_size],
                                       mode=self.mode,
                                       align_corners=self.align_corners)
        return sim_mat_resize.squeeze().unsqueeze(0)

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, gcn_num, bias,
                 poolsize, act, end_cnn=False, **kwargs):
        super().__init__()

        #same padding calc
        self.kernel_size = kernel_size
        self.stride = stride
        self.convs  = nn.ModuleList()
        self.maxpools = nn.ModuleList()
        self.pool_stride =  poolsize
        self.pool_size = poolsize
        self.activation = create_act(act)
        self.end_cnn = end_cnn
        self.gcn_num = gcn_num
        self.out_channels = out_channels
        self.reduce_factor = 2 #the same as simgnn
        
        #MLP 
        if self.end_cnn:
            proj_layers = []
            D = self.gcn_num * self.out_channels
            while D > 1:
                next_D = D // self.reduce_factor
                if next_D < 1:
                    next_D = 1
                linear_layer = nn.Linear(D, next_D, bias=False)
                nn.init.xavier_normal_(linear_layer.weight)
                proj_layers.append(linear_layer)
                if next_D != 1:
                    proj_layers.append(nn.ReLU())
                D = next_D
            self.proj_layers = nn.ModuleList(proj_layers)
            self.criterion = nn.MSELoss(reduction='mean')


        for i in range(gcn_num):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias))
            self.convs[-1].apply(self.weights_init)
            self.maxpools.append(nn.MaxPool2d(poolsize, stride=poolsize))

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)

    def forward(self, ins, batch_data, model):
        inshape = ins[0].shape
        num_batch = inshape[0]
        #print("num_batch",num_batch)
        pair_list = batch_data.split_into_pair_list(ins, 'x')
        if self.end_cnn:
            true = torch.zeros(len(pair_list), 1, device=FLAGS.device)
            for i, pair in enumerate(pair_list):
                true[i] = pair.get_ds_true(
                    FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred, FLAGS.ds_kernel)

        H_in = inshape[2]
        W_in = inshape[3]

        pad_cnn_h = self._same_pad_calc(H_in, self.kernel_size, self.stride)
        pad_cnn_w = self._same_pad_calc(W_in, self.kernel_size, self.stride)
        pad_pool_h = self._same_pad_calc(H_in, self.pool_size, self.pool_stride)
        pad_pool_w = self._same_pad_calc(W_in, self.pool_size, self.pool_stride)

        result = []

        for i in range(self.gcn_num):
            result.append(self._conv_and_pool(ins[i, :, :, :, :], i, pad_cnn_h,
                                              pad_cnn_w, pad_pool_h,
                                              pad_pool_w))
        rtn = torch.stack(result)
        if self.end_cnn:
            rtn = rtn.squeeze(4).squeeze(3)
            rtn = rtn.permute(1,0,2)
            rtn = torch.reshape(rtn, [num_batch, self.out_channels * self.gcn_num])
            #MLP
            for proj_layer in self.proj_layers:
                rtn = proj_layer(rtn)
            for i, pair in enumerate(pair_list):
                pair.assign_ds_pred(rtn[i])
            if FLAGS.dos_true == 'sim':
                true_score = torch.tensor(true).float()
                true_score.requires_grad = True
                loss = torch.sum(torch.relu(0.1 + true_score * (0.5 - rtn)))
                pos_index = torch.where(rtn > 0.6)
                neg_index = torch.where(rtn < 0.4)
                classification = torch.zeros(true_score.size()).long()
                classification[neg_index] = -1
                classification[pos_index] = 1
                auc = classification * true_score.long() - 1
                index = auc.nonzero()
                batch_data.auc_classification = true_score.size()[0] - index.size()[0]
                batch_data.sample_num = true_score.size()[0]
            else:
                loss = self.criterion(rtn, true)

            return loss
        return rtn


    def _conv_and_pool(self, gcn_sim_mat, gcn_ind, pad_cnn_h, pad_cnn_w, pad_pool_h, pad_pool_w):
        gcn_sim_mat_pad = F.pad(gcn_sim_mat, (pad_cnn_w[0], pad_cnn_w[1],pad_cnn_h[0], pad_cnn_h[1]))

        conv_x = self.convs[gcn_ind](gcn_sim_mat_pad)
        conv_x = self.activation(conv_x)

        conv_x = F.pad(conv_x, (pad_pool_w[0], pad_pool_w[1],pad_pool_h[0], pad_pool_h[1]))
        pool_x = self.maxpools[gcn_ind](conv_x)
        return pool_x


    # to mimic the tensorflow implementation
    def _same_pad_calc(self, in_dim, kernel_size, stride):
        pad = ((math.ceil(in_dim/stride)-1)*stride-in_dim + kernel_size)
        if pad % 2 == 0:
            return (int(pad/2), int(pad/2))
        else:
            return (int(pad/2), int(pad/2)+1)

