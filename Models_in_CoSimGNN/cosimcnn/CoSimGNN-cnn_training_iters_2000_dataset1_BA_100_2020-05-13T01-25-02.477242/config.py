from solve_parent_dir import solve_parent_dir
from dataset_config import get_dataset_conf
from dist_sim import get_ds_metric_config
from utils import format_str_list, C, get_user, get_host
import argparse
import torch

solve_parent_dir()
parser = argparse.ArgumentParser()

"""
Data.
"""

""" 
dataset: 
    (for MCS)
    debug, mini_debug, debug_no-1, mini_debug_no-1 debug_single_iso
    mcsplain mcsplain-connected sip (tune sip with smaller D/batch_size)
    ptc redditmulti10k
    mcs33ve (dropped) mcs33ve-connected (dropped)
    aids700nef linux imdbmulti ptc nci109 webeasy redditmulti10k mutag
    (for similarity)
    aids700nef_old linux_old imdbmulti_old ptc_old aids700nef_old_small
"""
# dataset = 'debug_BA:train_size=1000,test_size=100,num_nodes_training=16,num_nodes_testing=64'
#dataset = 'debug_BA:train_size=1000,test_size=100,num_nodes_training=0,num_nodes_testing=0'
dataset = 'aids700nef'
parser.add_argument('--dataset', default=dataset)

dataset_version = None  # 'v2'
parser.add_argument('--dataset_version', default=dataset_version)

filter_large_size = None 
parser.add_argument('--filter_large_size', type=int, default=filter_large_size)  # None or >= 1

select_node_pair = None
parser.add_argument('--select_node_pair', type=str, default=select_node_pair)  # None or gid1_gid2

c = C()#counting
parser.add_argument('--node_fe_{}'.format(c.c()), default='one_hot')

# parser.add_argument('--node_fe_{}'.format(c.c()),
#                     default='local_degree_profile')

natts, eatts, tvt_options, align_metric_options, *_ = \
    get_dataset_conf(dataset)

""" Must use exactly one alignment metric across the entire run. """
#align_metric = align_metric_options[0]
#if len(align_metric_options) == 2:
""" Choose which metric to use. """
#align_metric = 'ged'
align_metric = 'mcs'
parser.add_argument('--align_metric', default=align_metric)

#dos_true, _ = get_ds_metric_config(align_metric)
dos_true="dist"
parser.add_argument('--dos_true', default=dos_true)

# Assume the model predicts None. May be updated below.
dos_pred = "sim"#None

parser.add_argument('--node_feats', default=format_str_list(natts))

parser.add_argument('--edge_feats', default=format_str_list(eatts))
"""
Evaluation.
"""

parser.add_argument('--tvt_options', default=format_str_list(tvt_options))

""" holdout, (TODO) <k>-fold. """
tvt_strategy = 'holdout'
parser.add_argument('--tvt_strategy', default=tvt_strategy)

if tvt_strategy == 'holdout':
    if tvt_options == ['all']:
        parser.add_argument('--train_test_ratio', type=float, default=0.8)
    elif tvt_options == ['train', 'test']:
        pass
    else:
        raise NotImplementedError()
else:
    raise NotImplementedError()

parser.add_argument('--debug', type=bool, default='debug' in dataset)

# Assume normalization is needed for true dist/sim scores.
parser.add_argument('--ds_norm', type=bool, default=True)

parser.add_argument('--ds_kernel', default='exp')

"""
Model.
"""
#model = 'simgnn'
# model = 'simgnn_fast'
#model = 'gsim_cnn'
#model = 'CoSimGNN'
#model = 'GCN-Mean'
#model = 'GCN-Max'
model = 'CoSimGNN-cnn'
#model = 'GMN_icml_mlp_gin'
#model = 'CoSimGNN-MemPool'
#model = 'GMN_icml_mlp'
parser.add_argument('--model', default=model)

rank = True
# rank = False
parser.add_argument('--rank', type=bool, default=rank)

# traditional_method = False
traditional_method = True
parser.add_argument('--traditional_method', type=bool, default=traditional_method)

num_partitions = 3
parser.add_argument('--num_partitions', default=num_partitions)

num_select = 3
parser.add_argument('--num_select', default=num_select)

n_outputs = 10 # TODO: tune this
parser.add_argument('--n_outputs', type=int, default=n_outputs)

hard_mask = True
parser.add_argument('--hard_mask', type=bool, default=hard_mask)

model_name = 'fancy'
parser.add_argument('--model_name', default=model_name)

c = C()

D = 64

if dataset == 'aids700nef':
    alpha = 1  # 0.01
    beta = 0  # 0.01
    gamma = 0  # 0.2
    tau = 0  # 1
    theta = 0.5
elif dataset == 'linux':
    alpha = 1
    beta = 0  # 2.5
    gamma = 0  # 10
    tau = 0  # 1
    theta = 0.5  # 0.5
elif dataset == 'imdbmulti':
    alpha = 1
    beta = 0  # 2
    gamma = 0  # 0.1
    tau = 0  # 1
    theta = 0.7  # 0.3
elif dataset == 'redditmulti10k':
    alpha = 1  # 10
    beta = 0  # 10
    gamma = 0  # 5
    tau = 0  # 1e-2
    theta = 0.5  # 0.5
elif dataset == 'mutag':
    alpha = 1  # 1
    beta = 0  # 1
    gamma = 0  # 100
    tau = 0  # 1
    theta = 0.5  # 0.5
elif dataset == 'alchemy':
    alpha = 1  # 1
    beta = 0  # 1
    gamma = 0  # 100
    tau = 0  # 1
    theta = 0.5  # 0.5
else:
    alpha = 1  # 1
    beta = 0  # 0
    gamma = 0  # 0
    tau = 0  # 1
    theta = 0.5  # 0.5
    # assert False

parser.add_argument('--theta', type=float, default=theta)

########################################
# Node Embedding
########################################
if model == 'simgnn':
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,output_dim={},act=relu,bn=False'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=False'.format(D, D // 2)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=False'.format(D // 2, D // 2 // 2)

    # D //= 2
    parser.add_argument(n, default=s)
    
    n = '--layer_{}'.format(c.c())
    input_dim = 16
    att_times = 1
    att_num = 1
    att_style = 'dot'
    att_weight = True
    feature_map_dim = 16
    bias = True
    ntn_inneract = "relu" #tanh, relu before
    apply_u = False#True
    mne_inneract = "relu" #sigmoid
    mne_method = 'hist_16'
    branch_style = 'an'
    reduce_factor = 2
    criterion = 'MSELoss'
    s = 'simgnn:input_dim={},att_times={},att_num={},att_style={},att_weight={},feature_map_dim={},bias={},ntn_inneract={},apply_u={},mne_inneract={},mne_method={},branch_style={},reduce_factor={},criterion={}'.\
    format(input_dim, att_times, att_num, att_style,att_weight, feature_map_dim, bias, ntn_inneract, apply_u, mne_inneract, mne_method, branch_style, reduce_factor, criterion)
    parser.add_argument(n, default=s)
elif model == 'simgnn_fast':
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,output_dim={},act=prelu,bn=True'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=prelu,bn=True'. \
        format(D, D // 2)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=prelu,bn=True'. \
        format(D // 2, D // 2 // 2)
    # D //= 2
    parser.add_argument(n, default=s)
    # D = 16
    # n = '--layer_{}'.format(c.c())
    # s = 'GMNEncoder_FAST:,output_dim={},act=relu'.format(D)
    # parser.add_argument(n, default=s)
    
    n = '--layer_{}'.format(c.c())
    input_dim = 16
    att_times = 1
    att_num = 1
    att_style = 'dot'#'ntn_16'
    att_weight = True
    feature_map_dim = 16
    bias = True
    ntn_inneract = "relu" #tanh, relu before
    apply_u = False
    mne_inneract = "relu" #sigmoid
    mne_method = 'hist_16'
    branch_style = 'anpm'
    reduce_factor = 2
    criterion = 'MSELoss'
    s = 'simgnn_fast:input_dim={},att_times={},att_num={},att_style={},att_weight={},feature_map_dim={},bias={},ntn_inneract={},apply_u={},mne_inneract={},mne_method={},branch_style={},reduce_factor={},criterion={}'.\
    format(input_dim, att_times, att_num, att_style,att_weight, feature_map_dim, bias, ntn_inneract, apply_u, mne_inneract, mne_method, branch_style, reduce_factor, criterion)
    parser.add_argument(n, default=s)

# begin gmn...
#     D = 16
#     # n = '--layer_{}'.format(c.c())
#     # s = 'GMNEncoder_FAST:input_dim={},output_dim={},act=relu'.format(D,D)
#     # parser.add_argument(n, default=s)
#
#     more_nn = None  # 'EdgeConv'  # None, 'EdgeConv'
#
#     n = '--layer_{}'.format(c.c())
#     s = 'GMNPropagator_FAST:input_dim={},' \
#         'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
#     parser.add_argument(n, default=s)
#
#     n = '--layer_{}'.format(c.c())
#     s = 'GMNPropagator_FAST:input_dim={},' \
#         'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
#     parser.add_argument(n, default=s)
#
#     n = '--layer_{}'.format(c.c())
#     s = 'GMNPropagator_FAST:input_dim={},' \
#         'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
#     parser.add_argument(n, default=s)
#
#     n = '--layer_{}'.format(c.c())
#     s = 'GMNAggregator_FAST:input_dim={},output_dim={}'.format(D, D)
#     parser.add_argument(n, default=s)
#
#     n = '--layer_{}'.format(c.c())
#     s = 'GMNLoss_FAST:ds_metric=cosine'
#     parser.add_argument(n, default=s)

elif model == 'gsim_cnn':
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,output_dim={},act=relu,bn=True'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D // 2)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D // 2, D // 2 // 2)
    # D //= 2
    parser.add_argument(n, default=s)
    
    n = '--layer_{}'.format(c.c())
    gcn_num = 3
    fix_size = 10
    mode = 0
    padding_value = 0
    align_corners =  False
    s = 'GraphConvolutionCollector:gcn_num={},fix_size={},mode={},padding_value={},align_corners={}'. \
        format(gcn_num, fix_size, mode, padding_value, align_corners)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    in_channels = 1
    out_channels = 16
    kernel_size = 6
    stride = 1
    gcn_num = 3
    bias = True
    poolsize = 2
    act = 'relu'
    end_cnn = False
    s = 'CNN:in_channels={},out_channels={},kernel_size={},stride={},gcn_num={},bias={},poolsize={},act={},end_cnn={}'. \
        format(in_channels, out_channels, kernel_size, stride, gcn_num, bias, poolsize, act, end_cnn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    in_channels = 16
    out_channels = 32
    kernel_size = 6
    stride = 1
    gcn_num = 3
    bias = True
    poolsize = 2
    act = 'relu'
    end_cnn = False
    s = 'CNN:in_channels={},out_channels={},kernel_size={},stride={},gcn_num={},bias={},poolsize={},act={},end_cnn={}'. \
        format(in_channels, out_channels, kernel_size, stride, gcn_num, bias, poolsize, act, end_cnn)
    parser.add_argument(n, default=s)


    n = '--layer_{}'.format(c.c())
    in_channels = 32
    out_channels = 64
    kernel_size = 5
    stride = 1
    gcn_num = 3
    bias = True
    poolsize = 2
    act = 'relu'
    end_cnn = False
    s = 'CNN:in_channels={},out_channels={},kernel_size={},stride={},gcn_num={},bias={},poolsize={},act={},end_cnn={}'. \
        format(in_channels, out_channels, kernel_size, stride, gcn_num, bias, poolsize, act, end_cnn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    in_channels = 64
    out_channels = 128
    kernel_size = 5
    stride = 1
    gcn_num = 3
    bias = True
    poolsize = 3
    act = 'relu'
    end_cnn = False
    s = 'CNN:in_channels={},out_channels={},kernel_size={},stride={},gcn_num={},bias={},poolsize={},act={},end_cnn={}'. \
        format(in_channels, out_channels, kernel_size, stride, gcn_num, bias, poolsize, act, end_cnn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    in_channels = 128
    out_channels = 128
    kernel_size = 5
    stride = 1
    gcn_num = 3
    bias = True
    poolsize = 3
    act = 'relu'
    end_cnn = True
    s = 'CNN:in_channels={},out_channels={},kernel_size={},stride={},gcn_num={},bias={},poolsize={},act={},end_cnn={}'. \
        format(in_channels, out_channels, kernel_size, stride, gcn_num, bias, poolsize, act, end_cnn)
    parser.add_argument(n, default=s)

elif model == 'GCN-Mean':
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,output_dim={},act=relu,bn=True'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'AVG_Pooling:input_dim={},end_pooling=False,max_pooling=False'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'AVG_Pooling:input_dim={},end_pooling=True,max_pooling=False'.format(D)
    parser.add_argument(n, default=s)

elif model == 'GCN-Max':
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,output_dim={},act=relu,bn=True'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'AVG_Pooling:input_dim={},end_pooling=False,max_pooling=True'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gcn,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'AVG_Pooling:input_dim={},end_pooling=True,max_pooling=True'.format(D)
    parser.add_argument(n, default=s)

elif model =='GMN_icml_mlp':

    n = '--layer_{}'.format(c.c())
    s = 'GMNEncoder:output_dim={},act=relu'.format(D)
    parser.add_argument(n, default=s)

    more_nn = None  # 'EdgeConv'  # None, 'EdgeConv'

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNAggregator:input_dim={},output_dim={}'.format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNLoss:ds_metric=cosine'
    parser.add_argument(n, default=s)
elif model =='GMN_icml_mlp_gin':

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,output_dim={},act=relu,bn=True'. \
        format(D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    more_nn = None  # 'EdgeConv'  # None, 'EdgeConv'

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNAggregator:input_dim={},output_dim={}'.format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNLoss:ds_metric=cosine'
    parser.add_argument(n, default=s)
elif model == 'CoSimGNN':

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,output_dim={},act=relu,bn=True'. \
        format(D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'Memory_Based_Pooling:heads={},input_dim={},output_num={},output_dim={},CosimGNN=True'.format(5,D,10,D)
    parser.add_argument(n, default=s)

    more_nn = None  # 'EdgeConv'  # None, 'EdgeConv'

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNAggregator:input_dim={},output_dim={}'.format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNLoss:ds_metric=cosine'
    parser.add_argument(n, default=s)

elif model == 'CoSimGNN-cnn':

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,output_dim={},act=relu,bn=True'. \
        format(D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'Memory_Based_Pooling:heads={},input_dim={},output_num={},output_dim={},CosimGNN={}'.format(5,D,10,D,True)
    parser.add_argument(n, default=s)

    more_nn = None  # 'EdgeConv'  # None, 'EdgeConv'

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    gcn_num = 3
    fix_size = 10
    mode = 0
    padding_value = 0
    align_corners = False
    s = 'GraphConvolutionCollector:gcn_num={},fix_size={},mode={},padding_value={},align_corners={}'. \
        format(gcn_num, fix_size, mode, padding_value, align_corners)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    in_channels = 1
    out_channels = 16
    kernel_size = 6
    stride = 1
    gcn_num = 3
    bias = True
    poolsize = 2
    act = 'relu'
    end_cnn = False
    s = 'CNN:in_channels={},out_channels={},kernel_size={},stride={},gcn_num={},bias={},poolsize={},act={},end_cnn={}'. \
        format(in_channels, out_channels, kernel_size, stride, gcn_num, bias, poolsize, act, end_cnn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    in_channels = 16
    out_channels = 32
    kernel_size = 6
    stride = 1
    gcn_num = 3
    bias = True
    poolsize = 2
    act = 'relu'
    end_cnn = False
    s = 'CNN:in_channels={},out_channels={},kernel_size={},stride={},gcn_num={},bias={},poolsize={},act={},end_cnn={}'. \
        format(in_channels, out_channels, kernel_size, stride, gcn_num, bias, poolsize, act, end_cnn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    in_channels = 32
    out_channels = 64
    kernel_size = 5
    stride = 1
    gcn_num = 3
    bias = True
    poolsize = 2
    act = 'relu'
    end_cnn = False
    s = 'CNN:in_channels={},out_channels={},kernel_size={},stride={},gcn_num={},bias={},poolsize={},act={},end_cnn={}'. \
        format(in_channels, out_channels, kernel_size, stride, gcn_num, bias, poolsize, act, end_cnn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    in_channels = 64
    out_channels = 128
    kernel_size = 5
    stride = 1
    gcn_num = 3
    bias = True
    poolsize = 3
    act = 'relu'
    end_cnn = False
    s = 'CNN:in_channels={},out_channels={},kernel_size={},stride={},gcn_num={},bias={},poolsize={},act={},end_cnn={}'. \
        format(in_channels, out_channels, kernel_size, stride, gcn_num, bias, poolsize, act, end_cnn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    in_channels = 128
    out_channels = 128
    kernel_size = 5
    stride = 1
    gcn_num = 3
    bias = True
    poolsize = 3
    act = 'relu'
    end_cnn = True
    s = 'CNN:in_channels={},out_channels={},kernel_size={},stride={},gcn_num={},bias={},poolsize={},act={},end_cnn={}'. \
        format(in_channels, out_channels, kernel_size, stride, gcn_num, bias, poolsize, act, end_cnn)
    parser.add_argument(n, default=s)
elif model == 'CoSimGNN-MemPool':
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,output_dim={},act=relu,bn=True'. \
        format(D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gin,input_dim={},output_dim={},act=relu,bn=True'. \
        format(D, D)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'Memory_Based_Pooling:heads={},input_dim={},output_num={},output_dim={},CosimGNN=False'.format(5, D, 10, D)
    parser.add_argument(n, default=s)

    more_nn = None  # 'EdgeConv'  # None, 'EdgeConv'

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,more_nn={}'.format(D, D, more_nn)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNAggregator:input_dim={},output_dim={}'.format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNLoss:ds_metric=cosine'
    parser.add_argument(n, default=s)

parser.add_argument('--layer_num', type=int, default=c.t())

# Finally we set dos_pred.
parser.add_argument('--dos_pred', default=dos_pred)

"""
Optimization.
"""
# lr = 1e-1
# lr = 1e-2
lr = 1e-3
# lr = 0
parser.add_argument('--lr', type=float, default=lr)

gpu = -1
device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)

sub_graph_path = "../../sub_graph/"
parser.add_argument('--sub_graph_path', type=str, default=sub_graph_path)

num_epochs = None
parser.add_argument('--num_epochs', default=num_epochs)

'''
lmbda = 1.0
parser.add_argument('--lmbda', type=float, default=lmbda)
'''

num_iters = 10# TODO: tune this #I changede it fj
parser.add_argument('--num_iters', type=int, default=num_iters)

parser.add_argument('--dataset_dzh', type=str, default='dataset1_BA_60')

validation = False  # TODO: tune this
parser.add_argument('--validation', type=bool, default=validation)

throw_away = 0  # TODO: tune this
parser.add_argument('--throw_away', type=float, default=throw_away)

print_every_iters = 10
parser.add_argument('--print_every_iters', type=int, default=print_every_iters)

only_iters_for_debug = None  # only train and test this number of pairs
parser.add_argument('--only_iters_for_debug', type=int, default=only_iters_for_debug)

save_model = True  # TODO: tune this
parser.add_argument('--save_model', type=bool, default=save_model)

load_model = None
#load_model = '/test/models/CoSimGNN_training_iters_10000_dataset2_ER_100_2020-05-02T09-09-11.156602/trained_model_88.pt'
#load_model = '/test/models/CoSimGNN_training_iters_2000_dataset1_BA_200_2020-05-02T04-02-45.299293/trained_model_17.pt'
#load_model = '/test/models/CoSimGNN_training_iters_2000_dataset1_BA_100_2020-05-07T21-39-37.864991/trained_model_17.pt'
#load_model = '/test/models/CoSimGNN_training_iters_2000_dataset1_BA_60_2020-05-01T22-44-03.898100/trained_model_17.pt'
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/GCN-Max_training_iters_5000_dataset1_BA_60_2020-05-10T04-22-21.082836/trained_model_0.pt'#2
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/CoSimGNN-cnn_training_iters_5000_dataset1_BA_60_2020-05-09T17-06-09.931167/trained_model_2.pt'
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/GCN-Mean_training_iters_5000_dataset1_BA_60_2020-05-09T10-54-51.488649/trained_model_8.pt'
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/gsim_cnn_training_iters_5000_dataset1_BA_60_2020-05-08T23-50-00.162445/trained_model_9.pt'
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/CoSimGNN_training_iters_5000_dataset1_BA_60_2020-05-08T20-08-21.946794/trained_model_8.pt'
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/GMN_icml_mlp_training_iters_5000_dataset1_BA_60_2020-05-08T09-35-54.711797/trained_model_7.pt'
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/CoSimGNN_training_iters_5000_dataset1_BA_60_2020-05-08T20-08-21.946794/trained_model_10.pt'
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/GMN_icml_mlp_training_iters_2000_dataset1_BA_100_2020-05-08T02-06-04.936265/trained_model_13.pt'
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/gsim_cnn_training_iters_2000_dataset1_BA_100_2020-05-08T01-13-35.543753/trained_model_3.pt'
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/GCN-Max_training_iters_1000_dataset1_BA_100_2020-05-08T17-56-24.784499/trained_model_2.pt'
# load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/GCN-Mean_training_iters_1000_dataset1_BA_100_2020-05-08T18-42-33.831982/trained_model_3.pt'
#load_model = '/test/part/GraphMatching_submission/model/OurGED/logs/GMN_icml_mlp_training_iters_5000_dataset1_BA_60_2020-05-08T09-35-54.711797/trained_model_12.pt'
#load_model = "/test/graph_up/GraphMatching_submission/model/OurGED/logs/CoSimGNN_training_iters_2000_dataset1_BA_60_2020-05-01T22-44-03.898100/trained_model_17.pt"
#load_model = '/test/graph_up/GraphMatching_submission/model/OurGED/logs/GCN-Max_training_iters_1000_dataset1_BA_200_2020-05-05T22-30-51.024183/trained_model_7.pt'
#load_model = '/test/graph_up/GraphMatching_submission/model/OurGED/logs/GCN-Max_training_iters_1000_dataset1_BA_60_2020-05-03T08-30-23.693186/trained_model_4.pt'
#load_model = '/test/graph_up/GraphMatching_submission/model/OurGED/logs/GCN-Mean_training_iters_500_dataset1_BA_60_2020-05-03T05-42-48.916612/trained_model_4.pt'
#load_model = '/test/graph_up/GraphMatching_submission/model/OurGED/logs/GCN-Mean_training_iters_300_dataset1_BA_200_2020-05-03T06-52-20.150625/trained_model_2.pt'
#load_model = '/test/graphupup/gsim_cnn_training_iters_2000_dataset1_BA_60_2020-04-29T08-37-15.979294/trained_model_17.pt'
#load_model = '/test/graphupup/gsim_cnn_training_iters_2000_dataset1_BA_200_2020-04-29T01-09-56.315274/trained_model_17.pt'
#load_model = '/test/graphupup/GMN_icml_mlp_training_iters_2000_dataset1_BA_200_2020-04-30T01-43-54.360670/trained_model_17.pt'
#load_model = '/test/graphupup/GMN_icml_mlp_training_iters_2000_dataset1_BA_60_2020-04-29T22-11-06.909077/trained_model_17.pt'
#load_model = '/test/graphupup/gsim_cnn_training_iters_2000_dataset1_BA_60_2020-04-29T08-37-15.979294/trained_model_17.pt'
# "/Users/chenrj/Desktop/Graph_sim/GraphMatching_submission/model/OurGED/our_model/CoSimGNN_training_iters_2000_dataset2_ER_100_2020-05-02T07-58-31.533058/trained_model_17.pt"
#"/Users/chenrj/Desktop/Graph_sim/GraphMatching_submission/model/OurGED/logs/GCN-Max_training_iters_1000_dataset1_BA_100_2020-05-03T09-42-33.663708/trained_model_7.pt"
# load_model = "/home/russell/russell/GraphMatching_BA_small/model/OurGED/logs/gsim_cnn_aids700nef_2020-04-22T21-16-16.159028/trained_model_3.pt"
# load_model = "/home/russell/russell/GraphMatching_BA_small/model/OurGED/logs/gsim_cnn_aids700nef_2020-04-23T19-48-42.292400/trained_model_4.pt"
# load_model = "/home/russell/russell/GraphMatching_BA_small/model/OurGED/logs/gsim_cnn_aids700nef_2020-04-23T20-30-33.319191/trained_model_0.pt"
# load_model = "/home/russell/russell/GraphMatching_BA_small/model/OurGED/logs/gsim_cnn_aids700nef_2020-04-23T20-39-49.479164/trained_model_1.pt"
# load_model = "/home/russell/russell/GraphMatching_BA_small/model/OurGED/logs/simgnn_aids700nef_2020-04-23T21-12-49.297075/trained_model_5.pt"
# load_model = "/home/russell/russell/GraphMatching_BA_small/model/OurGED/logs/simgnn_aids700nef_2020-04-23T22-16-05.753234/trained_model_17.pt"
# load_model = "/home/russell/russell/GraphMatching_BA_small/model/OurGED/logs/GMN_icml_mlp_aids700nef_2020-04-24T01-12-08.403360/trained_model_2.pt"
parser.add_argument('--load_model', default=load_model)

batch_size = 128
parser.add_argument('--batch_size', type=int, default=batch_size)

num_node_feat = 30
parser.add_argument('--num_node_feat', type=bool, default=num_node_feat)
# draw_sub_graph = True
draw_sub_graph = False
parser.add_argument('--draw_sub_graph', type=bool, default=draw_sub_graph)

# save_sub_graph = True
save_sub_graph = False
parser.add_argument('--save_sub_graph', type=bool, default=save_sub_graph)

save_every_epochs = 1
parser.add_argument('--save_every_epochs', type=int, default=save_every_epochs)

parser.add_argument('--node_ordering', default='bfs')
parser.add_argument('--no_probability', default=False)
parser.add_argument('--positional_encoding', default=False)  # TODO: dataset.py cannot see this

"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

FLAGS = parser.parse_args()