from load_data import load_dataset
from node_feat import encode_node_features
from config import FLAGS
from torch.utils.data import Dataset as TorchDataset
import torch
from utils_our import get_flags_with_prefix_as_list
from utils import get_save_path, save, load
from os.path import join
from warnings import warn
import os
import networkx as nx
from graph import RegularGraph
import random
from graph_pair import GraphPair
from dataset import OurDataset
import csv
import shutil
import numpy as np
import metis
import pickle
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
import sys

class OurModelData(TorchDataset):
    """Stores a list of graph id pairs with known pairwise results."""

    def __init__(self, dataset, num_node_feat):
        self.dataset, self.num_node_feat = dataset, num_node_feat

        gid_pairs = list(self.dataset.pairs.keys())
        self.gid1gid2_list = torch.tensor(
            sorted(gid_pairs),
            device=FLAGS.device)  # takes a while to move to GPU
    

    def __len__(self):
        return len(self.gid1gid2_list)

    def __getitem__(self, idx):
        return self.gid1gid2_list[idx]

    def get_pairs_as_list(self):
        return [self.dataset.look_up_pair_by_gids(gid1.item(), gid2.item())
                for (gid1, gid2) in self.gid1gid2_list]

    def truncate_large_graphs(self):
        gid_pairs = list(self.dataset.pairs.keys())
        if FLAGS.filter_large_size < 1:
            raise ValueError('Cannot filter graphs of size {} < 1'.format(
                FLAGS.filter_large_size))
        rtn = []
        num_truncaed = 0
        for (gid1, gid2) in gid_pairs:
            g1 = self.dataset.look_up_graph_by_gid(gid1)
            g2 = self.dataset.look_up_graph_by_gid(gid2)
            if g1.get_nxgraph().number_of_nodes() <= FLAGS.filter_large_size and \
                    g2.get_nxgraph().number_of_nodes() <= FLAGS.filter_large_size:
                rtn.append((gid1, gid2))
            else:
                num_truncaed += 1
        warn('{} graph pairs truncated; {} left'.format(num_truncaed, len(rtn)))
        self.gid1gid2_list = torch.tensor(
            sorted(rtn),
            device=FLAGS.device)  # takes a while to move to GPU

    def select_specific_for_debugging(self):
        gid_pairs = list(self.dataset.pairs.keys())
        gids_selected = FLAGS.select_node_pair.split('_')
        assert(len(gids_selected) == 2)
        gid1_selected, gid2_selected = int(gids_selected[0]), int(gids_selected[1])
        rtn = []
        num_truncaed = 0
        for (gid1, gid2) in gid_pairs:
            g1 = self.dataset.look_up_graph_by_gid(gid1).get_nxgraph()
            g2 = self.dataset.look_up_graph_by_gid(gid2).get_nxgraph()
            if g1.graph['gid'] == gid1_selected and g2.graph['gid'] == gid2_selected:
                rtn.append((gid1, gid2))
            else:
                num_truncaed += 1
        warn('{} graph pairs truncated; {} left'.format(num_truncaed, len(rtn)))
        FLAGS.select_node_pair = None # for test
        self.gid1gid2_list = torch.tensor(
            sorted(rtn),
            device=FLAGS.device)  # takes a while to move to GPU


def _none_empty_else_underscore(v):
    if v is None:
        return ''
    return '_{}'.format(v)


def _load_train_test_data_helper():
    if FLAGS.tvt_options == 'all':
        dataset = load_dataset(FLAGS.dataset, 'all', FLAGS.align_metric,
                               FLAGS.node_ordering)
        dataset.print_stats()
        # Node feature encoding must be done at the entire dataset level.
        print('Encoding node features')
        dataset, num_node_feat = encode_node_features(dataset=dataset)
        print('Splitting dataset into train test')
        dataset_train, dataset_test = dataset.tvt_split(
            [FLAGS.train_test_ratio], ['train', 'test'])
    elif FLAGS.tvt_options == 'train,test':
        dataset_test = load_dataset(FLAGS.dataset, 'test', FLAGS.align_metric,
                                    FLAGS.node_ordering)
        dataset_train = load_dataset(FLAGS.dataset, 'train', FLAGS.align_metric,
                                     FLAGS.node_ordering)
        dataset_train, num_node_feat_train = \
            encode_node_features(dataset=dataset_train)
        dataset_test, num_node_feat_test = \
            encode_node_features(dataset=dataset_test)
        if num_node_feat_train != num_node_feat_test:
            raise ValueError('num_node_feat_train != num_node_feat_test '
                             '{] != {}'.
                             format(num_node_feat_train, num_node_feat_test))
        num_node_feat = num_node_feat_train
    else:
        print(FLAGS.tvt_options)
        raise NotImplementedError()
    dataset_train.print_stats()
    dataset_test.print_stats()
    train_data = OurModelData(dataset_train, num_node_feat)
    test_data = OurModelData(dataset_test, num_node_feat)
    return train_data, test_data


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    from torch.utils.data.sampler import SubsetRandomSampler
    from batch import BatchData
    import random

    # print(len(load_dataset(FLAGS.dataset).gs))
    data = OurModelData()
    print(len(data))
    # print('model_data.num_features', data.num_features)
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(dataset_size * 0.2)
    random.Random(123).shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    loader = DataLoader(data, batch_size=3, shuffle=True)
    print(len(loader.dataset))
    for i, batch_gids in enumerate(loader):
        print(i, batch_gids)
        batch_data = BatchData(batch_gids, data.dataset)
        print(batch_data)
        # print(i, batch_data, batch_data.num_graphs, len(loader.dataset))
        # print(batch_data.sp)

def cal_ged(g1,g2):
    g1_gid = g1.graph['gid']
    g2_gid = g2.graph['gid']
    # print(g1_gid,g2_gid)
    if g1_gid == g2_gid:
        ged = 0
    elif g1_gid<100 & g2_gid<100:
        ged = len(g1.nodes())+len(g2.nodes())
    elif g1_gid < 100:
        if g1_gid != g2_gid//100:
            ged = len(g1.nodes())+len(g2.nodes())
        else:
            ged = g2_gid%100
    elif g2_gid < 100:
        if g2_gid != g1_gid//100:
            ged = len(g1.nodes())+len(g2.nodes())
        else:
            ged = g1_gid%100
    else:
        if g1_gid//100 != g2_gid//100:
            ged = len(g1.nodes())+len(g2.nodes())
        else:
            ged = g1_gid%100 + g2_gid%100
    # print(ged)
    return ged

def graph_node_resort(graph):
    g = nx.Graph()
    g_nodes = list(graph.nodes())
    g_edges = list(graph.edges())
    add_edges_list = []
    g.add_nodes_from(list(range(0,len(g_nodes))))
    for u,v,_ in (graph.edges.data()):
        # print(u,v)
        add_edges_list.append((g_nodes.index(u),g_nodes.index(v)))
    g.add_edges_from(add_edges_list)
    return g

aids_node_feature = ['O', 'C', 'N', 'S', 'Ga', 'Br', 'Si', 'P', \
'Cl', 'I', 'Pb', 'Co', 'B', 'Ru', 'Pt', 'Cu', 'F', 'Li', 'Pd', \
'Bi', 'Ho', 'Hg', 'Sb', 'As', 'Tb', 'Sn', 'Se', 'Te', 'Ni']


def aids_graph(g):
    # print(g.nodes())
    g_feature = torch.zeros(len(g.nodes()), 29, device=FLAGS.device)
    for i in range(len(g.nodes())):
        for j in range(len(g.nodes())):
            if g.node[str(j)]["label"] == str(i):
                node_type = g.node[str(j)]["type"]
                # print(node_type)
                # print(aids_node_feature.index(node_type))
                g_feature[i][aids_node_feature.index(node_type)] = 1
    # print("******************")
    return g_feature


def load_our_data(train_path, csv_path):
    train_graph_set = []
    gs1 = []
    gs2 = []

    num_node_feat = FLAGS.num_node_feat

    for g_file in os.listdir(train_path):
        g = nx.read_gexf(train_path + g_file)
        g = nx.Graph(g)
        g.graph['gid'] = int(g_file.replace(".gexf", ""))
        if FLAGS.our_dataset == "AIDS":
            g_feature = aids_graph(g)
        elif FLAGS.our_dataset == "Openssl":
            feature_path = train_path.replace("train/", "feature/").replace("test/", "feature/").replace("val/", "feature/") + g_file.replace(
                ".gexf", ".npy")
            g_feature = torch.Tensor(np.load((feature_path)))
        else:
            g_feature = torch.ones(len(g.nodes()), num_node_feat, device=FLAGS.device)

        g.init_x = g_feature
        train_graph_set.append(RegularGraph(g))
    random.shuffle(train_graph_set)
    pairs = {}
    length = len(train_graph_set)
    for sg1 in range(length):
        for sg2 in range(length):
            gid1, gid2 = train_graph_set[sg1].gid(), train_graph_set[sg2].gid()
            pairs[gid1, gid2] = GraphPair(g1=train_graph_set[sg1], g2=train_graph_set[sg2])

    # 读取csv至字典
    csvFile = open(csv_path, "r")
    reader = csv.reader(csvFile)

    next(reader)
    count = 1
    for item in reader:
        g1 = int(item[0])
        g2 = int(item[1])
        true = int(item[2])
        if (g1, g2) in pairs:
            (pairs[(g1, g2)]).input_ds_true(true)


    name = "our_data"
    tvt = "train"
    align_metric = "ged"
    node_ordering = "bfs"

    data = OurDataset(name=name, gs1=gs1, gs2=gs2, graphs=train_graph_set, natts=['type'], eatts=[], pairs=pairs,
                      tvt=tvt, align_metric=align_metric, node_ordering=node_ordering, glabel=None, loaded_dict=None,
                      mini="mini", my="my")
    train_data = OurModelData(data, num_node_feat)
    return train_data


def load_data():
    if FLAGS.dos_true == 'sim':
        train_data = load_our_data("../../data/"+FLAGS.dataset+"/train/","../../data/pair_class/"+FLAGS.dataset+"_pair_class.csv")

        test_data = load_our_data("../../data/"+FLAGS.dataset+"/test/","../../data/pair_class/"+FLAGS.dataset+"_pair_class.csv")
        val_data = load_our_data("../../data/" + FLAGS.dataset + "/val/",
                                 "../../data/pair_class/" + FLAGS.dataset + "_pair_class.csv")
    else:
        train_data = load_our_data("../../data/" + FLAGS.dataset + "/train/",
                                   "../../data/" + FLAGS.dataset + "/" + FLAGS.dataset + ".csv")

        test_data = load_our_data("../../data/" + FLAGS.dataset + "/test/",
                                  "../../data/" + FLAGS.dataset + "/" + FLAGS.dataset + ".csv")
        val_data = load_our_data("../../data/" + FLAGS.dataset + "/val/",
                                 "../../data/" + FLAGS.dataset + "/" + FLAGS.dataset + ".csv")

    return train_data, test_data,val_data
