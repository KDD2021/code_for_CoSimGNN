from config import FLAGS
from layers_factory import create_layers
from layers import NodeEmbedding
from layers_subgraph import NTN, glorot
import torch.nn as nn
from time import time
from utils import Timer
from utils_our import get_branch_names, get_flag
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
import numpy as np
import community
import networkx as nx
import matplotlib.pyplot as plt
import torch
from graph import RegularGraph
from dataset import OurDataset
from graph_pair import GraphPair
from batch import BatchData
import datetime
import metis
import pickle
#from more_itertools import chunked

class Model(nn.Module):
    def __init__(self, data):
        super(Model, self).__init__()

        self.train_data = data
        self.num_node_feat = data.num_node_feat

        self.layers = create_layers(self, 'layer', FLAGS.layer_num)
        assert (len(self.layers) > 0)
        self._print_layers(None, self.layers)

        bnames = get_branch_names()
        
        print(bnames)
        if bnames:
            for bname in bnames:
                blayers = create_layers(
                    self, bname,
                    get_flag('{}_layer_num'.format(bname), check=True))
                setattr(self, bname, blayers)  # so that print(model) can print
                self._print_layers(bname, getattr(self, bname))

        self.layer_output = {}

        self.criterion = nn.MSELoss(reduction='mean')

        self.GNN_1 = NodeEmbedding(type="gin",in_dim=FLAGS.num_node_feat,out_dim=64,act="prelu",bn=True)
        self.GNN_2 = NodeEmbedding(type="gin",in_dim=64,out_dim=32,act="prelu",bn=True)
        self.GNN_3 = NodeEmbedding(type="gin",in_dim=32,out_dim=16,act="prelu",bn=True)
        # self.vars = {}
        self.emb_dim = 16
        self.W_0 = glorot([self.emb_dim, self.emb_dim])
        
        self.ntn_layer = NTN(input_dim=16,feature_map_dim=16,inneract="relu",apply_u=False,bias=False)
        graph_mlp_layers = []
        D = 16
        while D > 1:
            next_D = D // 2
            if next_D < 1:
                next_D = 1
            linear_layer = nn.Linear(D, next_D, bias=False)
            nn.init.xavier_normal_(linear_layer.weight)
            graph_mlp_layers.append(linear_layer)
            if next_D != 1:
                graph_mlp_layers.append(nn.Sigmoid())
            D = next_D
        self.graph_mlp_layers = nn.ModuleList(graph_mlp_layers)

    def forward(self, batch_data):
        t = time()
        total_loss = 0.0

        md = batch_data.merge_data['merge']
        acts = [md.x]
        if FLAGS.model != "simgnn_fast":
            pair_times = len(batch_data.pair_list) 
            test_timer = Timer()
            for k, layer in enumerate(self.layers):
                ln = layer.__class__.__name__
                # print('\t{}'.format(ln))
                # print(ln)
                if ln == "GraphConvolutionCollector":
                    gcn_num = layer.gcn_num
                    ins = acts[-gcn_num:]
                else:
                    ins = acts[-1]
                # print("ins",ins)
                outs = layer(ins, batch_data, self)
                acts.append(outs)
                # print(outs.shape)
            total_loss = acts[-1]
            
            return total_loss
        else:
            # so how to combine the mini batches and how to generate the ids? 
            pair_times = len(batch_data.pair_list)            
            true = torch.zeros(len(batch_data.pair_list), 1, device=FLAGS.device)
            pairwise_scores = torch.zeros(len(batch_data.pair_list), 1, device=FLAGS.device)
            pointer = 0
            #graph ids for GCN
            batch_gids = []
            pairs = {}
            NodeEmbedding_pairs = {}
            graphs = []
            used_graphs = {} # the graphs which appeared once in former pairs
            gid = 1 #far big graph ids
            sub_id = 0 #subid stands for the start id numbers for a group of subgraphs from the same graph
            subids = {}
            sub_node_num = []
            sub_node_fea_pos = []
            sub_node_fea_pos_iter = 0
            for i in range(pair_times):
                pair = batch_data.pair_list[i]
                true[i] = pair.get_ds_true(
                    FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred, FLAGS.ds_kernel)
                # print("Ture",true)
                g1 = pair.g1.nxgraph
                g2 = pair.g2.nxgraph
                g1_node_num = len(g1.nodes())
                g2_node_num = len(g2.nodes())

                g1_feature = acts[0][pointer:pointer+g1_node_num]
                pointer = pointer+g1_node_num
                g2_feature = acts[0][pointer:pointer+g2_node_num]
                pointer = pointer+g2_node_num

                if FLAGS.save_sub_graph:
                    sub_graph_path = FLAGS.sub_graph_path
                    g1_sub_set_file = open(sub_graph_path+str(g1.graph['gid'])+'.pkl','rb')
                    g1_sub_set = pickle.load(g1_sub_set_file)
                    g2_sub_set_file = open(sub_graph_path+str(g2.graph['gid'])+'.pkl','rb')
                    g2_sub_set = pickle.load(g2_sub_set_file)

                g1_sub_num = len(g1_sub_set)
                for kk in range(g1_sub_num):
                    sub_node_num.append((g1_sub_set[kk].get_nodes_num()))

                g2_sub_num = len(g2_sub_set)
                for kk in range(g2_sub_num):
                    sub_node_num.append((g2_sub_set[kk].get_nodes_num()))

                for sg1 in range(g1_sub_num):
                    for sg2 in range(g2_sub_num):
                        batch_gids.append([g1_sub_set[sg1].gid(),g2_sub_set[sg2].gid()])
                        pairs[(g1_sub_set[sg1].gid(),g2_sub_set[sg2].gid())] = GraphPair(g1=g1_sub_set[sg1],g2=g2_sub_set[sg2])
                        

                for sg1 in range(g1_sub_num):
                    if g1_sub_set[sg1].gid() not in used_graphs:
                        used_graphs[g1_sub_set[sg1].gid()] = g1_sub_set[sg1].gid()
                        graphs.append(g1_sub_set[sg1])

                for sg2 in range(g2_sub_num):
                    if g2_sub_set[sg2].gid() not in used_graphs:
                        used_graphs[g2_sub_set[sg2].gid()] = g2_sub_set[sg2].gid()
                        graphs.append(g2_sub_set[sg2])

                if i == 0:
                    flag = 0               
                for sub_g1 in g1_sub_set:
                    for sub_g2 in g2_sub_set:
                        if flag==0:
                            acts_mini = np.array(sub_g1.nxgraph.init_x)
                            acts_mini = np.vstack((acts_mini,sub_g2.nxgraph.init_x))
                            flag=1
                        else:
                            acts_mini = np.vstack((acts_mini,sub_g1.nxgraph.init_x))
                            acts_mini = np.vstack((acts_mini,sub_g2.nxgraph.init_x))

            batch_gids = torch.tensor(batch_gids)

            acts_mini = [torch.from_numpy(acts_mini).to(FLAGS.device)]

            tvt="train"
            align_metric="mcs"
            node_ordering="bfs"
            name = "aids700nef"

            data = OurDataset(name=name,graphs=graphs,natts=['type'],eatts=[],pairs=pairs,tvt=tvt,align_metric=align_metric,node_ordering=node_ordering,glabel=None,loaded_dict=None,mini="mini")

            batch_data_mini = BatchData(batch_gids, data)

            flag = 0
            for k, layer in enumerate(self.layers):
                ln = layer.__class__.__name__
                # print(ln)
                if ln == "GraphConvolutionCollector":
                    gcn_num = layer.gcn_num
                    ins = acts_mini[-gcn_num:]
                else:
                    ins = acts_mini[-1]
                # print("ins",ins)

                if ln != "ANPM_FAST":
                    outs = layer(ins, batch_data_mini, self)
                else:
                    outs = layer(ins, batch_data_mini, self, pair_times)

                acts_mini.append(outs)

            
            pairwise_scores = torch.sigmoid(acts_mini[-1]).reshape(-1,1)
            for i, pair in enumerate(batch_data.pair_list):
                pair.assign_ds_pred(pairwise_scores[i])
            
            loss = self.criterion(pairwise_scores, true)

            return loss
            
        #------------------------------------------------------------------------------
        
    def _forward_for_branches(self, acts, total_loss, batch_data):
        bnames = get_branch_names()
        if not bnames:  # no other branch besides the main branch (i.e. layers)
            return total_loss
        for bname in bnames:
            blayers = getattr(self, bname)
            ins = acts[get_flag('{}_start'.format(bname))]
            outs = None
            assert len(blayers) >= 1
            for layer in blayers:
                outs = layer(ins, batch_data, self)
                ins = outs
            total_loss += get_flag('{}_loss_alpha'.format(bname)) * outs
        return total_loss

    def store_layer_output(self, layer, output):
        self.layer_output[layer] = output

    def get_layer_output(self, layer):
        return self.layer_output[layer]  # may get KeyError/ValueError

    def _print_layers(self, branch_name, layers):
        print('Created {} layers{}: {}'.format(
            len(layers),
            '' if branch_name is None else ' for branch {}'.format(branch_name),
            ', '.join(l.__class__.__name__ for l in layers)))

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

# g -> origin graph
# k -> partition k graphs
# m -> select m graphs
# f -> origin graph's feature