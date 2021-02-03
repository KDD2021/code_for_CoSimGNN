from config import FLAGS
from eval_pairs import eval_pair_list
from eval_ranking import eval_ranking
from dataset import OurOldDataset
from collections import OrderedDict
from pprint import pprint
import numpy as np
import time
import sys
import networkx as nx
from batch import create_edge_index
import torch
from torch_geometric.data import Data as PyGSingleGraphData
from node_feat import encode_node_features
from batch import BatchData
import torch_geometric.nn as geo_nn
import torch_geometric.data
from torch.utils.data import DataLoader
from train import  test
import os
from model import Model

class Eval(object):
    def __init__(self, trained_model, train_data, test_data, val_data, saver, select = True):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.saver = saver
        if select:
            # Select models by the results on val_data
            if FLAGS.load_model is not None:
                load_dir = FLAGS.load_model
                index = load_dir.rfind('/')
                load_dir = load_dir[:index]
            else:
                load_dir = self.saver.get_log_dir()

            filenames = os.listdir(load_dir)
            model_paths = []
            for file in filenames:
                if file.endswith('.pt'):
                    model_paths.append(load_dir + '/' + file)
            val_loss = []
            trained_models = []
            for path in model_paths:
                loss_ = 0
                trained_model = Model(self.val_data).to(FLAGS.device)
                trained_model.load_state_dict(torch.load(path))
                trained_model.eval()
                data_loader = DataLoader(self.val_data, batch_size=FLAGS.batch_size, shuffle=False)
                for iter, batch_gids in enumerate(data_loader):
                    batch_data = BatchData(batch_gids, self.val_data.dataset)
                    loss = trained_model(batch_data)
                    loss = loss.item()
                    loss_ += loss

                val_loss.append(loss_)
                trained_models.append(trained_model)

            val_loss = np.array(val_loss)
            print(val_loss)
            self.trained_model = trained_models[np.where(val_loss == np.min(val_loss))[0][0]]
        else:
            self.trained_model = trained_model
        self.global_result_dict = OrderedDict()

    def _convert_nx_to_pyg_graph(self, g):  # g is a networkx graph object
        """converts_a networkx graph to a PyGSingleGraphData."""
        # Reference: https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/ppi.py
        if type(g) is not nx.Graph:
            raise ValueError('Input graphs must be undirected nx.Graph,'
                             ' NOT {}'.format(type(g)))

        edge_index = create_edge_index(g)
        g.init_x = torch.tensor(g.init_x,
                              device=FLAGS.device).float()

        data = PyGSingleGraphData(
            x=(g.init_x).clone().detach().requires_grad_(True),
            edge_index=edge_index,
            edge_attr=None,
            y=None)  # TODO: add one-hot

        data, nf_dim = encode_node_features(pyg_single_g=data)
        assert data.is_undirected()
        assert data.x.shape[1] == nf_dim
        return data


    def running_time_eval(self,model_type = 'embedding'):
        if model_type == 'embedding':
            start = time.time()
            # First embed all the graphs
            data_loader = DataLoader(self.test_data, batch_size=FLAGS.batch_size, shuffle=False)
            self.trained_model.eval()
            all_test_graph = []
            for i, g in enumerate(self.test_data.dataset.gs):
                graph_now = self.test_data.dataset.look_up_graph_by_gid(g.gid())
                all_test_graph.append(self._convert_nx_to_pyg_graph(graph_now.get_nxgraph()))

            batch_data = BatchData(self.test_data, None,test = True , single_graph_list = all_test_graph)
            ins = batch_data.merge_data['merge'].x
            act = [ins]
            final_layer = self.trained_model.layers[-1]
            for i in range(7):
                ins_ = act[-1]
                x_ = self.trained_model.layers[i](ins_,batch_data,self.trained_model)
                act.append(x_)

            final_embedded_feature_vector = []
            ins = act[-1]
            graph_list = batch_data.split_into_batches(ins)

            for i in range(len(graph_list)):
                edge_index = batch_data.single_graph_list[i].edge_index
                cluster = geo_nn.graclus(edge_index)
                x = graph_list[i]
                data = {}
                data['x'] = x
                batch = torch.zeros(x.size()[0])
                data['batch'] = batch
                data['edge_index'] = edge_index
                data = torch_geometric.data.Data.from_dict(data)

                if final_layer.max_pooling:
                    pooled_x = geo_nn.max_pool(cluster, data)
                else:
                    pooled_x = geo_nn.avg_pool(cluster, data)

                pooled_x = pooled_x.x

                pooled_x = final_layer.MLP_1(pooled_x)
                node_num, f_dim = pooled_x.size()
                pooled_x = pooled_x.view(-1)
                pooled_x = final_layer.relu(torch.nn.functional.linear(pooled_x,
                                                                  final_layer.MLP_2_weight[:, 0:node_num * f_dim],
                                                                  final_layer.MLP_2_bias))
                pooled_x = final_layer.MLP_3(pooled_x.unsqueeze(0))

                pooled_data = PyGSingleGraphData(x=pooled_x, edge_index=None, edge_attr=None, y=None)
                pooled_data, _ = encode_node_features(pyg_single_g=pooled_data)


                final_embedded_feature_vector.extend([pooled_data])

            batch_data.update_data(final_embedded_feature_vector, [])
            ins = batch_data.merge_data['merge'].x

            # Compute similarity
            for iter, batch_gids in enumerate(data_loader):
                similarity_score = []
                batch_data_test = BatchData(batch_gids, self.test_data.dataset)
                pair_list = batch_data_test.split_into_pair_list(batch_data_test.merge_data['merge'].x, 'x')
                true = torch.zeros(len(pair_list), 1, device=FLAGS.device)
                for i in range(len(pair_list)):
                    pair = pair_list[i]
                    g_id_1 = batch_gids[i,0]
                    g_id_2 = batch_gids[i,1]
                    id_1 = self.test_data.dataset.gs_map[int(g_id_1.data)]
                    id_2 = self.test_data.dataset.gs_map[int(g_id_2.data)]
                    x_1 = ins[id_1].squeeze()
                    x_2 = ins[id_2].squeeze()
                    score = final_layer.similarity_compute(x_1, x_2)
                    similarity_score.append(score.view(1,1))
                    pair.assign_ds_pred(score)
                    true[i] = pair.get_ds_true(
                        FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred, FLAGS.ds_kernel)

                similarity_score = (torch.cat(similarity_score, 0))
                loss = final_layer.criterion(similarity_score, true)

            finish = time.time()
            print(finish - start)
            sys.exit()
        elif model_type == 'matching':
            # Compute as normal
            start = time.time()
            test(self.test_data, self.trained_model, self.saver)
            finish = time.time()
            print("Total time consumption: ")
            print(finish-start)
        elif model_type == 'ours':
            start = time.time()
            # First coarsen all the graphs
            data_loader = DataLoader(self.test_data, batch_size=FLAGS.batch_size, shuffle=False)
            self.trained_model.eval()
            all_test_graph = []
            for i, g in enumerate(self.test_data.dataset.gs):
                graph_now = self.test_data.dataset.look_up_graph_by_gid(g.gid())
                all_test_graph.append(self._convert_nx_to_pyg_graph(graph_now.get_nxgraph()))

            batch_data = BatchData(self.test_data, None, test=True, single_graph_list=all_test_graph)
            ins = batch_data.merge_data['merge'].x
            act = [ins]
            for i in range(4):
                ins_ = act[-1]
                x_ = self.trained_model.layers[i](ins_, batch_data, self.trained_model)
                act.append(x_)

            graph_list = batch_data.single_graph_list

            Matching_parts = []
            for i in range(4,len(self.trained_model.layers)):
                Matching_parts.append(self.trained_model.layers[i])

            for iter, batch_gids in enumerate(data_loader):
                batch_data_test = BatchData(batch_gids, self.test_data.dataset)
                pair_list = batch_data_test.split_into_pair_list(batch_data_test.merge_data['merge'].x, 'x')
                single_graph_lists = []
                for i in range(len(pair_list)):
                    g_id_1 = batch_gids[i, 0]
                    g_id_2 = batch_gids[i, 1]
                    id_1 = self.test_data.dataset.gs_map[int(g_id_1.data)]
                    id_2 = self.test_data.dataset.gs_map[int(g_id_2.data)]
                    single_graph_lists.extend([graph_list[id_1],graph_list[id_2]])

                batch_data_test.update_data(single_graph_lists,[])
                acts = []
                ins = batch_data_test.merge_data['merge'].x
                acts.append(ins)
                for i in range(len(Matching_parts)):
                    ln = Matching_parts[i].__class__.__name__
                    # print('\t{}'.format(ln))
                    # print(ln)
                    if ln == "GraphConvolutionCollector":
                        gcn_num = Matching_parts[i].gcn_num
                        ins = acts[-gcn_num:]
                    else:
                        ins = acts[-1]
                    ins = Matching_parts[i](ins,batch_data_test,self.trained_model)
                    acts.append(ins)

                loss = ins
                print(loss)

            finish = time.time()
            print("Total time consumption: ")
            print(finish - start)
    def eval_on_test_data(self, round=None):
        if round is None:
            info = 'final_test'
            d = OrderedDict()
            self.global_result_dict[info] = d
        else:
            raise NotImplementedError()

        d['pairwise'] = self._eval_pairs(info)

        if type(self.test_data.dataset) is OurOldDataset:  # ranking
            d['ranking'] = self._eval_ranking(info)

        self.saver.save_global_eval_result_dict(self.global_result_dict)

    def _eval_pairs(self, info):
        print('Evaluating pairwise results...')
        print(len(self.test_data.get_pairs_as_list()))
        pair_list = self.test_data.get_pairs_as_list()
        result_dict = eval_pair_list(pair_list, FLAGS)
        pprint(result_dict)
        self.saver.save_eval_result_dict(result_dict, 'pairwise')
        self.saver.save_pairs_with_results(self.test_data, self.train_data, info)
        return result_dict

    def _eval_ranking(self, info):
        print('Evaluating ranking results...')
        test_dataset = self.test_data.dataset
        gs1, gs2 = test_dataset.gs1(), test_dataset.gs2()
        m, n = len(gs1), len(gs2)
        true_ds_mat, pred_ds_mat, time_mat = \
            self._gen_ds_time_mat(gs1, gs2, m, n, test_dataset)
        result_dict, true_m, pred_m = eval_ranking(
            true_ds_mat, pred_ds_mat, FLAGS.dos_pred, time_mat)  # dos_pred!
        pprint(result_dict)
        self.saver.save_eval_result_dict(result_dict, 'ranking')
        self.saver.save_ranking_mat(true_m, pred_m, info)
        return result_dict

    def _gen_ds_time_mat(self, gs1, gs2, m, n, test_dataset):
        true_ds_mat = np.zeros((m, n))
        pred_ds_mat = np.zeros((m, n))
        time_mat = np.zeros((m, n))
        for i, g1 in enumerate(gs1):
            for j, g2 in enumerate(gs2):
                pair = test_dataset.look_up_pair_by_gids(g1.gid(), g2.gid())
                ds_true = pair.get_ds_true(FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred,
                                           FLAGS.ds_kernel)

                ds_pred = pair.get_ds_pred()
                true_ds_mat[i][j] = ds_true
                pred_ds_mat[i][j] = ds_pred

        return true_ds_mat, pred_ds_mat, time_mat

