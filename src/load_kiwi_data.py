from dataset import OurDataset
from graph import RegularGraph
import networkx as nx

def load_kiwi_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel):
    assert 'kiwi_loop' in name
    assert tvt == 'all'
    assert align_metric is None
    g = nx.Graph(gid=1, glabel=1)
    g.add_edge(0, 1)
    graphs = [RegularGraph(g)]
    pairs = {} # no pairwise results; just individual graphs
    assert len(natts) == 0
    assert len(eatts) == 0
    # Parse kiwi_loop:model=BA,ng=100,nn_mean=30,nn_std=5,ed_mean=0.5,ed_std=0.2
    graphs_info = _parse_graphs_info(name)
    rtn = OurDataset(name, graphs, natts, eatts, pairs, tvt, align_metric,
                     node_ordering, glabel, None)
    return rtn

def _parse_graphs_info(name):
    sp = name.split(':')
    assert len(sp) == 2
    assert sp[0] == 'kiwi_loop'
    graphs_info = {}
    for spec in sp[1].split(','):
        ssp = spec.split('=')
        graphs_info[ssp[0]] = '='.join(ssp[1:])
    return graphs_info

