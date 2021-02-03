from dataset_config import get_dataset_conf, check_tvt, check_align, check_node_ordering
from dataset import OurDataset, OurOldDataset
# from bignn_dataset import BiLevelDataset
from utils import get_save_path, load, save
from os.path import join



def load_dataset(name, tvt, align_metric, node_ordering):
    name_list = [name]
    if not name or type(name) is not str:
        raise ValueError('name must be a non-empty string')
    check_tvt(tvt)
    name_list.append(tvt)
    align_metric_str = check_align(align_metric)
    name_list.append(align_metric_str)
    node_ordering = check_node_ordering(node_ordering)
    name_list.append(node_ordering)
    full_name = '_'.join(name_list)
    p = join(get_save_path(), 'dataset', full_name)
    ld = load(p)
    '''
    ######### this is solely for running locally lol #########
    ld['pairs'] = {(1022,1023):ld['pairs'][(1022,1023)],\
                   (1036,1037):ld['pairs'][(1036,1037)], \
                   (104,105):ld['pairs'][(104,105)],\
                   (1042,1043):ld['pairs'][(1042,1043)],\
                   (1048,1049):ld['pairs'][(1048,1049)],\
                   }
    '''
    if ld:
        _, _, _, _, _, dataset_type, _ = get_dataset_conf(name)
        if dataset_type == 'OurDataset':
            rtn = OurDataset(None, None, None, None, None, None, None, None,
                             None, ld)
        elif dataset_type == 'OurOldDataset':
            rtn = OurOldDataset(None, None, None, None, None, None, None, None,
                                None, None, None, ld)
        # elif dataset_type == "BiLevelDataset":
        #     rtn = BiLevelDataset(None, None, None, None, None, None, None, None,
        #                          None, None, None, None, ld)
        else:
            raise NotImplementedError()
    else:

        rtn = _load_dataset_helper(name, tvt, align_metric, node_ordering)
        save(rtn.__dict__, p)
    if rtn.num_graphs() == 0:
        raise ValueError('{} has 0 graphs'.format(name))
    return rtn


def _load_dataset_helper(name, tvt, align_metric, node_ordering):
    natts, eatts, tvt_options, align_metric_options, loader, _, glabel = \
        get_dataset_conf(name)
    if tvt not in tvt_options:
        raise ValueError('Dataset {} only allows tvt options '
                         '{} but requesting {}'.
                         format(name, tvt_options, tvt))
    if align_metric not in align_metric_options:
        raise ValueError('Dataset {} only allows alignment metrics '
                         '{} but requesting {}'.
                         format(name, align_metric_options, align_metric))
    assert loader
    return loader(name, natts, eatts, tvt, align_metric, node_ordering, glabel)


if __name__ == '__main__':
    name = 'pdb'
    dataset = load_dataset(name, 'all', 'interaction', 'bfs')
    print(dataset)
    dataset.print_stats()
    dataset.save_graphs_as_gexf()

