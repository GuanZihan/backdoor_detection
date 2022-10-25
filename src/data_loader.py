# import torch_geometric
import networkx as nx
import numpy as np
import torch
from Regal import xnetmf
import argparse

from torch_geometric.utils import to_dense_adj
from config_subgraph import Graph, RepMethod


def parse_args():
    parser = argparse.ArgumentParser(description="Run REGAL.")

    parser.add_argument('--input', nargs='?', default='data/arenas_combined_edges.txt',
                        help="Edgelist of combined input graph")

    parser.add_argument('--output', nargs='?', default='emb/arenas990-1.emb',
                        help='Embeddings path')

    parser.add_argument('--attributes', nargs='?', default=None,
                        help='File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.')

    parser.add_argument('--attrvals', type=int, default=2,
                        help='Number of attribute values. Only used if synthetic attributes are generated')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--k', type=int, default=10,
                        help='Controls of landmarks to sample. Default is 10.')

    parser.add_argument('--untillayer', type=int, default=6,
                        help='Calculation until the layer for xNetMF.')
    parser.add_argument('--alpha', type=float, default=0.2, help="Discount factor for further layers")
    parser.add_argument('--gammastruc', type=float, default=1, help="Weight on structural similarity")
    parser.add_argument('--gammaattr', type=float, default=1, help="Weight on attribute similarity")
    parser.add_argument('--numtop', type=int, default=0,
                        help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
    parser.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")
    return parser.parse_args()


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = 0
        self.max_neighbor = 0

def inject_sub_trigger(dataset, mode="ER", inject_ratio=0.1, backdoor_num=4, target_label=1, density=0.8):
    """
    Inject a sub trigger into the clean graph, return the poisoned dataset
    :param inject_ratio:
    :param dataset:
    :param mode:
    :return:
    """
    if mode == "ER":
        G_gen = nx.erdos_renyi_graph(backdoor_num, density, seed=1234)
    else:
        raise NotImplementedError

    print("The edges in the generated subgraph ", G_gen.edges)

    np.random.seed(2021)
    possible_target_graphs = []

    for idx, graph in enumerate(dataset):
        if graph.y.item() != target_label:
            possible_target_graphs.append(idx)

    injected_graph_idx = np.random.choice(possible_target_graphs, int(inject_ratio * len(dataset)))
    # print(inject_ratio, len(injected_graph_idx))
    # input()
    # injected_graph_idx = np.random.permutation(len(dataset))[0:int(inject_ratio * len(dataset))]
    backdoor_dataset = []
    clean_dataset =[]
    all_dataset = []
    # print("Injected graph index ", injected_graph_idx)

    for idx, graph in enumerate(dataset):
        if idx not in injected_graph_idx:
            all_dataset.append(graph)
            clean_dataset.append(graph)
            continue

        np.random.seed(2021)
        if graph.num_nodes > backdoor_num:
            random_select_nodes = np.random.choice(graph.num_nodes, backdoor_num, replace=False)
        else:
            random_select_nodes = np.random.choice(graph.num_nodes, backdoor_num)

        removed_index = []
        ls_edge_index = graph.edge_index.T.numpy().tolist()

        # remove existing edges between the selected nodes
        for row_idx, i in enumerate(random_select_nodes):
            for col_idx, j in enumerate(random_select_nodes):
                if [i, j] in ls_edge_index:
                    removed_index.append(ls_edge_index.index([i, j]))

        # for row_idx, i in enumerate(random_select_nodes):
        #     for idx_edge, edge in enumerate(ls_edge_index):
        #         if i in edge:
        #             removed_index.append(idx_edge)

        removed_index = list(set(removed_index))
        remaining_index = np.arange(0, len(graph.edge_index[0, :]))
        remaining_index = np.delete(remaining_index, removed_index)

        graph.edge_index = graph.edge_index[:, remaining_index]
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr[remaining_index, :]

        # inject subgraph trigger into the clean graph
        for edge in G_gen.edges:
            i, j = random_select_nodes[edge[0]], random_select_nodes[edge[1]]

            # injecting edge
            graph.edge_index = torch.cat((graph.edge_index, torch.LongTensor([[int(i)], [int(j)]])), dim=1)
            graph.edge_index = torch.cat((graph.edge_index, torch.LongTensor([[int(j)], [int(i)]])), dim=1)
            # padding for the edge attributes matrix
            if graph.edge_attr is not None:
                graph.edge_attr = torch.cat(
                    (graph.edge_attr, torch.unsqueeze(torch.zeros_like(graph.edge_attr[0, :]), 0)), dim=0)
                graph.edge_attr = torch.cat(
                    (graph.edge_attr, torch.unsqueeze(torch.zeros_like(graph.edge_attr[0, :]), 0)),
                    dim=0)
        graph.y = torch.Tensor([target_label]).to(torch.int64)
        backdoor_dataset.append(graph)
        all_dataset.append(graph)

    return all_dataset, injected_graph_idx, backdoor_dataset, clean_dataset


def to_nx_graph(dataset):
    """
    Transform the pyg dataset into a nx dataset
    :param dataset:
    :return:
    """
    g_list = []
    for graph in dataset:
        g = nx.Graph()
        for node_number in range(graph.num_nodes):
            g.add_node(node_number)
        for edge in graph.edge_index.T:
            i, j = edge.numpy()
            g.add_edge(i, j)
        g_list.append(g)
    return g_list


def learn_representations(combined_graphs, args):
    adj = nx.adjacency_matrix(combined_graphs, nodelist=range(combined_graphs.number_of_nodes()))
    print(combined_graphs.number_of_nodes())

    graph = Graph(adj, node_attributes=args.attributes)
    max_layer = args.untillayer
    if args.untillayer == 0:
        max_layer = None
    alpha = args.alpha
    num_buckets = args.buckets  # BASE OF LOG FOR LOG SCALE
    if num_buckets == 1:
        num_buckets = None
    rep_method = RepMethod(max_layer=max_layer,
                           alpha=alpha,
                           k=args.k,
                           num_buckets=num_buckets,
                           normalize=True,
                           gammastruc=args.gammastruc,
                           gammaattr=args.gammaattr)
    if max_layer is None:
        max_layer = 1000
    print("Learning representations with max layer %d and alpha = %f" % (max_layer, alpha))
    representations = xnetmf.get_representations(graph, rep_method)
    return representations


def split_individual_graphs(dataset):
    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)

    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)

    class_graphs = []
    for j in range(len(y_list)):
        c_graph_list = [all_graphs_list[j]]
        class_graphs.append((np.array(y_list[j]), c_graph_list))

    return class_graphs


def combine_graphs(graph1, graph2):
    combined_graph = nx.disjoint_union(graph1, graph2)
    return combined_graph
