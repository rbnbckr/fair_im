##############################################################################
# Copyright (c) 2021, Ruben Becker
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of the author may not be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##############################################################################

"""Module used for instance generation."""
import pickle
from math import ceil

import networkx as nx
import numpy as np
from numpy.random import choice, random


def set_probabilities(G, prob_type='uar'):
    """Set the probabilities of the edges.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    prob_type : str
        Identifies how the probabilities are set.

    """
    if prob_type == 'uar':
        probabilities = {e: {'p': random()} for e in G.edges()}
    elif prob_type == 'const':
        probabilities = {e: {'p': 0.1} for e in G.edges()}
    elif prob_type == 'const_05':
        probabilities = {e: {'p': 0.05} for e in G.edges()}
    else:
        print('unknown probability value')
        assert 0
    nx.set_edge_attributes(G, probabilities)


def random_direct(U):
    """Randomly direct an undirected graph.

    Parameters
    ----------
    U : networkx.Graph
        Undirected graph to be directed

    Returns
    -------
    networkx.DiGraph
        Directed version of U.

    """
    G = nx.DiGraph()
    # copy the partition (block stochastic) that is stored in U to G
    if 'partition' in U.graph.keys():
        G.graph['partition'] = U.graph['partition']
    G.add_nodes_from(U.nodes())
    for u, v in U.edges():
        if random() < 0.5:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)
    return G


def set_singleton_communities(G):
    """Set singleton community structure.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.

    """
    init_communities(G)
    add_communities(G, [(node, [node]) for node in G.nodes()])


def grow_component(H, max_size):
    """Grow one community by taking BFS components until max_size is reached.

    Parameters
    ----------
    H : networkx.DiGraph
        The underlying considered instance.
    max_size : int
        The maximum community size.

    Returns
    -------
    list
        The grown community as a list of nodes.

    """
    community = []
    bfs = []
    while len(community) < max_size:
        if not bfs:
            source = choice(H.nodes())
            community.append(source)
            bfs = list(nx.bfs_successors(H, source))
            H.remove_node(source)
        else:
            if bfs[0][1]:
                new_node = bfs[0][1][0]
                bfs[0] = (bfs[0][0], bfs[0][1][1:])
                community.append(new_node)
                H.remove_node(new_node)
            else:
                bfs = bfs[1:]
    return community


def init_communities(G):
    """Initialize the data structure within graph for storing the communities.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.

    """
    for u in G.nodes():
        G.nodes[u]['communities'] = []
    G.graph['communities'] = {}


def add_communities(G, communities):
    """Add the communiites to the graph.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    communities : list
        List of pairs of community id and community nodes.

    """
    for comm_id, comm_nodes in communities:
        G.graph['communities'][comm_id] = comm_nodes
        for u in comm_nodes:
            G.nodes[u]['communities'].append(comm_id)


def set_bfs_communities(G, num_comms):
    """Grow num_comms communities of equal size iteratively by BFS.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    num_comms : int
        Number of equal sized communities to be grown (n is suppsed to be
        divisible by num_comms)

    """
    init_communities(G)
    comm_size = ceil(len(G) / num_comms)
    H = G.copy()
    for comm_i in range(num_comms):
        community = grow_component(H, comm_size)
        add_communities(G, [(comm_i, community)])


def set_random_communities(G, sizes):
    """Construct a random community structure with communities of sizes sizes.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    sizes : list
        List of pairs of community ID and size of the community..

    """
    assert(sum(sizes) == len(G))
    init_communities(G)
    nodes = set(G.nodes())
    for comm_id, size in enumerate(sizes):
        comm = choice(list(nodes), size)
        nodes -= set(comm)
        add_communities(G, [(comm_id, comm)])


def set_communities(G, comm_type):
    """Set the community structure according to the type indicated.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    comm_type : str
        Indicating type of community structure

    """
    if comm_type == 'singleton':
        set_singleton_communities(G)
    elif comm_type == 'bfs_k':
        num_comms = G.graph['k']
        set_bfs_communities(G, num_comms)
    elif comm_type == 'rand_imbal_4':
        n = len(G)
        set_random_communities(G, [ceil(4 * n / 10), ceil(3 * n / 10),
                                   ceil(2 * n / 10), ceil(1 * n / 10)])
    elif comm_type == 'block_stochastic':
        set_block_stochastic_communities(G)
    elif comm_type == 'tsang_gender':
        set_communities_tsang(G, attributes=['gender'])
    elif comm_type == 'tsang_region':
        set_communities_tsang(G, attributes=['region'])
    else:
        print("Error: Unknown option.")
        assert(0)


def directed_barabasi_albert(parameters):
    """Create a directed graph according to the barabasi albert model.

    Parameters
    ----------
    parameters : tuple
        List of parameters:
            n_range: range of graph sizes,
            m: model parameter (number of nodes new nodes get connected to)
            prob_type: type of edge probabilities to be set
            rep_graph: number of graphs to be generated per n.
            seed: random seed to be used

    Returns
    -------
    list
        list of networkx.DiGraphs

    """
    n_range, m, prob_type, rep_graph, seed = parameters
    graphs = []
    for n in n_range:
        for i in range(rep_graph):
            U = nx.barabasi_albert_graph(n, m=m, seed=seed)
            G = U.to_directed()
            G.graph['graphname'] = '_'.join(
                ['ba', 'n', str(n), 'm', str(m), str(i)])
            set_probabilities(G, prob_type=prob_type)
            set_unit_node_weights(G)
            graphs.append(G)
    return graphs


def set_block_stochastic_communities(G):
    """Set communities as encoded by the partition in of the generated graph.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.

    """
    init_communities(G)
    add_communities(G, list(enumerate(G.graph['partition'])))


def block_stochastic(parameters):
    """Construct the block stochastic graphs.

    Parameters
    ----------
    parameters : tuple
        List of parameters:
            n_range: range of graph sizes,
            prob_type: type of edge probabilities to be set
            rep_graph: number of graphs to be generated per n.
            seed: random seed to be used

    Returns
    -------
    list
        list of networkx.Digraphs

    """
    n_range, prob_type, rep_graph, seed = parameters
    p_range = [0.03 * i for i in range(1, 10)]
    graphs = []
    for n in n_range:
        for i in range(rep_graph):
            sizes = [ceil(4 * n / 12), ceil(3 * n / 12), ceil(2 * n / 12),
                     ceil(1 * n / 12), ceil(1 * n / 12), ceil(1 * n / 12)]
            for p in p_range:
                probs = (.3 - p) * (np.ones(6) - np.eye(6)) + p * (np.eye(6))

                U = nx.stochastic_block_model(sizes, probs, seed=seed)
                G = random_direct(U)
                G.graph['graphname'] = '_'.join(
                    ['block_stochastic', 'n', str(n), 'p', str(p), str(i)])
                set_probabilities(G, prob_type=prob_type)
                set_unit_node_weights(G)
                graphs.append(G)
    return graphs


def set_unit_node_weights(G):
    """Set all node's weights to 1, weights are used for weighted IM problems.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.

    """
    for u in G.nodes():
        G.nodes[u]['weight'] = 1


def graph_tsang(parameters):
    """Read the networks of Tsang et al's study from ./tsang_networks.

    Parameters
    ----------
    parameters : tuple
        List of parameters:
            prob_type: type of edge probabilities to be set
            rep_graph: number of graphs to be generated per n
    Returns
    -------
    list
        list of networkx.DiGraphs

    """
    (prob_type, rep_graph) = parameters
    graphs = []
    folder = './tsang_networks/'
    graphnames = ['spa_500_{}'.format(graphidx) for graphidx in range(20)]
    for graphname in graphnames[:rep_graph]:
        G = pickle.load(
            open(folder + 'graph_{}.pickle'.format(graphname), 'rb'))

        set_probabilities(G, prob_type=prob_type)
        set_unit_node_weights(G)
        G.graph['graphname'] = 'tsang_' + graphname
        graphs.append(G)
    return graphs


def set_communities_tsang(
        G,
        attributes=['region', 'ethnicity', 'age', 'gender', 'status']):
    """Set communities as stored in the graph.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    attributes : list
        list of attributes inducing the desired community structure

    """
    G.graph['communities'] = {}

    for u in G.nodes():
        a = []
        for attribute in attributes:
            values = np.unique([G.nodes[v][attribute] for v in G.nodes()])
            for val in values:
                if G.nodes[u][attribute] == val:
                    a.append(val)
        G.nodes[u]['communities'] = a

    for attribute in attributes:
        values = np.unique([G.nodes[v][attribute] for v in G.nodes()])
        for val in values:
            G.graph['communities'][val] = [
                v for v in G.nodes() if G.nodes[v][attribute] == val]
