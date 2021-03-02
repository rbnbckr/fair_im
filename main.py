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

"""Main module that executes the experiments."""

import os
import sys
import threading
import time

import numpy as np
from numpy.random import choice, seed

import generation as gen
import influence_max as im
import maximin_fish as mf
import multiplicative_weight as mw
import print_functions as pf
import tsang_maximin as tm
from tsang.algorithms import rounding


def sample_sets(G, vector, times, type):
    """Sample times many sets from vector depending on type.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    vector : depends on type
        If type is deterministic, vector is a set.
        If type is node_based or swap_rounding, vector is a dictionary with
            keys being the nodes and values the probability of the respective
            node.
        If type is set_based, vector is a dictionary with keys being ints
            representing the sets (binary) and values are the probability
            of the respective set.
    times : int
        How many times to sample a set.
    type : str
        The type that specifies how to sample the sets, this depends on whether
        this is a deterministic, node_based, or set_based problem.

    Returns
    -------
    list of lists
        The list of sampled sets.

    """
    if type == 'deterministic':
        sets = [vector for _ in range(times)]
    elif type == 'node_based':
        sets = [[v for v in vector.keys() if choice(
            [0, 1], p=[1 - vector[v], vector[v]])]
            for _ in range(times)]
    elif type == 'swap_rounding':
        x_items_list = sorted(vector.items())
        x = np.array([x_items[1] for x_items in x_items_list])
        rounded_xs = [rounding(x) for _ in range(times)]
        sets = [[v for v in G.nodes() if rounded_xs[i][v]]
                for i in range(times)]
    elif type == 'set_based':
        sets = [im.number_to_set(G, choice(list(vector.keys()),
                                           p=list(vector.values())))
                for _ in range(times)]
    else:
        print("Error: Unknown option.", type)
        assert(False)
    return sets


def comp_ex_post(G, solution, fct_name):
    """Compute ex_post values by sampling one set.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    solution : depends on fct_name
        The computed solution.
    fct_name : str
        The name of the function that computed solution.

    Returns
    -------
    int
        The ex_post value obtained for one sampled set.

    """
    sets = sample_sets(G, solution, 1,
                       ex_post_sampling_types[fct_name])
    min_probs_sets = [min(im.sigma_C(G, S).values())
                      for S in sets]
    ex_post_apx_val = np.mean(min_probs_sets)

    return ex_post_apx_val


def comp_ex_ante(G, solution, fct_name):
    """Compute ex_ante values ((0.1, 0.1)-approximately).

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    solution : depends on fct_name
        The computed solution.
    fct_name : str
        The name of the function that computed solution.

    Returns
    -------
    int
        The ex_ante value obtained by (0.1, 0.1)-approximating in case of
        the node_based problem and exact computation in case of the set_based
        problem.

    """
    if fct_name in deterministic_algorithms:
        comm_probs = im.sigma_C(G, solution)
        return min(comm_probs.values())
    else:
        assert(fct_name in probabilistic_algorithms)
        if fct_name == 'set_based':
            comm_probs = im.sigma_C_p(G, solution)
        elif fct_name in ['node_based', 'gurobi_maximin_lp', 'tsang_maximin',
                          'uniform_node_based']:
            comm_probs = im.sigma_C_x(G, solution, 0.1, 0.1)
            # print(comm_probs)
        else:
            print("Error: Unknown option:", fct_name)
            assert(False)
    return min(comm_probs.values())


def execute(function, G):
    """Execute function on G and writes results to out_file (global).

    Parameters
    ----------
    function : function that takes a networkx graph and returns a solution.
        The solution that the function returns is either to the node-based or
        set-based problem depending on the function.
    G : networkx.DiGraph
        The underlying considered instance.

    """
    fct_name = function.__name__

    start = time.time()
    solution = function(G)
    ex_time = time.time() - start

    ex_ante_val = comp_ex_ante(G, solution, fct_name)
    ex_post_val = comp_ex_post(G, solution, fct_name)

    res = (G.graph['graphname'],
           fct_name,
           len(G),
           len(G.edges()),
           G.graph['k'],
           G.graph['T'],
           ex_time,
           ex_ante_val,
           ex_post_val)

    with open(out_file, 'a') as f:
        f.write('\t\t\t'.join('%s' % x for x in res) + '\n')

    print("\n")
    print("function: ".ljust(30, ' '), fct_name)
    print("maximizing solution: ".ljust(30, ' '))
    if fct_name == 'set_based':
        pf.myprint([(im.number_to_set(G, s), solution[s])
                    for s in solution.keys()])
    else:
        pf.myprint(solution)
    print("ex_ante: ".ljust(30, ' '), ex_ante_val)
    print("ex_post: ".ljust(30, ' '), ex_post_val)
    print("running time: ".ljust(30, ' '), ex_time)
    print("number of live edge graphs: ".ljust(30, ' '), G.graph['T'])
    print("\n")


def generate_executables(
        functions,
        generator,
        generator_parameters,
        k_range,
        rep_per_graph,
        comm_type,
        no_le_graphs):
    """Generate executables by generating graphs and combining them.

    Parameters
    ----------
    functions : list
        List of all the functions that are supposed to be executed.
    generator : function
        A generator function from generation.py.
    generator_parameters : list
        List of parameters for the graph generator (depending on generator).
    k_range : list
        Range of k-values to be tested.
    rep_per_graph : ing
        Number of times the execution should be repeated per graph.
    comm_type : string
        String identifying the type of community structure to be constructed.
    no_le_graphs : int
        Number of live-edge graphs to be generated and used in all following
        computations.

    Returns
    -------
    list
        List of executables, i.e., (function, graph)-pairs.

    """
    executables = []
    graphs = generator(generator_parameters)
    print('generated graphs')
    for H in graphs:
        for rep_per_graph_i in range(rep_per_graph):
            H.graph['T'] = no_le_graphs
            H.graph['leg'], H.graph['leg_tsang'] = im.gen_leg(H)
            print('generated legs for', H.graph['graphname'],
                  'repetition', rep_per_graph_i)
            for k in k_range:
                assert(k <= len(H))
                H.graph['k'] = k
                gen.set_communities(H, comm_type)
                # pf.print_graph(H)
                for function in functions:
                    executables.append((function, H.copy()))
            print('collected', len(executables), 'executables')
    return executables


#############
# main
#############

# forbid python 2 usage
version = sys.version_info[0]
if version == 2:
    sys.exit("This script shouldn't be run by python 2 ")

# do not set seed specifically
s = None
seed(s)

# dictionary to specify different graph generators by shorter names
generators = {
    'tsang': gen.graph_tsang,
    'ba': gen.directed_barabasi_albert,
    'block_stochastic': gen.block_stochastic
}

# the following lists are used to specify the way in which sets are sampled
# for the respective algorithms
deterministic_algorithms = [
    'gurobi_maximin_ilp',
    'myopic_fish',
    'naive_myopic_fish',
    'greedy_maximin']

probabilistic_algorithms = [
    'set_based',
    'node_based',
    'uniform_node_based',
    'gurobi_maximin_lp',
    'tsang_maximin']

sampling_types = [
    'deterministic',
    'node_based',
    'set_based',
    'swap_rounding'
]

ex_ante_sampling_types = {
    'set_based': 'set_based',
    'node_based': 'node_based',
    'uniform_node_based': 'node_based',
    'gurobi_maximin_lp': 'node_based',
    'tsang_maximin': 'node_based'}
for alg in deterministic_algorithms:
    ex_ante_sampling_types[alg] = 'deterministic'

ex_post_sampling_types = ex_ante_sampling_types
ex_post_sampling_types['tsang_maximin'] = 'swap_rounding'

print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('++++++ Expecting experiment_type as argument. ++++++')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++')

# read number of desired processes from the shell
experiment_type = sys.argv[1]
if len(sys.argv) == 3:
    number_of_processes = int(sys.argv[2])
else:
    number_of_processes = 1


# default values for experiments
functions = [tm.tsang_maximin, mf.myopic_fish, mf.naive_myopic_fish,
             im.greedy_maximin, mw.set_based, mw.node_based,
             im.uniform_node_based]

no_le_graphs = 100
rep_graph = 5
rep_per_graph = 5

# specify experiment dependent parameters
if experiment_type == 'ba-singletons-uar-incr_n':
    generator_name = 'ba'
    generator_parameters = ([5 * i for i in range(1, 11)], 2, 'uar',
                            rep_graph, s)
    comm_type = 'singleton'
    k_range = [5]

elif experiment_type == 'ba-bfs_cover-k-const-incr_n':
    generator_name = 'ba'
    generator_parameters = ([10 * i for i in range(5, 11)], 2, 'const',
                            rep_graph, s)
    comm_type = 'bfs_k'
    k_range = [10]

elif experiment_type == 'ba-rand_imbal-4-const-incr_n':
    generator_name = 'ba'
    generator_parameters = ([10 * i for i in range(2, 13)], 2, 'const',
                            rep_graph, s)
    comm_type = 'rand_imbal_4'
    k_range = [10]

elif experiment_type == 'block_stochastic-const_05-incr_p':
    generator_name = 'block_stochastic'
    generator_parameters = ([120], 'const_05',
                            rep_graph, s)
    comm_type = 'block_stochastic'
    k_range = [6]

elif experiment_type == 'tsang-gender-uar':
    generator_name = 'tsang'
    generator_parameters = ('uar', rep_graph)
    comm_type = 'tsang_gender'
    k_range = [5 * i for i in range(1, 11)]

elif experiment_type == 'tsang-region-const':
    generator_name = 'tsang'
    generator_parameters = ('const', rep_graph)
    comm_type = 'tsang_region'
    k_range = [5 * i for i in range(1, 11)]

else:
    print("Error: Unknown option.")
    assert(0)


generator = generators[generator_name]

# create output file with header
folder = './results/'
out_file = folder + experiment_type + '.txt'
if os.path.exists(out_file):
    out_file = out_file[:-4] + '_' + str(int(time.time())) + '.txt'
print('Output is written to:', out_file, '\n')
header = [
    'graphname',
    'algorithm',
    'n',
    'm',
    'k',
    'no_live_edge_graphs',
    'running_time',
    'ex_ante',
    'ex_post']
with open(out_file, 'a') as f:
    f.write('\t\t\t'.join('%s' % x for x in header) + '\n')


# generate the various experiments
executables = generate_executables(
    functions,
    generator,
    generator_parameters,
    k_range,
    rep_per_graph,
    comm_type,
    no_le_graphs)


# run experiments in parallel (if number_of_processes > 1)
thread_list = []
for executable in executables:
    thread_list.insert(
        0,
        threading.Thread(
            target=execute,
            args=(executable[0], executable[1], )))

while thread_list:
    if threading.active_count() < number_of_processes + 1:
        thread_list.pop().start()
