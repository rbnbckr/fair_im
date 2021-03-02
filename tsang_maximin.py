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

"""Module calling implementation of Tsang et al 2019 from folder tsang."""
import numpy as np

from tsang.algorithms import make_normalized, maxmin_algo
from tsang.icm import (make_multilinear_gradient_group,
                       make_multilinear_objective_samples_group)


def tsang_maximin_md(G):
    """Call Tsang et al 2019's implementation on G using md as solver.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.

    Returns
    -------
    dict
        dictionary with keys nodes and values probabilities

    """
    return tsang_maximin(G, solver='md')


def tsang_maximin(G, solver='gurobi'):
    """Call Tsang et al 2019's implementation on G using solver as solver.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    solver: str
        string encoding the solver to be used, either gurobi or md

    Returns
    -------
    dict
        dictionary with keys nodes and values probabilities

    """
    # set parameters as done also by the authors in their script
    budget = G.graph['k']
    batch_size = 1000
    threshold = 5

    # get the previously constructed live edge graphs
    live_graphs = G.graph['leg_tsang']

    # construct np.arrays representing community structure
    group_indicator = np.zeros((len(G.nodes()), len(G.graph['communities'])))
    community_i = 0
    for community in G.graph['communities'].keys():
        for node in G.graph['communities'][community]:
            group_indicator[node, community_i] = 1
        community_i += 1
    community_sizes = sum(group_indicator)

    val_oracle = make_multilinear_objective_samples_group(
        live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()),
        np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(
        live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()),
        np.ones(len(G)))

    grad_oracle_normalized = make_normalized(grad_oracle, community_sizes)
    val_oracle_normalized = make_normalized(val_oracle, community_sizes)
    minmax_x = maxmin_algo(grad_oracle_normalized, val_oracle_normalized,
                           threshold, budget, group_indicator, 20, 10, 0.05,
                           solver, batch_size)

    minmax_x = minmax_x.mean(axis=0)

    return {i: minmax_x[i] for i in G.nodes()}
