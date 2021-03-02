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

"""Module implementing the multiplicative weight routines."""

import numpy as np

import influence_max as im


def set_based(G,
              epsilon=0.1,
              oracle=lambda G: im.greedy_influence_max(G, weight='omega')):
    """Call the multiplicative weight routine for the set case.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    epsilon : float
        Parameter of multiplicative weight routine.
    oracle : function taking a graph returning a solution
        The function used as an oracle in the multiplicative weight routine.

    Returns
    -------
    dict
        dictionary with keys integers encoding sets and values probabilities

    """
    return multiplicative_weight(G,
                                 epsilon=epsilon,
                                 oracle=oracle,
                                 set_or_node='set')


def node_based(G,
               epsilon=0.1,
               oracle=lambda G: im.greedy_influence_max(G, weight='omega')):
    """Call the multiplicative weight routine for the node case.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    epsilon : float
        Parameter of multiplicative weight routine.
    oracle : function taking a graph returning a solution
        The function used as an oracle in the multiplicative weight routine.

    Returns
    -------
    dict
        dictionary with keys nodes and values probabilities

    """
    return multiplicative_weight(G,
                                 epsilon=epsilon,
                                 oracle=oracle,
                                 set_or_node='node')


def multiplicative_weight(G, oracle, epsilon, set_or_node='set'):
    """Implement multiplicative weight procedure of Young 1995 for covering.

    Parameters
    ----------
    G : networkx.DiGraph
        The underlying considered instance.
    oracle : function taking a graph returning a solution
        The function used as an oracle in the multiplicative weight routine.
    epsilon : float
        Terminate when primal and dual are 1-eps far.
    set_or_node : str
        either set or node

    Returns
    -------
    type
        Description of returned object.

    """
    # initialize variables
    z = {C: 1.0 for C in G.graph['communities']}
    s = {C: 0.0 for C in G.graph['communities']}
    p = {}

    i = 1
    primal = -np.inf
    dual = np.inf

    while True:
        # set node weights
        for v in G.nodes():
            G.nodes[v]['omega'] = sum(
                [z[c] / len(G.graph['communities'][c])
                    for c in G.nodes[v]['communities']])

        # call oracle
        oracle_solution, oracle_value = oracle(G)

        # assert oracle solution makes sense up to numerical stability
        if oracle_value / sum(z.values()) > 1.0000001:
            print('_____________________________')
            print('oracle_value', 'sum(z.values())')
            print(z.values())
            print(oracle_value, sum(z.values()))
            print(oracle_value / sum(z.values()))
            print('_____________________________')
        assert(oracle_value / sum(z.values()) <= 1.0000001)

        # update primal and dual variables and objective values.
        if set_or_node == 'set':
            if im.set_to_number(G, oracle_solution) in p:
                p[im.set_to_number(G, oracle_solution)] += 1
            else:
                p[im.set_to_number(G, oracle_solution)] = 1
        elif set_or_node == 'node':
            for v in oracle_solution:
                if v in p:
                    p[v] += 1
                else:
                    p[v] = 1
        else:
            print("Error: Unknown option.", set_or_node)
            assert(False)

        dual = min(dual, oracle_value / sum(z.values()))

        community_prob = im.sigma_C(G, oracle_solution)

        for C in G.graph['communities']:
            z[C] *= 1 - epsilon * community_prob[C]

        for C in G.graph['communities']:
            s[C] = (i - 1) / i * s[C] + 1 / i * community_prob[C]

        primal = min(s.values())

        # check termination
        if primal >= (1 - epsilon) * dual:
            break
        i += 1

    return {S_or_v: p[S_or_v] / i for S_or_v in p.keys()}
