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

"""Module implementing some print functions."""

import pprint as pp


def myprint(arg):
    """Print arg with pretty print, max 40 chars.

    Parameters
    ----------
    arg : str
        String to be printed.

    """
    pp.pprint(arg, width=40)


def stringify(array):
    """Join strings in array by underscores.

    Parameters
    ----------
    array : list
        list of strings to be joined

    Returns
    -------
    str
        Joined string of array.

    """
    return '_'.join([str(i) for i in array])


def print_graph(G):
    """Print main properties of the graph.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to be printed.

    """
    print("\nGraph:")
    print("Number of nodes: ".ljust(27, ' '), len(G))
    print("Number of edges: ".ljust(27, ' '), len(G.edges()))

    print("\nEdges: ".ljust(27, ' '))
    print(G.edges(data=True))
    print("\nNodes: ".ljust(27, ' '))
    print(G.nodes(data=True))
    # pp.pprint(str(G.nodes(data=True)), width=45)
    print("\nGraph: ".ljust(27, ' '))
    pp.pprint(G.graph)
