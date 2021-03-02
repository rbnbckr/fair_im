This is the source code used for the experimental part of the paper

    "Fairness in Influence Maximization through Randomization" by Ruben Becker, Gianlorenzo D'Angelo, Sajjad Ghobadi, and Hugo Gilbert, published at AAAI-21. A complete version is available on the arxiv under: https://arxiv.org/abs/2010.03438

For execution of the experiments use
    python main.py experiment_type N

where experiment_type is one of
        [   'ba-singletons-uar-incr_n',
            'ba-bfs_cover-k-const-incr_n',
            'ba-rand_imbal-4-const-incr_n',
            'block_stochastic-const_05-incr_p',
            'tsang-gender-uar',
            'tsang-region-const']
and N is the number of experiments that are supposed to be run in parallel.

- For running tests including the implementation of Tsang et al. [1] download their code from github [2] into a subfolder called 'tsang'. Their code is executed by the function 'tsang_maximin' in the file 'tsang_maximin.py'. For running tests on the networks from their tests, download the instances into a folder 'tsang_networks'.

- The files have the following content:
    - main.py: contains the infrastructure for the execution of the experiments.
    - generation.py: contains the functions that generate the different instances.
    - influence_max.py: contains the necessary implementations of influence maximization functions: computation of the influence sigma of a set (and related functions sigma_v, sigma_C, see the paper), computation of nodes reachable from a given set, generation of live edge graphs, greedy algorithms for influence maximization and maximin
    - maximin_fish.py: contains the implementations of the myopic and naive_myopic routine of Fish et al. [4]
    - multiplicative_weight.py: contains the implementation of Young's multiplicative weight routine [3] for the node and set-based problems.
    - print_functions.py: contains some functions for prettier printing of graphs etc.
    - tsang_maximin.py: contains the function used to call the code of Tsang et al. [1]

- The execution generates an output file within the folder 'results' with the name being the experiment_type and the following sample content:
    graphname	                     algorithm	    n	m	  k	  no_live_edge_graphs	running_time	ex_ante		ex_post
    block_stochastic_n_120_p_0.03_0	 greedy_fish	120	1603  6	  100			        25.30			0.126		0.0989


- The experiment_types have the following meaning, see the paper for the interpretation of the community structure types:
    - 'ba-singletons-uar-incr_n':           + Barabasi albert graph (parameter m=2),
                                            + singleton community structure
                                            + edge weights uniformly at random in [0,1]
                                            + k = 5
                                            + n increasing from 5 to 50 in steps of 5

    - 'ba-bfs_cover-k-const-incr_n':        + Barabasi albert graph (parameter m=2)
                                            + BFS community structure
                                            + edge weights 0.1,
                                            + k = 10
                                            + n increasing from 10 to 50 in steps of 10

    - 'ba-rand_imbal-4-const-incr_n':       + Barabasi albert graph (parameter m=2)
                                            + Random imbalanced community structure
                                            + edge weights 0.1
                                            + k = 10
                                            + n increasing from 20 to 120 in steps of 10

    - 'block_stochastic-const_05-incr_p':   + block stochastic graph
                                            + block community structure with p increasing from 0.03 to 0.27 in steps of 0.03
                                            + edge weights 0.05
                                            + k = 6
                                            + n = 120

    - 'tsang-gender-uar':                   + instances of Tsang et al. (2019)
                                            + community structure induced by attribute gender
                                            + edge weights uniformly at random in [0,1]
                                            + k increasing from 5 to 50 in steps of 5
                                            + n = 500

    - 'tsang-region-const':                 + instances of Tsang et al. (2019)
                                            + community structure induced by attribute region
                                            + edge weights 0.1
                                            + k increasing from 5 to 50 in steps of 5
                                            + n = 500

[1] Tsang, Wilder, Rice, Tambe, Zick. Group-Fairness in Influence Maximization. IJCAI 2019.
    https://www.ijcai.org/Proceedings/2019/0831.pdf
[2] https://github.com/bwilder0/fair_influmax_code_release
[3] Young. Randomized Rounding without Solving the Linear Program. SODA 1995.
    https://arxiv.org/pdf/cs/0205036.pdf
[4] Fish, Bashardoust, Boyd, Friedler, Scheidegger, Venkatasubramanian. Gaps in Information Access in Social Networks. WWW 2019.
    http://sorelle.friedler.net/papers/access_networks_www19.pdf
