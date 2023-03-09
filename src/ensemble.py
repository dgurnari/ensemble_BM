import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

from pyballmapper import BallMapper


def get_points_in_same_cc(bm, point):
    '''Return a list of points that are covered by balls 
    in the same connected component of `bm` as the one 
    that covers `point` 
    
    Parameters
    ----------
    bm: BallMapper object
        The BallMapper graph to consider

    point: int
        the id of the point to consider

    Returns
    -------
    points_in_cc: set
        ids of points in the same connected component of `point`
        
    '''

    for node in bm.Graph.nodes:
        if point in bm.Graph.nodes[node]['points covered']:
            cc = nx.node_connected_component(bm.Graph, node)
            break

    points_in_cc = set().union(*[bm.Graph.nodes[node]['points covered'] for node in cc])

    return points_in_cc



def similarity_matrix_from_BM(bm):
    ''' Computes the similarity matrix from a BallMapper graph
    
    Parameters
    ----------
    bm: BallMapper object
        The BallMapper graph to consider.

    Returns
    -------
    points_in_cc: ndarray of shape (n_samples, n_samples) where n_samples
        is the total number of points covered by the BallMapper graph.
        Similarity matrix.
        
    '''

    all_points = sorted(set().union(*[bm.Graph.nodes[node]['points covered'] 
                                      for node in bm.Graph.nodes]))
    
    sim = np.eye(len(all_points), dtype=int)

    for i, p in enumerate(all_points[:-1]):
        for j, q in enumerate(all_points[i+1:], start=i+1):
            if p in get_points_in_same_cc(bm, q):
                sim[i,j] = 1
                sim[j,i] = 1

    return sim




def ensemble_BM(X, eps, num_it, seed=42, **kwargs):
    ''' Runs BM num_int times by randomly permuting the order of points and 
    computes the normalized similarity matrix.
    
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Data.

    eps: float
        epsilon parameter for the BallMapper

    num_it: int
        number of iterations

    seed: int, default=42
        seed of the numpy default_rng to permute the input data

     **kwargs: 
        additional BM parameters

    Returns
    -------
    sim: ndarray of shape (n_samples, n_samples) 
        Normalized similarity matrix.
        
    '''
    rng = np.random.default_rng(seed=seed)

    sim = np.zeros((len(X), len(X)), dtype=float)

    idx_list = [i for i in range(len(X))]

    for i in range(num_it):
        current_order = rng.permutation(idx_list).tolist()
        current_bm = BallMapper(X, eps, order=current_order, **kwargs)

        sim += similarity_matrix_from_BM(current_bm)

    return sim / num_it



def match_cc(ensemble_cc, target_bm):
    '''compute the optimal matching between ccs'''    
    B = nx.Graph()

    ccT_temp = get_connected_components_dict(target_bm)
    ccT = {'T_{}'.format(k):value for k, value  in ccT_temp.items()}

    del ccT_temp

    B.add_nodes_from(ensemble_cc.keys(), bipartite=0)
    B.add_nodes_from(ccT.keys(), bipartite=1)

    for n1 in ensemble_cc.keys():
        for n2 in ccT.keys():
            # we flip the sign of the weights beause the code minimizes the cost...        
            B.add_edge(n1, n2, 
                    weight=-1*len(ensemble_cc[n1].intersection(ccT[n2])) / len(ensemble_cc[n1].union(ccT[n2])) )

    full_matching = nx.bipartite.minimum_weight_full_matching(B, weight='weight')

    return {k: ccT[full_matching[k]] for k in ensemble_cc.keys() if k in full_matching}



def get_connected_components_dict(bm):

    '''Return a dict containing the points covered by each connected
    component in `bm` '''

    cc_dict = dict()

    for i, cc in enumerate(sorted(nx.connected_components(bm.Graph), key=len, reverse=True)):
        # for each connected components, find the set of points covered by its nodes
        points_in_cc = set().union(*[bm.points_covered_by_landmarks[idx] for idx in cc])
        cc_dict[i] = points_in_cc
    
    return cc_dict



def ensemble_BM_cc(X, eps, default_cc, num_it, seed=42, **kwargs):
    '''runs BM on the default order and compares it with (n-1) runs'''
    
    matched_cc = {k: [] for k in default_cc.keys()}

    rng = np.random.default_rng(seed=seed)

    idx_list = [i for i in range(len(X))]

    for i in range(num_it):
        current_order = rng.permutation(idx_list).tolist()
        current_bm = BallMapper(X, eps, order=current_order, **kwargs)

        current_matching = match_cc(default_cc, current_bm)

        for key in current_matching:
            matched_cc[key].append(current_matching[key])

    return matched_cc
