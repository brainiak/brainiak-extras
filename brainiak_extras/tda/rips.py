# Copyright 2016 Intel Corporation
#
# This file is part of brainiak-extras.
#
# brainiak-extras is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# brainiak-extras is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with brainiak-extras.  If not, see <http://www.gnu.org/licenses/>.

"""
Authors: Michael Lesnick, Bryn Keller

Computes barcodes of a Vietoris-Rips filtration of a distance matrix.

Input:
dist_mat, a distance matrix, represented as a list of lists;
max_scale, a real number;
max_dim, a non-negative integer.

Output:
The barcodes up to dimension max_dim, for the truncated Vietoris-Rips
filtration, including only simplices whose index of appearance is <=  max_scale
The barcodes are output as a list of three-element lists. Each three-element
lists represents one interval of one barcode and has the form
[birth,death,dimension]

More details: This code is designed to work with Bryn Keller's Python wrapper
for PHAT, the persistent homology code written by Ulrich Bauer, Michael Kerber,
Jan Reininghaus, and Hubert Wagner, available at
https://bitbucket.org/phat-code/phat. The approach to building the filtration
is based on the "Incremental Algorithm" described in the paper "Fast
Construction of the Vietoris-Rips Complex" by Afra Zomorodian. In the case that
we are building the whole Vietoris-Rips filtration, this is not the most
efficient approach, but may be good enough for the purposes of this wrapper.

By default, PHAT uses the twist optimization of Kerber and Chen to speed up the
computation When used together with persistent cohomology, the twist algorithm
drastically speeds up the computation of the persistent homology of Rips
complexes. Thus, in this code we do a cohomological computation (i.e., we build
the coboundary matrix rather than the boundary matrix.) The output, however, is
the same as for an ordinary persistent homology computation.

Note that the PHAT code also provides an option to take the primal matrix as
input and compute the coboundary matrix from this. (In the language of the PHAT
paper, forming the coboundary matrix amounts to taking the anti-transpose of
the boundary matrix.) In this case, it seems slightly nicer to just compute the
coboundary matrix directly, and input this to the PHAT wrapper; this is what we
do.

"""
import numpy as np
import phat
import sys
import scipy.sparse as sp


def _lower_neighbors(dist_mat, max_scale):
    """
    Converts a distance matrix to neighbor information.

    Takes a square, possibly lower triangular, and returns a list
    of lists of neighbor indices, for neighbors up to the specified scale.

    Parameters
    ----------
    dist_mat: 2D array
        the distance matrix, which may be lower triangular
    max_scale: float
        the highest scale (distance) to consider

    Returns
    -------
    neighbors: list of lists of int
    """
    d = sp.lil_matrix(dist_mat)
    d[d == 0] = sys.float_info.epsilon
    d[np.diag_indices(d.shape[0])] = 0
    d[d > max_scale] = 0
    d = sp.tril(d)
    result = [[] for i in range(d.shape[0])]
    for k, v in np.transpose(d.nonzero()):
        result[k].append(v)
    return result


def _add_cofaces(lower_neighbors, max_dim, dist_mat, start):
    """
    Returns all cofaces for the given start node.

    Cofaces are represented by lists of indices, paired with their
    stepwise distance from the start node.

    Parameters
    ----------
    lower_neighbors: list of lists of int
        neighbors for each index, as returned by the
        `_lower_neighbors` function
    max_dim: int
        the largest simplex dimension to consider
    dist_mat: 2D array
        the distance matrix, which may be lower triangular

    Returns
    -------
    simplices: list of (coface, distance) pairs
    """
    # TODO: iterative implementation, maybe,
    # since Python doesn't have tailcall elimination
    simplices = []

    def coface(tau, tau_dist, N):
        simplices.append((tau, tau_dist))
        if len(tau) >= max_dim + 1 or len(N) == 0:
            return
        else:
            for v in N:
                sigma = tau + [v]
                M = [val for val in N if val in lower_neighbors[v]]
                # get the distance at which sigma appears
                sigma_dist = max(tau_dist, *[dist_mat[u][v] for u in tau])
                coface(sigma, sigma_dist, M)

    coface([start], 0, lower_neighbors[start])
    return simplices


def _faces(tau):
    """
    Returns all faces (sublists with one element deleted) of tau.

    Parameters
    ----------
    tau: list of int
        a simplex

    Returns
    -------
    faces: list of list of int
    """
    for i in tau:
        tau_hat = tau[:]
        tau_hat.remove(i)
        yield tuple(tau_hat)


def gte_zero(n):
    """Type predicate: greater than or equal to zero"""
    return n >= 0


def gt_zero(n):
    """Type predicate: greater than zero"""
    return n > 0


def _rips_simplices(max_dim, max_scale, dist_mat):
    """
    Creates simplices from a distance matrix.

    Parameters
    ----------
    max_dim: int
        the largest simplex dimension to consider
    dist_mat: 2D array
        the distance matrix, which may be lower triangular
    max_scale: float
        the highest scale (distance) to consider

    Returns
    -------
    simplices: 2d int matrix
    """
    LN = _lower_neighbors(dist_mat, max_scale)
    simplices = np.concatenate([_add_cofaces(LN, max_dim, dist_mat, u)
                                for u in range(len(dist_mat))])

    # now, sort the simplices to put them in reverse filtration order.
    # The following line gives a valid filtration order because the python
    # sort is stable and the above method for constructing the filtration
    # always adds a simplex after its lower-dimensional faces
    sorted_simplices = sorted(simplices,
                              key=lambda labeled_simplex: labeled_simplex[1])

    # Now reverse the order to get the reverse filtration order.

    return sorted_simplices


def _create_coboundary_matrix(sorted_simplices, max_dim):
    """
    Creates a coboundary matrix for the given simplices.

    Parameters
    ----------
    sorted_simplices: list of simplices
        The simplices, in colex order
    max_dim: int
        the largest simplex dimension to consider

    Returns
    -------
    coboundaries: 2d int list
    """
    # now that the simplices are sorted, expand the list into a coboundary
    # matrix. For this, we use a Python dictionary, i.e. hash table.
    # Keys are simplices, represented as tuples of vectors, and values are
    # simplex indices. We build the dictionary as we build the boundary matrix

    # this will be our coboundary matrix. simplex dimensions are also stored,
    # as per the convention of PHAT and the PHAT wrapper.
    cobdy_matrix_pre = []

    # this will be our dictionary
    simplex_index_dict = {}

    # This builds the dictionary and initializes each column in
    # cobdy_matrix_pre to an empty column, with the appropriate dimension
    for i, (tau, _) in enumerate(sorted_simplices):
        # add each simplex tau together with its associated index to the
        # dictionary. If there are j simplices added already, we take the
        # new simplex to have index j.
        curr_index = len(sorted_simplices) - 1 - i
        simplex_index_dict[tuple(tau)] = curr_index
        # get the dimension of tau

        # note: PHAT requires each column to be labelled with an index.
        # The extra indices are needed to specify the order in which columns
        # are handled when using the twist optimization.
        # in the case of ordinary homology, this extra index is just the
        # dimension of the corresponding simplex, but in cohomology it is the
        # codimension", as defined in the following line of code.
        codim_tau = max_dim - (len(tau) - 1)
        # add a column in cobdy_matrx corresponding to tau, initially empty.
        cobdy_matrix_pre.insert(0, (codim_tau, []))

    # now we add in all of the column entries
    for i, (tau, _) in enumerate(sorted_simplices):
        curr_index = len(sorted_simplices) - 1 - i
        # for each face sigma of tau, add an entry corresponding to tau into
        # the coboundary column of sigma. Note how this uses the dictionary.

        if len(tau) > 1:
            for tau_hat in _faces(tau):
                cobdy_matrix_pre[simplex_index_dict[tau_hat]][1]\
                    .append(curr_index)

                # finally we sort each column of the coboundary matrix.

    for i, (tau, _) in enumerate(sorted_simplices):
        cobdy_matrix_pre[i][1].sort()

    return cobdy_matrix_pre


def rips_filtration(max_dim, max_scale, dist_mat):
    """
    Builds a boundary matrix for the boundary-Rips filtration up to dimension
     `max_dim`.

    Also builds the corresponding list of bigrades follows closely
    the "incremental algorithm" in the paper on fast Vietoris-Rips comptuation
    by Zomorodian, with some modification to store boundary matrix and
    filtration info. That in turn is based on a version of Bron-Kerbosch
    algorithm.

    Parameters
    ----------

    max_dim: int >= 0
        the highest dimension to compute
    max_scale: float
        the highest scale (distance) to consider
    dist_mat: 2D array
        an n x n distance matrix, which may be lower-triangular.

    Returns
    -------

    pairs: list of (column, grade) pairs
        The barcodes up to dimension max_dim, for the truncated Vietoris-Rips
        filtration, including only simplices whose index of appearance
        is <= max_scale. The barcodes are output as a list of three-element
        lists. Each three-element lists represents one interval of in barcode
        and has the form [birth,death,dimension]
    """
    sorted_simplices = _rips_simplices(max_dim, max_scale, dist_mat)
    len_minus_one = len(sorted_simplices) - 1
    cobdy_matrix_pre = _create_coboundary_matrix(sorted_simplices, max_dim)
    # print(cobdy_matrix_pre);

    # print(sorted_simplices)
    # print(bdy_matrix_pre)

    cobdy_matrix = phat.boundary_matrix(
        representation=phat.representations.bit_tree_pivot_column)
    cobdy_matrix.columns = cobdy_matrix_pre

    # call Bryn's PHAT wrapper for the persistence computation
    pairs = cobdy_matrix.compute_persistence_pairs()

    # next, rescale the pairs to their original filtration values, eliminating
    # pairs with the same birth and death time. In keeping with our chosen
    # output format, we also add the dimension to the pair.
    scaled_pairs = []
    for i in range(len(pairs)):
        cobirth = sorted_simplices[len_minus_one - pairs[i][0]][1]
        codeath = sorted_simplices[len_minus_one - pairs[i][1]][1]
        if codeath < cobirth:
            dimension = len(
                sorted_simplices[len_minus_one - pairs[i][1]][0]) - 1
            scaled_pairs.append((codeath, cobirth, dimension))

    # add in the intervals with endpoint inf
    # To do this, we first construct an array paired_indices such that
    # if the j^th simplex (in the coboundary order) appears in a pair,
    # paired_indices[j] = 1 otherwise paired_incides[j] = 0.

    paired_indices = np.zeros(len(sorted_simplices))
    for i in range(len(pairs)):
        paired_indices[pairs[i][0]] = 1
        paired_indices[pairs[i][1]] = 1
    for i in range(len(paired_indices)):
        if paired_indices[i] == 0:
            birth = sorted_simplices[len_minus_one - i][1]
            dimension = len(sorted_simplices[len_minus_one - i][0]) - 1
            scaled_pairs.append((birth, float("inf"), dimension))
    return scaled_pairs
