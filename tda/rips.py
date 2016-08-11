"""
Author: Michael Lesnick

Computes barcodes of a Vietoris-Rips filtration of a distance matrix.

Input:
dist_mat, a distance matrix, represented as a list of lists;
max_scale, a real number;
max_dim, a non-negative integer.
dist_mat[j] represents the j^{th} row of the distance matrix (assuming the
indexing starts at 0), or alternatively, the part of the distance matrix below
the diagonal (henceforth, PBD). The code does not use any part of dist_mat
other than the PBD, and behaves the same regardless of whether the full matrix
or just the PBD is given. Note that if only the PBD is given, we have
dist_mat[0] = [].


Output:
The barcodes up to dimension max_dim, for the truncated Vietoris-Rips
filtration, including only simplices whose index of appearance is <=  max_scale
The barcodes are output as a list of three-element lists. Each three-element
lists represents one interval of in barcode and has the form
[birth,death,dimension]

More details: This code is designed to work with Bryn Kellers's Python wrapper
for PHAT, the persistent homology code written by Ulrich Bauer, Michael Kerber,
Jan Reininghaus, and Hubert Wagner, available at
https://bitbucket.org/phat-code/phat. The approach to building the bifiltration
is based on the ``Incremental Algorithm" described in the paper ``Fast
Construction of the Vietoris-Rips Complex" by Afra Zomorodian. In the case that
we are building the whole Vietoris-Rips filtration, this is not the most
efficient approach, but may be good enough for the purposes of this wrapper.

 Example usage:
 COMING SOON, ALSO SEE BELOW IN THIS VERY FILE
"""
import numpy as np
import phat
import itertools as it

#To prepare for construction of the boundary matrices, first convert dist_mat
#into a column-sparse lower triangular incidence matrix N for the
#max_scale-thresholded neighborhood graph
def lower_neighbors(dist_mat,max_scale):
    d = np.tril(dist_mat)
    d[d > max_scale] = 0
    #TODO: use csc_matrix instead of lists?
    result = [[] for i in range(d.shape[0])]
    for k, v in np.transpose(d.nonzero()):
        result[k].append(v)
    return result

# helper function for rips_filtration
def add_cofaces(lower_neighbors, max_dim, dist_mat, start):
    #TODO: iterative implementation (maybe), since Python doesn't have tailcall elimination
    simplices = []
    def coface(tau, tau_dist, N):
        simplices.append((tau, tau_dist))
        if len(tau) >=  max_dim+1 or len(N) == 0:
            return
        else:
            for v in N:
                sigma = tau + [v]
                M = [val for val in N if val in lower_neighbors[v]]
                #get the distance at which sigma appears
                sigma_dist = max(tau_dist, *[dist_mat[v][u] for u in tau])
                coface(sigma, sigma_dist, M)
    coface([start], 0, lower_neighbors[start])
    return simplices

def faces(tau):
    for i in tau:
        tau_hat = tau[:]
        tau_hat.remove(i)
        yield tuple(tau_hat)

def rips_simplices(max_dim, max_scale, dist_mat):

    LN = lower_neighbors(dist_mat,max_scale)
    simplices = np.concatenate([add_cofaces(LN, max_dim, dist_mat, u)
                                for u in range(len(dist_mat))])

    #now, sort the simplices to put them in filtration order the following line
    #gives a valid filtration order because the python sort is stable and the
    #above method for constructing the filtration always adds a simplex after
    #its lower-dimensional faces
    sorted_simplices = sorted(simplices, key = lambda labeled_simplex: labeled_simplex[1])

    return sorted_simplices

def create_boundary_columns(sorted_simplices):

    #now that the simplices are sorted, expand the list into a boundary matrix.
    #For this, we use a Python dictionary, i.e. hash table.
    #Keys are simplices, represented as tuples of vectors, and values are simplex indices.
    #We build the dictionary as we build the boundary matrix

    #this will be our boundary matrix. simplex dimensions are also stored, as
    #per the convention of PHAT and the PHAT wrapper.
    bdy_matrix_pre = []

    #this will be our dictionary
    simplex_index_dict = {}

    for i, (tau, _) in enumerate(sorted_simplices):
        #add each simplex tau together with its associated index to the dictionary.
        #if there are j simplices added already, we take the new simplex to have index j.
        simplex_index_dict[tuple(tau)] = i

        #get the dimension of tau
        dim_tau = len(tau)-1

        #now compute the boundary column associated to tau with the help of the dictionary
        tau_column = [simplex_index_dict[tau_hat] for tau_hat in faces(tau) if len(tau_hat) > 0]
        tau_column.sort()
        bdy_matrix_pre.append((dim_tau,tau_column))
    return bdy_matrix_pre

#builds a boundary matrix for the boundary-Rips bifiltration up to dimension k.
#also builds the corresponding list of bigrades
#follows closely the ``incremental algorithm" in the paper on fast Vietoris-Rips comptuation by Zomorodian, with some modification to store boundary matrix and filtration info
#That in turn is based on a version of Bron-Kerbosch algorithm
#Input is clear from the argument names
#Output: simplices, a list of [Column,grade] pairs.  Each pair is in list form.
def rips_filtration(max_dim, max_scale, dist_mat):
    sorted_simplices = rips_simplices(max_dim, max_scale, dist_mat)

    bdy_matrix_pre = create_boundary_columns(sorted_simplices)
    
    #print(sorted_simplices)
    #print(bdy_matrix_pre)
    
    bdy_matrix = phat.boundary_matrix(representation = phat.representations.bit_tree_pivot_column)
    bdy_matrix.columns = bdy_matrix_pre

    #call Bryn's PHAT wrapper for the persistence computation
    pairs = bdy_matrix.compute_persistence_pairs()

    #next, rescale the pairs to their original filtration values, eliminating pairs with the same birth and death time.
    #In keeping with our chosen output format, we also add the dimension to the pair.
    scaled_pairs = []
    for i in range(len(pairs)):
        birth = sorted_simplices[pairs[i][0]][1]
        death = sorted_simplices[pairs[i][1]][1]
        if birth<death:
           dimension = len(sorted_simplices[pairs[i][0]])-1
           scaled_pairs.append([birth,death,dimension])

    #add in the intervals with endpoint inf
    #To do this, we first construct an array paired_indices such that
    #if the j^th simplex appears in a pair, paired_incides[j] = 1
    #otherwise paired_incides[j] = 0.

    paired_indices = np.zeros(len(sorted_simplices))
    for i in range(len(pairs)):
        paired_indices[pairs[i][0]] = 1
        paired_indices[pairs[i][1]] = 1

    for i in range(len(paired_indices)):
        if paired_indices[i] == 0:
            birth = sorted_simplices[i][1]
            dimension = len(sorted_simplices[i][0])-1
            scaled_pairs.append([birth,float("inf"),dimension])
    return scaled_pairs

if __name__ ==  '__main__':

    #TODO: turn this into a proper unit test
    # example, based on the four-element point cloud (0,0),(0,1),(1,0),(1,1)
    #note that we approximate sqrt(2) by 1.4.
    my_dist_mat = [[0,1,1,1.4],[1,0,1.4,1],[1,1.4,0,1],[1.4,1,1,0]]
    pairs_with_dim = rips_filtration(3,10,my_dist_mat)
    sorted_pairs = sorted(map(tuple, list(pairs_with_dim)))
    print("\nThere are %d persistence pairs: " % len(pairs_with_dim))
    for triplet in pairs_with_dim:
        print("Birth: ",triplet[0],", Death: ",triplet[1], type(triplet[1]), ", Dimension: ",triplet[2])
    #print(sorted_pairs)
    #assert(sorted_pairs == [(0, 1, 1), (0, 1, 1), (0, 1.4, 1), (0, float('inf'), 0), (1, 1.4, 1), (1, 1.4, 1)])
