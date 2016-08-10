#rips.py
#Author: Michael Lesnick
#
#Computes barcodes of a Vietoris-Rips filtration of a distance matrix.
#
#Input: 
#dist_Mat, a distance matrix, represented as a list of lists; 
#max_Scale, a real number; 
#max_Dim, a non-negative integer.  
#dist_Mat[j] represents the j^{th} row of the distance matrix (assuming the indexing starts at 0), 
#or alternatively, the part of the distance matrix below the diagonal (henceforth, PBD). 
#The code does not use any part of dist_Mat other than the PBD, and behaves the same regardless of whether the full matrix or just the PBD is given.  
#Note that if only the PBD is given, we have dist_Mat[0]=[].
#
#
#Output: 
#The barcodes up to dimension max_Dim, for the truncated Vietoris-Rips filtration, including only simplices whose index of appearance is <= max_Scale  
#The barcodes are output as a list of three-element lists.  
#Each three-element lists represents one interval of in barcode and has the form [birth,death,dimension] 
#
#More details: This code is designed to work with Bryn Kellers's Python wrapper for PHAT, the persistent homology code written by 
#Ulrich Bauer, Michael Kerber, Jan Reininghaus, and Hubert Wagner, available at
#https://bitbucket.org/phat-code/phat
#The approach to building the bifiltration is based on the ``Incremental Algorithm" described in the paper 
#``Fast Construction of the Vietoris-Rips Complex" by Afra Zomorodian.
# In the case that we are building the whole Vietoris-Rips filtration, this is not the most efficient appraoch, 
#but may be good enough for the purposes of this wrapper.
#
# Example usage:
# COMING SOON, ALSO SEE BELOW IN THIS VERY FILE

#To prepare for construction of the boundary matrices, first convert dist_Mat into a column-sparse upper tringular incidence matrix N for the max_Scale-thresholded neighborhood graph
import numpy as np
import phat

def Lower_Neighbors(dist_Mat,max_Scale):
    LN=[]    
    for i in range(np.shape(dist_Mat)[0]):                   
        LN_List=[]        
        for j in range(i):
            if dist_Mat[i][j]<=max_Scale:
                LN_List.append(j)
        LN.append(LN_List)   
    return LN

# helper function for Rips_Filtration
def add_cofaces(Lower_Neighbors,max_Dim,tau,tau_dist,N,simplices,dist_Mat):   
    simplices.append([tau,tau_dist])     
    if len(tau) >= max_Dim+1 or len(N)==0:
        return simplices
    else:
        for v in N:   
            sigma=tau[:]
            sigma.append(v)
            M = [val for val in N if val in Lower_Neighbors[v]]          
            #get the distance at which sigma appears           
            sigma_dist=tau_dist            
            for u in tau:
                sigma_dist=max(sigma_dist,dist_Mat[v][u])
            simplices=add_cofaces(Lower_Neighbors,max_Dim,sigma,sigma_dist,M,simplices,dist_Mat)
        return simplices 

#builds a boundary matrix for the boundary-Rips bifiltration up to dimension k.
#also builds the corresponding list of bigrades 
#follows closely the ``incremental algorithm" in the paper on fast Vietoris-Rips comptuation by Zomorodian, with some modification to store boundary matrix and filtration info
#That in turn is based on a version of Bron-Kerbosch algorithm

#Input is clear from the argument names
#Output: simplices, a list of [Column,grade] pairs.  Each pair is in list form.  
def Rips_Filtration(max_Dim,max_Scale,dist_Mat):
    LN=Lower_Neighbors(dist_Mat,max_Scale)    
    simplices=[]
    for u in range(len(dist_Mat)):     
        N=LN[u]
        simplices=add_cofaces(LN,max_Dim,[u],0,N,simplices,dist_Mat)

    #now, sort the simplices to put them in filtration order  
    #the following line gives a valid filtration order because the python sort is stable and     
    #the above method for constructing the filtration always adds a simplex after its lower-dimensional faces   
    sorted_simplices=sorted(simplices, key=lambda labelled_simplex: labelled_simplex[1])
    
    #now that the simplices are sorted, expand the list into a boundary matrix.
    #For this, we use a Python dictionary, i.e. hash table.
    #Keys are simplices, represented as tuples of vectors, and values are simplex indices.
    #We build the dictionary as we build the boundary matrix    
    
    #this will be our boundary matrix.  simplex dimensions are also stored, as per the convention of PHAT and the PHAT wrapper.    
    bdy_matrix_pre=[]    
    
    #this will be our dictionary
    simplex_index_dict={}  
    
    for i in range(len(simplices)):    
        #add each simplex tau together with its associated index to the dictionary.  
        #dictionary takes immutatble data types for keys; that's why we convert the list to a tuple here.
        tau=sorted_simplices[i][0]        
        cur_tuple=tuple(tau)   
        #if there are j simplices added already, we take the new simplex to have index j.  
        simplex_index_dict[cur_tuple] = i   
        
        #get the dimension of tau
        dim_tau=len(tau)-1
        
        #now compute the boundary column associtated to tau with the help of the dictionary
        tau_column=[];
        for i in tau:
            tau_hat=tau[:]
            tau_hat.remove(i)
            if len(tau_hat)>0:
                tau_column.append(simplex_index_dict[tuple(tau_hat)])  
        bdy_matrix_pre.append((dim_tau,tau_column))       
    print(sorted_simplices)    
    print(bdy_matrix_pre)
    bdy_matrix = phat.boundary_matrix(representation = phat.representations.vector_vector) 
    #bdy_matrix = phat.boundary_matrix(representation = phat.representations.bit_tree_pivot_column)         
    bdy_matrix.columns=bdy_matrix_pre
    
    pairs = bdy_matrix.compute_persistence_pairs()
    for i in range(len(pairs)):
        print(pairs[i])
    
    #next, rescale the pairs to their original filtration values, eliminating pairs with the same birth and death time.
    #In keeping with our chosen output format, we also add the dimension to the pair.
    scaled_pairs=[]
    for i in range(len(pairs)):
        birth=sorted_simplices[pairs[i][0]][1]
        death=sorted_simplices[pairs[i][1]][1]
        print('first simplex',sorted_simplices[pairs[i][0]][0])
        print('second simplex',sorted_simplices[pairs[i][1]][0])
        print('birth:',birth)
        print('death:',death)
        if birth<death:
           dimension=len(sorted_simplices[pairs[i][0]])-1
           scaled_pairs.append([birth,death,dimension])

    #add in the intervals with endpoint inf
    #To do this, we first construct an array paired_indices such that
    #if the j^th simplex appears in a pair, paired_incides[j]=1
    #otherwise paired_incides[j]=0.
    
    paired_indices=np.zeros(len(simplices))
    for i in range(len(pairs)):
        paired_indices[pairs[i][0]]=1
        paired_indices[pairs[i][1]]=1
      
    for i in range(len(paired_indices)):
        if paired_indices[i]==0:
            birth=sorted_simplices[i][1]            
            dimension=len(sorted_simplices[i][0])-1
            scaled_pairs.append([birth,float("inf"),dimension])
    return scaled_pairs

# example, based on the four-element point cloud (0,0),(0,1),(1,0),(1,1)
#note that we approximate sqrt(2) by 1.4.
my_dist_Mat=[[0,1,1,1.4],[1,0,1.4,1],[1,1.4,0,1],[1.4,1,1,0]]
    
pairs_with_dim=Rips_Filtration(3,10,my_dist_Mat)
print("\nThere are %d persistence pairs: " % len(pairs_with_dim))
for triplet in pairs_with_dim:    
    print("Birth: ",triplet[0],", Death: ",triplet[1],", Dimension: ",triplet[2])
        
