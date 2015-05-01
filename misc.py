##
# Miscellaneous helper functions
##

import numpy as np

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####


    epsilon = np.sqrt(6)/np.sqrt(m+n)
    A0 = (np.random.random((m,n)) - 0.5) * epsilon * 2
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0
