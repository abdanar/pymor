import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt_biorth
from pymor.vectorarrays.numpy import NumpyVectorSpace

def pod_projection(pod_rom_V, pod_rom_W, pod_reductor_W, pod_reductor_V, mu, b, c):

    '''
    Inputs:
    ------------------------------------------------
    pod_rom_V
    pod_rom_W
    pod_reductor_V
    pod_reductor_W
    mu - list of -mu_i values -> type(mu[i]) = pymor.parameters.base.Mu (list of Mu objects)
    b - NumPy array -> b.shape = (r, 1) where r = len(mu)
    c - NumPy array -> c.shape = (r, 1) where r = len(mu)
    validation_set - an array containing parameters used to evaluate the reduced model after its construction -> type(validation_set[i]) = pymor.parameters.base.Mu (list of Mu objects)
    ------------------------------------------------
    Outputs: Biorthonormal pair of projection matrices V, W using biorthonormal Gram-Schmidt process
    ------------------------------------------------
    V - projection matrix V -> NumpyVectorArray -> V.shape = (n, r)
    W - projection matrix W -> NumpyVectorArray -> W.shape = (n, r)
    '''
    
    # Solution arrays containing len(validation_set) many reduced samples
    r = len(mu)
    reduced_solution_V = pod_rom_V.solution_space.empty()
    reduced_solution_W = pod_rom_W.solution_space.empty()
    for s in mu:
        reduced_solution_V.append(pod_rom_V.solve(s))
        reduced_solution_W.append(pod_rom_W.solve(s))
        
    # It would be better to get matrices where columns are the reconstructed reduced solutions as in theory we will use such matrix; however PyMor only has vstack option (appending as a row of a matrix)
    reduced_solution_reconstruct_V_T = pod_reductor_V.reconstruct(reduced_solution_V) # a matrix with rows representing the reconstructed reduced solutions for different parameter values to first parametrized coercive model (row i will give us (s_{i}I - A)^{-1}B)
    reduced_solution_reconstruct_W_T = pod_reductor_W.reconstruct(reduced_solution_W) # a matrix with rows representing the reconstructed reduced solutions for different parameter values to second parametrized coercive model (row i will give us (s_{i}I - A)^{-*}C^T)

    # To align with the theory, we take the transpose of the result. Also, note that the transpose operation does not exist in PyMor for `NumpyVectorArray`, so we first take the transpose of the NumPy array and then convert it back
    space_V_numpy = NumpyVectorSpace(r)
    space_W_numpy = NumpyVectorSpace(r)
    R_V = space_V_numpy.make_array(reduced_solution_reconstruct_V_T.to_numpy().T)
    R_W = space_W_numpy.make_array(reduced_solution_reconstruct_W_T.to_numpy().T)

    R_V, R_W = R_V.to_numpy(), R_W.to_numpy() # Note: One may also use R_V.impl._array to get np.ndarray type needed for computation
    D_b, D_c = np.diag(b.flatten()), np.diag(c.flatten())

    V_numpy = np.matmul(R_V, D_b)
    W_numpy = np.matmul(R_W, D_c)

    space = NumpyVectorSpace(V_numpy.shape[0])
    V = space.make_array(V_numpy.T)
    W = space.make_array(W_numpy.T)
    [V_bi, W_bi] = gram_schmidt_biorth(V, W) # NumpyVectorArray

    return [V_bi, W_bi]  
