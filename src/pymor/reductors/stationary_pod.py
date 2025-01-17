import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.reductors.basic import StationaryRBReductor

def matrixreductor(model_V, model_W, training_set, reduced_order_V, reduced_order_W):
    
    '''
    Inputs:
    ------------------------------------------------
    model_V - Stationary Model of linear coercive model w*(sI_n - A)v = w*B -> StationaryModel
    model_W - Stationary Model of linear coercive model w*(sI_n - A)*v = w*C^{T} -> StationaryModel
    training_set - an array containing parameters used to construct the snapshot matrix -> type(training_set[i]) = pymor.parameters.base.Mu (list of Mu objects)
    ------------------------------------------------
    Outputs:
    ------------------------------------------------
    pod_rom_V
    pod_rom_W
    pod_reductor_V
    pod_reductor_W
    '''

    # Compute FOM solutions for the parameters in the training set
    solution_snapshots_V = model_V.solution_space.empty()
    solution_snapshots_W = model_W.solution_space.empty()
    for s in training_set:
        solution_snapshots_V.append(model_V.solve(s))
        solution_snapshots_W.append(model_W.solve(s))
        
    # Snapshot matrices
    snapshot_matrix_V = solution_snapshots_V.to_numpy().T # Note: One may also use solution_snapshots_V.impl._array.T to get np.ndarray type needed for computation
    snapshot_matrix_W = solution_snapshots_W.to_numpy().T

    # Finding the Singular Value Decomposition (SVD) of snapshot matrices -> S = UΣV^T
    U_V, _, _ = np.linalg.svd(snapshot_matrix_V, full_matrices = True)
    U_W, _, _ = np.linalg.svd(snapshot_matrix_W, full_matrices = True)

    if reduced_order_V > min(snapshot_matrix_V.shape):
        raise ValueError("'reduced_order_V' cannot exceed the rank of the snapshot matrix.")
    if reduced_order_W > min(snapshot_matrix_W.shape):
        raise ValueError("'reduced_order_W' cannot exceed the rank of the snapshot matrix.")

    # The reduced bases (POD bases)
    pod_basis_numpy_V = U_V[:,:reduced_order_V]
    pod_basis_numpy_W = U_W[:,:reduced_order_W]

    # Convert NumPy array into VectorArray 
    space_V = NumpyVectorSpace(model_V.order) #number of columns = model_V.order
    space_W = NumpyVectorSpace(model_W.order)
    pod_basis_V = space_V.make_array(pod_basis_numpy_V.T) #This is actually transpose of POD-RB basis
    pod_basis_W = space_W.make_array(pod_basis_numpy_W.T) #This is actually transpose of POD-RB basis
    
    # POD-Galerkin RB method
    pod_reductor_V = StationaryRBReductor(model_V, RB = pod_basis_V) 
    pod_reductor_W = StationaryRBReductor(model_W, RB = pod_basis_W) 
    pod_rom_V = pod_reductor_V.reduce()
    pod_rom_W = pod_reductor_W.reduce()

    return [pod_rom_V, pod_reductor_V, pod_rom_W, pod_reductor_W]