import numpy as np

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.basic import StationaryModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator
from pymor.parameters.functionals import ProjectionParameterFunctional, ConjugateParameterFunctional

def matrixmodel(A, B, C):

    '''
    This function creates stationary models for the following linear coercive models derived for the given three matrices A, B, and C:
    
        a_1(v, w; s) = w*(sI_n - A)v and l_1(w) = w*B
        a_2(v, w; s) = w*(sI_n - A)*v and l_2(w) = w*C^T.

    Inputs:
    ------------------------------------------------
    A - matrix -> NumPy array or NumpyMatrixOperator
    B - vector -> NumPy array or NumpyMatrixOperator
    C - vector -> NumPy array or NumpyMatrixOperator
    ------------------------------------------------
    Outputs:
    ------------------------------------------------
    model_V - Stationary Model of linear coercive model w*(sI_n - A)v = w*B -> StationaryModel
    model_W - Stationary Model of linear coercive model w*(sI_n - A)*v = w^*C^{T} -> StationaryModel
    '''

    # Define operators (and also a dimension of a model)
    if isinstance(A, np.ndarray):
        dim = A.shape[0]
        A_op = NumpyMatrixOperator(A)
    else:
        dim = to_matrix(A).shape[0]
        A_op = A

    if isinstance(B, np.ndarray):
        B_op = NumpyMatrixOperator(B) 
    else:
        B_op = B  

    if isinstance(C, np.ndarray):
        C_op = NumpyMatrixOperator(C.T)  
    else:
        C_op = C.H  # C is real, so adjoint is transpose

    I_op = NumpyMatrixOperator(np.eye(dim))
    
    # Define parameter functional for 's'
    s_param = ProjectionParameterFunctional('s', 1)

    # Define bilinear form a(v, w; s) = w*(sI - A)v
    a_op_1 = LincombOperator([I_op, A_op], [s_param, -1])

    # Define bilinear form a(v, w; s) = w*(sI - A)*v -> Note: (sI - A)* = s*I - A*
    a_op_2 = LincombOperator([I_op, A_op.H], [ConjugateParameterFunctional(s_param), -1])
    
    # Define linear functional l(w) = w^*B
    l_op_1 = B_op

    # Define linear functional l(w) = w^*C^{T}
    l_op_2 = C_op

    # Define the StationaryModels
    model_V = StationaryModel(operator=a_op_1, rhs=l_op_1)
    model_W = StationaryModel(operator=a_op_2, rhs=l_op_2)

    return [model_V, model_W] 
