import numpy as np

from pymor.basic import *
from pymor.parameters.base import Mu
from pymor.algorithms.to_matrix import to_matrix
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional, ConjugateParameterFunctional
from pymor.vectorarrays.numpy import NumpyVectorSpace