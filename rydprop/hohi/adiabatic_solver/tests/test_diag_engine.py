from ..plugins.diag_engine import Diagonaliser
import pytest
import numpy as np
from scipy.sparse import csr_matrix,isspmatrix,coo_matrix


def test_sparse_logic_numpy():
    """
    correctly determines if needs to create sparse or dense matrices, given a numpy input
    """
    return_vecs = False
    num_eig = 3
    h0 = np.random.rand(3,3)
    h1 = np.random.rand(3,3)
    my_diag = Diagonaliser(return_vecs,num_eig,h0,h1)
    assert my_diag.sparse == False
    

def test_sparse_logic_csr():
    """
    correctly determines if needs to create sparse or dense matrices, given a numpy input
    """
    return_vecs = False
    num_eig = 3
    h0 = csr_matrix(np.random.rand(3,3))
    h1 = csr_matrix(np.random.rand(3,3))
    my_diag = Diagonaliser(return_vecs,num_eig,h0,h1)
    assert my_diag.sparse == False
    
def test_sparse_logic_numpy_true():
    """
    correctly determines if needs to create sparse or dense matrices, given a numpy input
    """
    return_vecs = False
    num_eig = 1
    h0 = np.random.rand(3,3)
    h1 = np.random.rand(3,3)
    my_diag = Diagonaliser(return_vecs,num_eig,h0,h1)
    assert my_diag.sparse == True
    

def test_sparse_logic_csr_true():
    """
    correctly determines if needs to create sparse or dense matrices, given a numpy input
    """
    return_vecs = False
    num_eig = 1
    h0 = csr_matrix(np.random.rand(3,3))
    h1 = csr_matrix(np.random.rand(3,3))
    my_diag = Diagonaliser(return_vecs,num_eig,h0,h1)
    assert my_diag.sparse == True
    
    
def test_matrix_converter_dense_to_sparse():
    """
    given dense matrices and a sparse toggle will convert to csr_matrices
    """
    return_vecs = False
    h0 = np.zeros((3,3))
    h1= np.zeros((3,3))
    num_eig = 1
    my_diag = Diagonaliser(return_vecs,num_eig,h0,h1)

    assert type(my_diag.h0) == csr_matrix and type(my_diag.h0) == csr_matrix
    
def test_matrix_converter_dense_to_dense():
    """
    given dense matrices and a sparse toggle will convert to csr_matrices
    """
    return_vecs = False
    h0 = np.zeros((3,3))
    h1= np.zeros((3,3))
    num_eig = 3
    my_diag = Diagonaliser(return_vecs,num_eig,h0,h1)

    assert type(my_diag.h0) == np.ndarray and type(my_diag.h0) == np.ndarray
    
def test_matrix_converter_sparse_to_dense():
    """
    given dense matrices and a sparse toggle will convert to csr_matrices
    """
    return_vecs = False
    h0 = csr_matrix(np.zeros((3,3)))
    h1= csr_matrix(np.zeros((3,3)))
    num_eig = 3
    my_diag = Diagonaliser(return_vecs,num_eig,h0,h1)

    assert type(my_diag.h0) == np.ndarray and type(my_diag.h0) == np.ndarray
    
def test_matrix_converter_sparse_to_sparse():
    """
    given dense matrices and a sparse toggle will convert to csr_matrices
    """
    return_vecs = False
    h0 = csr_matrix(np.zeros((3,3)))
    h1= csr_matrix(np.zeros((3,3)))
    num_eig = 1
    my_diag = Diagonaliser(return_vecs,num_eig,h0,h1)

    assert type(my_diag.h0) == csr_matrix and type(my_diag.h0) == csr_matrix
    
def test_matrix_converter_coo_to_sparse():
    """
    matrices generated as coo will be accepted
    """
    return_vecs = False
    h0 = coo_matrix(np.zeros((3,3)))
    h1= coo_matrix(np.zeros((3,3)))
    num_eig = 1
    my_diag = Diagonaliser(return_vecs,num_eig,h0,h1)

    assert type(my_diag.h0) == csr_matrix and type(my_diag.h0) == csr_matrix