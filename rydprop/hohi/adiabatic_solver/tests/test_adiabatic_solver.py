from ..adiabatic_solver import *
from ..plugins.state_init import *
import pytest
import numpy as np
from scipy.sparse import csr_matrix,isspmatrix,coo_matrix



@pytest.mark.xfail(raises=ValueError)
def test_diager_diff_dimension():
    h0 = np.zeros((3,3))
    h1 = np.zeros((4,4))
    return_vecs = True
    num_eigs = 1
    Diag_engine = init_diagonalizer(return_vecs,num_eigs,h0,h1)

@pytest.mark.xfail(raises=ValueError)
def test_diager_num_eigs():
    """
    return more eigs than dimension of system
    """
    h0 = np.zeros((3,3))
    h1 = np.zeros((3,3))
    return_vecs = True
    num_eigs = 6
    Diag_engine = init_diagonalizer(return_vecs,num_eigs,h0,h1)
    
@pytest.mark.xfail(raises=ValueError)
def test_diager_num_eigs():
    """
    return more eigs than dimension of system
    """
    h0 = np.zeros((3,4))
    h1 = np.zeros((3,3))
    return_vecs = True
    num_eigs = 6
    Diag_engine = init_diagonalizer(return_vecs,num_eigs,h0,h1)
    
@pytest.mark.xfail(raises=TypeError)
def test_diager_matrix_type():
    """
    return more eigs than dimension of system
    """
    h0 = np.zeros((3,4))
    h1 = 'foo'
    return_vecs = True
    num_eigs = 1
    Diag_engine = init_diagonalizer(return_vecs,num_eigs,h0,h1)
    
@pytest.mark.xfail(raises=TypeError)
def test_diager_matrix_type_different():
    """
    return more eigs than dimension of system
    """
    h0 = np.zeros((3,3))
    h1 = csr_matrix(np.zeros((3,3)))
    return_vecs = True
    num_eigs = 1
    Diag_engine = init_diagonalizer(return_vecs,num_eigs,h0,h1)
    
def test_diager_matrix_type_fine_csr():
    """
    return more eigs than dimension of system
    """
    h0 = csr_matrix(np.zeros((3,3)))
    h1 = csr_matrix(np.zeros((3,3)))
    return_vecs = True
    num_eigs = 1
    Diag_engine = init_diagonalizer(return_vecs,num_eigs,h0,h1)
    
    assert Diag_engine.num_eigs == 1
    
def test_diager_matrix_type_fine_coo():
    """
    return more eigs than dimension of system
    """
    h0 = coo_matrix(np.zeros((3,3)))
    h1 = coo_matrix(np.zeros((3,3)))
    return_vecs = True
    num_eigs = 1
    Diag_engine = init_diagonalizer(return_vecs,num_eigs,h0,h1)
    
    assert Diag_engine.num_eigs == 1
    
    
#tests on the sigma initialiser
    
def test_sigma_guesser_correct():
    sig_method = 'target'
    target = [1,2,3]
    ratio = [0.2,0.7,0.1]
    dimension = 3
    checker = init_sigma_checker(sig_method,target,ratio,dimension)

    assert checker.target_list[0] ==1 and checker.ratio_list[1] ==0.7

def test_sigma_guesser_no_ratio():
    sig_method = 'target'
    target = [1,2]
    ratio = None
    dimension = 3
    checker = init_sigma_checker(sig_method,target,ratio,dimension)

    assert checker.target_list[0] ==1 and checker.ratio_list[1] ==0.5
    
def test_sigma_guesser_no_ratio_single():
    sig_method = 'target'
    target = [1]
    ratio = None
    dimension = 3
    checker = init_sigma_checker(sig_method,target,ratio,dimension)

    assert checker.target_list[0] ==1 and checker.ratio_list[0] ==1.0
    
def test_sigma_guesser_no_ratio_single():
    sig_method = 'target'
    target = [1]
    ratio = None
    dimension = 3
    checker = init_sigma_checker(sig_method,target,ratio,dimension)

    assert checker.target_list[0] ==1 and checker.ratio_list[0] ==1.0
    
@pytest.mark.xfail(raises=ValueError)
def test_sigma_guesser_target_ratio_different_length():
    sig_method = 'target'
    target = [1,2,3]
    ratio = [0.6,0.4]
    dimension = 3
    checker = init_sigma_checker(sig_method,target,ratio,dimension)
    
@pytest.mark.xfail(raises=ValueError)
def test_sigma_guesser_target_ratio_different_length():
    sig_method = 'target'
    target = [1,2,3]
    ratio = [0.66,0.4]
    dimension = 3
    checker = init_sigma_checker(sig_method,target,ratio,dimension)

    
#TESTS ON STATE_INIT
@pytest.mark.xfail(raises=ValueError)
def test_state_init_both_energy_and_index():
    energy = np.array([0.1,1.2])
    index= np.array([0,1,2,3])
    dimension = 10
    init_adibatic_state(energy,index,dimension)
    
@pytest.mark.xfail(raises=IndexError)
def test_state_init_index_out_of_range():
    energy = None
    index= np.array([0,1,2,3,10])
    dimension = 10
    init_adibatic_state(energy,index,dimension)
    
@pytest.mark.xfail(raises=IndexError)
def test_state_init_index_out_of_range():
    energy = None
    index= np.array([0,1,2,3,10])
    dimension = 10
    init_adibatic_state(energy,index,dimension)
    
def test_index_based():
    
    energy = None
    index= np.array([0,1,2,3,8])
    dimension = 10
    my_S =init_adibatic_state(energy,index,dimension)
    assert my_S.logic == index_based

def test_energy_based():
    
    energy = np.array([0,1,2,3,8])
    index= None
    dimension = 10
    my_S =init_adibatic_state(energy,index,dimension)
    assert my_S.logic == energy_based
    
def test_all():
    
    energy = None
    index= None
    dimension = 10
    my_S =init_adibatic_state(energy,index,dimension)
    assert my_S.logic == return_all

@pytest.mark.xfail(raises=TypeError)
def test_init_matcher():
    """
    test giving not a string
    """
    algor = 5
    dim = 10
    num_eigs = None
    init_matcher(algor,dim,num_eigs)
    
@pytest.mark.xfail(raises=ValueError)
def test_init_matcher_basic():
    """
    test attempting basic algorithim with less eigenstates than system size.
    """
    algor = 'basic'
    dim = 10
    num_eigs = 7
    init_matcher(algor,dim,num_eigs)

    

    

