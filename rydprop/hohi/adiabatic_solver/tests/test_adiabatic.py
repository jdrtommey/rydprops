import pytest
from ..adiabatic_state import Adiabatic
import warnings
import numpy as np
from numpy.linalg import eigh,eig

tester = np.zeros((3,3))
tester[0][0] = 0
tester[0][1] = 1
tester[0][2] = 2
tester[1][0] = 3
tester[1][1] = 4
tester[1][2] = 5
tester[2][0] = 6
tester[2][1] = 7
tester[2][2] = 8  
initial_field = 5
initial_parameter = 0.0
initial_vector= tester[:,0]
vals,vecs = eig(tester)
index = 1

def test_initialise_index():
    """
    correctly initial index
    """
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    assert adi.index == 1
    
def test_initialise_field():
    """
    correctly initial field
    """    
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    assert adi.vals[0] == 5
    
def test_initialise_parameter():
    """
    correctly initial parameter
    """
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    assert adi.parameter[0] == 0.0
    
def test_intialise_vector():
    """
    test adding initial vector
    """
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    assert adi.vecs[0][0] == 0 and adi.vecs[0][1] == 3 and adi.vecs[0][2] ==6
    
def test_add_vector():
    
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    adi._add_vector(tester[:,1])
    assert adi.vecs[1][0] == 1 and adi.vecs[1][1] == 4 and adi.vecs[1][2] == 7
    
def test_add_vector_numpy_output():
    """
    vectors added via eigenstates of diagonalisation 
    """
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    adi._add_vector(vecs[:,0])
    assert adi.vecs[1][0] == vecs[0][0] and adi.vecs[1][1] == vecs[1][0] and adi.vecs[1][2] == vecs[2][0]
    
def test_add_val_numpy_output():
    """
    vectors added via eigenstates of diagonalisation 
    """
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    adi._add_value(vals[0])
    adi._add_value(vals[1])
    assert adi.vals[1] == vals[0] and adi.vals[2] ==vals[1] 
   
@pytest.mark.xfail(raises=AttributeError)
def test_add_wrong_length_vector():
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    wrong_vec = np.zeros(2)
    adi._add_vector(wrong_vec)
    
def test_user_add():
    """
    test the user facing add function
    """
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    adi.add(vals[0],vecs[:,0],0.1)
    assert adi.vals[1] == vals[0] and adi.vecs[1][1] == vecs[1,0] and adi.parameter[1] == 0.1 and adi.get_length()==2
    
    
def test_length_initial():
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    assert adi.get_length() == 1
    
def test_length_3():
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    adi._add_parameter(0)
    adi._add_parameter(1)
    assert adi.get_length() == 3
    
    
def test_current_val():
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    adi.add(vals[0],vecs[:,0],0.1)
    assert adi.get_current_value() == vals[0]

        
def test_current_coeff():
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    adi.add(vals[0],vecs[:,0],0.1)
    assert adi.get_current_coefficient() == vecs[adi.index,0]
    
def test_current_parameter():
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    adi.add(vals[0],vecs[:,0],0.1)
    assert adi.get_current_parameter() == adi.parameter[1]

def test_current_coeff_other():
    """
    return coefficient other than own index
    """
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    adi.add(vals[0],vecs[:,0],0.1)
    assert adi.get_current_coefficient(2) == vecs[2,0]
    
@pytest.mark.xfail(raises=IndexError)
def test_current_coeff_index_too_big():
    """
    return coefficient other than own index
    """
    adi = Adiabatic(index,initial_field,initial_parameter,initial_vector)
    adi.add(vals[0],vecs[:,0],0.1)
    adi.get_current_coefficient(5)