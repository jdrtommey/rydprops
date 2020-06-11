from ..plugins.state_init import *
import pytest
import numpy as np
State_initialiser
    
def test_index_based():
    method ='index'
    input_list = np.zeros(4)
    
    my_S = State_initialiser(method,input_list)
    assert my_S.logic == index_based

def test_energy_based():
    method ='energy'
    input_list = np.zeros(4)
    
    my_S = State_initialiser(method,input_list)
    assert my_S.logic == energy_based
    
def test_all():
    method ='all'
    input_list = np.zeros(4)
    
    my_S = State_initialiser(method,input_list)
    assert my_S.logic == return_all
    
@pytest.mark.xfail(raises=KeyError)
def test_all():
    method ='foo_nafonew'
    input_list = np.zeros(4)
    
    my_S = State_initialiser(method,input_list)
    
def test_get_diags():
    method ='all'
    input_list = np.zeros(4)
    
    my_S = State_initialiser(method,input_list)
    ham = np.ones((3,3))
    diags = my_S.get_diagonals(ham)
    assert diags.all() == 1
    
@pytest.mark.xfail(raises=TypeError)
def test_get_diags():
    method ='all'
    input_list = np.zeros(4)
    
    my_S = State_initialiser(method,input_list)
    ham = 'not a matrix'
    diags = my_S.get_diagonals(ham)
    assert diags.all() == 1
    
    
#test the add state function

def test_add_states_all():
    method ='all'
    input_list = None
    my_matrix = np.diag([1.0,2.0,3.0,4.0,5.0])
    my_S = State_initialiser(method,input_list)
    states = my_S(my_matrix)
    assert states[0].vals[0] == 1.0 and states[2].vals[0] == 3.0
    
def test_add_states_energy():
    method ='energy'
    input_list = [1.4,3.6]
    my_matrix = np.diag([1.0,2.0,3.0,4.0,5.0])
    my_S = State_initialiser(method,input_list)
    states = my_S(my_matrix)
    assert states[0].vals[0] == 2.0 and states[1].vals[0] == 3.0

def test_add_states_energy1():
    method ='energy'
    input_list = [1.4,4.6]
    my_matrix = np.diag([1.0,2.0,3.0,4.0,5.0])
    my_S = State_initialiser(method,input_list)
    states = my_S(my_matrix)
    assert len(states) == 3
    
def test_add_states_index():
    method ='index'
    input_list = [1,2,3]
    my_matrix = np.diag([1.0,2.0,3.0,4.0,5.0])
    my_S = State_initialiser(method,input_list)
    states = my_S(my_matrix)
    assert len(states) == 3
    
def test_add_states_index1():
    method ='index'
    input_list = [1,2,4]
    my_matrix = np.diag([1.0,2.0,3.0,4.0,5.0])
    my_S = State_initialiser(method,input_list)
    states = my_S(my_matrix)
    assert states[0].vals[0] == 2.0 and states[2].vals[0] == 5.0
    

# test the option functions.
def test_energy_based_true():
    input_list = np.array([-0.5,0.5])
    index = 'foo'
    value = 0.2
    assert energy_based(index,value,input_list) == True
    
def test_energy_based_true1():
    input_list = np.array([-0.5,0.5])
    index = 'foo'
    value = 0.5
    assert energy_based(index,value,input_list) == True
    
def test_energy_based_false():
    input_list = np.array([-0.5,0.5])
    index = 'foo'
    value = 0.51
    assert energy_based(index,value,input_list) == False

    
def test_energy_based_false1():
    input_list = np.array([-0.5,0.5])
    index = 'foo'
    value = -0.501
    assert energy_based(index,value,input_list) == False
    
def test_index_based_true():
    input_list = np.array([0,1,2,3,4,5])
    index = 0
    value = 'foo'
    assert index_based(index,value,input_list) == True
    
def test_index_based_True1():
    input_list = np.array([1])
    index = 1
    value = 'foo'
    assert index_based(index,value,input_list) == True
    
def test_index_based_False():
    input_list = np.array([0,1,2,3,4,5])
    index = 6
    value = 'foo'
    assert index_based(index,value,input_list) == False
    
def test_index_based_False1():
    input_list = np.array([3])
    index = 2
    value = 'foo'
    assert index_based(index,value,input_list) == False
    