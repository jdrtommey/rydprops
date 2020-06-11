from ..plugins.state_matcher import *
from ..adiabatic_state import Adiabatic
import pytest
import numpy as np
    
def test_assign_vector_algorithim():
    match = 'vec'
    my_matcher = Matcher('vec')
    assert my_matcher.func == vector_algorithim
    
def test_assign_basic_algorithim():
    match = 'basic'
    my_matcher = Matcher('basic')
    assert my_matcher.func == basic_algorithim

@pytest.mark.xfail(raises=KeyError)
def test_assign_wrong():
    """
    check raises error if given an incorrect key
    """
    match = 'frweofw'
    my_matcher = Matcher('wfihewf')
    
# tests on length_checker function.

def test_length_checker():
    #list of adiabatic states
    my_state_list = []
    state1 = Adiabatic(0,3.0,0.0,np.zeros(10))
    state1.add(0,np.zeros(10),4)
    state2 = Adiabatic(0,3.0,0.0,np.zeros(10))
    state2.add(0,np.zeros(10),4) 
    state3 = Adiabatic(0,3.0,0.0,np.zeros(10))

    my_state_list.append(state1)
    my_state_list.append(state2)
    my_state_list.append(state3)

    suc,fail = length_checker(my_state_list,1)
    #have two states which are of length 2 and one of length 1
    #shold pass two and fail the third
    
    assert len(suc) == 2 and len(fail) == 1
    
def test_length_checker1():
    #list of adiabatic states
    my_state_list = []
    state1 = Adiabatic(0,3.0,0.0,np.zeros(10))
    state1.add(0,np.zeros(10),4)
    state2 = Adiabatic(0,3.0,0.0,np.zeros(10))
    state2.add(0,np.zeros(10),4) 
    state3 = Adiabatic(0,3.0,0.0,np.zeros(10))

    my_state_list.append(state1)
    my_state_list.append(state2)
    my_state_list.append(state3)

    suc,fail = length_checker(my_state_list,1)
    #have two states which are of length 2 and one of length 1
    #shold pass two and fail the third
    
    assert suc[0] == state1 and fail[0] == state3
    
@pytest.mark.xfail(raises=RuntimeError)
def test_current_checker1():
    #list of adiabatic states
    my_state_list = []
    state1 = Adiabatic(0,3.0,0.0,np.zeros(10))
    state1.add(0,np.zeros(10),4)
    state2 = Adiabatic(0,3.0,0.0,np.zeros(10))
    state2.add(0,np.zeros(10),4) 
    state3 = Adiabatic(0,3.0,0.0,np.zeros(10))

    my_state_list.append(state1)
    my_state_list.append(state2)
    my_state_list.append(state3)

    length = current_length_check(my_state_list)
    
def test_current_checker2():
    #list of adiabatic states
    my_state_list = []
    state1 = Adiabatic(0,3.0,0.0,np.zeros(10))
    state1.add(0,np.zeros(10),4)
    state2 = Adiabatic(0,3.0,0.0,np.zeros(10))
    state2.add(0,np.zeros(10),4) 
    state3 = Adiabatic(0,3.0,0.0,np.zeros(10))
    state3.add(0,np.zeros(10),4) 


    my_state_list.append(state1)
    my_state_list.append(state2)
    my_state_list.append(state3)

    length = current_length_check(my_state_list)
    assert length == 2