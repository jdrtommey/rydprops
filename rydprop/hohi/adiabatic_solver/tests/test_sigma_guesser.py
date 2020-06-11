import pytest
from ..plugins.sigma_guesser import *
from ..adiabatic_state import Adiabatic

@pytest.mark.xfail(raises=KeyError)
def test_key_errror():
    method = 'definetly wrong key'
    target = np.zeros(3)
    ratio =np.zeros(3)
    Sigma_gen(method,target,ratio)
    
def test_correct_key_sigma():
    """
    test correct key results in static function bound to self.func
    """
    method = 'static'
    target = np.zeros(3)
    ratio =np.zeros(3)
    my_sigma = Sigma_gen(method,target,ratio)
    assert my_sigma.func == my_sigma.static
    
def test_correct_key_target():
    """
    test correct key results in static function bound to self.func
    """
    method = 'target'
    target = np.zeros(3)
    ratio =np.zeros(3)
    my_sigma = Sigma_gen(method,target,ratio)
    assert my_sigma.func == my_sigma.target

    
def test_correct_static_function():
    """
    test given a sigma_guesser initialised in static mode
    will return sigma
    """
    method = 'static'
    target = np.zeros(3)
    ratio =np.zeros(3)
    my_sigma = Sigma_gen(method,target,ratio)
    state_list = []
    assert my_sigma(0.0,state_list) == 0.0
def test_correct_static_function1():
    """
    test given a sigma_guesser initialised in static mode
    will return sigma
    """
    method = 'static'
    target = np.zeros(3)
    ratio =np.zeros(3)
    my_sigma = Sigma_gen(method,target,ratio)
    state_list = []
    assert my_sigma(5.1,state_list) == 5.1

    
#tests on the target routine.

my_state_list = []
my_state_list.append(Adiabatic(0,0.0,0.0,np.zeros(10)))
my_state_list.append(Adiabatic(1,1.0,0.0,np.zeros(10)))
my_state_list.append(Adiabatic(2,2.0,0.0,np.zeros(10)))
my_state_list.append(Adiabatic(3,3.0,0.0,np.zeros(10)))


def test_target_get_energy_list_energys():
    """
    check that given a list of adibatic states will find the ones with
    mathching indices and return their energies.
    """
    targets = [0,3]
    ratios=[0.1,0.9]
    energy_list,matching_ratio = target_get_energy_list(my_state_list,targets,ratios)
    
    assert energy_list[0] == 0.0 and energy_list[1] == 3.0
    
def test_target_get_energy_list_switched():
    """
    check that given a list of adibatic states will find the ones with
    mathching indices and return their energies. check it finds in differnet order is state_list switched.
    """
    my_state_list = []
    my_state_list.append(Adiabatic(3,3.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(1,1.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(2,2.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(0,0.0,0.0,np.zeros(10)))
    targets = [3,0]
    ratios=[0.9,0.1]
    energy_list,matching_ratio = target_get_energy_list(my_state_list,targets,ratios)
    
    assert energy_list[0] == 3.0 and energy_list[1] == 0.0
    
def test_target_get_energy_list_switched_ratio():
    """
    check that given a list of adibatic states will find the ones with
    mathching indices and return their energies. check it finds in differnet order is state_list switched.
    """
    my_state_list = []
    my_state_list.append(Adiabatic(3,3.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(1,1.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(2,2.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(0,0.0,0.0,np.zeros(10)))
    targets = [3,0]
    ratios=[0.9,0.1]
    energy_list,matching_ratio = target_get_energy_list(my_state_list,targets,ratios)
    
    assert matching_ratio[0] == 0.9 and matching_ratio[1] == 0.1
    
def test_target_get_average_energy():
    
    energy_list = np.array([1.0,2.0,3.0])
    matching_ratio = np.array([0.2,0.3,0.5])
    ratio_list = np.array([0.2,0.5,0.3])
    average = 1.0*0.2 + 2.0*0.3 + 3.0*0.5
    old_sigma = 0.0
    
    assert target_get_average_energy(energy_list,matching_ratio,ratio_list,old_sigma) == average
    
def test_target_get_average_energy():
    
    energy_list = np.array([1.0,2.0,3.0])
    matching_ratio = np.array([0.2,0.3,0.5])
    ratio_list = np.array([0.2,0.5,0.3,0.1])
    old_sigma = 0.0


    assert target_get_average_energy(energy_list,matching_ratio,ratio_list,old_sigma) == (3.0+1.0)/2.0
    
def test_target_get_average_energy_only_one_found():
    
    energy_list = np.array([1.0])
    matching_ratio = np.array([0.7])
    ratio_list = np.array([0.2,0.5,0.3,0.1])
    old_sigma = 0.5


    assert target_get_average_energy(energy_list,matching_ratio,ratio_list,old_sigma) == 1.0
    
def test_target_get_average_energy_none_found():
    
    energy_list = np.array([])
    matching_ratio = np.array([])
    ratio_list = np.array([0.2,0.5,0.3,0.1])
    old_sigma = 0.5


    assert target_get_average_energy(energy_list,matching_ratio,ratio_list,old_sigma) == 0.5
    
def test_target_logic_():
    
    my_state_list = []
    my_state_list.append(Adiabatic(0,0.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(1,1.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(2,2.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(3,3.0,0.0,np.zeros(10)))
    
    target_list = [3,4,5]
    ratio_list = [0.1,0.3,0.6]
    old_sigma = 0.3
    new_sigma = target_logic(target_list,ratio_list,my_state_list,old_sigma)

    assert new_sigma == 3.0
    
def test_target_logic_none():
    
    my_state_list = []
    my_state_list.append(Adiabatic(0,0.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(1,1.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(2,2.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(7,3.0,0.0,np.zeros(10)))
    
    target_list = [3,4,5]
    ratio_list = [0.1,0.3,0.6]
    old_sigma = 0.3
    new_sigma = target_logic(target_list,ratio_list,my_state_list,old_sigma)

    assert new_sigma == 0.3
    
def test_target_logic_none():
    
    my_state_list = []
    my_state_list.append(Adiabatic(0,7.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(1,8.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(2,2.0,0.0,np.zeros(10)))
    my_state_list.append(Adiabatic(7,3.0,0.0,np.zeros(10)))
    
    target_list = [0,1]
    ratio_list = [0.5,0.5]
    old_sigma = 0.3
    new_sigma = target_logic(target_list,ratio_list,my_state_list,old_sigma)

    assert new_sigma == 7.5