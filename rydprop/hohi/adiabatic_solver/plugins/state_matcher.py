from scipy.sparse.linalg import eigs
from numpy.linalg import eigh
import numpy as np
from numba import jit

class Matcher:
    def __init__(self,method):
        """
        class which given a set of eigen_vals and eigen_vecs can match these to the correct
        adiabatic states. Contains set of default methods.
        
        General idea: select an algorithim, this compares the eigenvalue/vector against all the 
        adiabatic states, if it finds a match will add that val/vec to the adiabatic state.
        A lengh_checker then looks at the lengths of adibatic state, and either puts it in a
        successful list if it has acquired an additional state, or a failed list if not.
        
        Parameters
        ----------
        method: string
            chose which matching algorithim to apply
            
        """        
        self.available_methods = {'vec':vector_algorithim,'basic':basic_algorithim,'energy':energy_algorithim,\
                                 'circ':circular_algorithim}    #maintains a list of the current methods which have been written
        try:
            self.func = self.available_methods[method]        
        except:
            raise KeyError("Not avaivable method, current dictionary of avaiable methods: " + str(self.available_methods.keys()))
        
    def __call__(self,state_list,vals,vecs,param):   
        current_length = current_length_check(state_list)        #at start of process check length of the adibatic states.
        state_list = self.func(state_list,vals,vecs,param)  #computes the changes to the adibatic states 
        success,fail = length_checker(state_list,current_length)   #checks which ones were successfully paired. 
        return success,fail
    
def current_length_check(state_list):
    """
    takes a list of adibataic states and confirms the current length
    """
    length = state_list[0].get_length()
    for state in state_list:
        if state.get_length() != length:
            raise RuntimeError("mismatch in adibataic state length, internal error")
            
    return length
        
def length_checker(state_list,current_length):
    """
    checks the lenght of every adiabatic state. returns two lists:
    the successful and the failed matchings
    """
    successful_list =[]
    failed_list=[]
    for state in state_list:
        if state.get_length() == current_length+1:
            successful_list.append(state)
        elif state.get_length() != current_length+1:
            failed_list.append(state)
    return successful_list,failed_list

#algorithim functions   

#########################
#                       #
#    vector algoithim   #
#                       # 
#########################

    
@jit
def vector_algorithim(state_list,vals,vecs,param,x=0.01):
    """
    For each state in the current set of adiabatic states, computes a range of x% around the current eigenenergy 
    and coefficient of its initial state, searches all the eigenvals and vecs stored in vals and vecs
    if it finds a state which if within the bounds of both value and coeffieicnt adds
    it too a candidate state list. If this list only has a single entry it adds this vector to the 
    adiaabatic state and places the vector in a taken list to stop it being compared to future 
    states.
    """
    taken_list = [] #stores the index of vals which have been assigned
    for state in state_list:
        candidate_list =[]
        predicted_energy = state.get_current_value()
        upperbound_energy = predicted_energy * (1+x)
        lowerbound_energy = predicted_energy * (1-x)
        energy_range = [lowerbound_energy,upperbound_energy]
            
        predicted_coeff = state.get_current_coefficient()            
        upperbound_coeff = abs(predicted_coeff) * (1+x)
        lowerbound_coeff = abs(predicted_coeff) * (1-x)
        coeff_range = [upperbound_coeff,lowerbound_coeff]
            
        for i,val in enumerate(vals):
            if i not in taken_list: 
                vec_coeff = abs(vecs[state.index,i])
                if val < np.max(energy_range) and val > np.min(energy_range):
                    if vec_coeff < np.max(coeff_range) and vec_coeff > np.min(coeff_range):
                            candidate_list.append(i)  

        if len(candidate_list) == 1:
            vec_index = candidate_list[0]
            state.add(vals[vec_index],vecs[:,vec_index],param)
            taken_list.append(vec_index)
        elif len(candidate_list) > 1:
            vec_index = candidate_list[0]
            state.add(vals[vec_index],vecs[:,vec_index],param)
            taken_list.append(vec_index)
    return state_list
                
#########################
#                       #
#    basic  algoithim   #
#                       # 
#                       #
#########################
    
def basic_algorithim(state_list,vals,vecs,param):
    """
    Just assigns the largest val to the largest vec in the list. stops when it
    reaches the last vec, only really useful when computing the dense space and
    not concerned about exact crossings.
    """
    #loop through and append it to the next in state_list
    
    for i,state in enumerate(state_list):
        try:
            state_list[i].add(vals[i],vecs[:,i],param)
        except:
            raise IndexError("Index error assigning  eigenvalue index "+ str(i) + " to adibatic state")
        
    return state_list

#########################
#                       #
#    basic  algoithim   #
#                       # 
#                       #
#########################


@jit
def energy_algorithim(state_list,vals,vecs,param,x=0.05):
    """
    For each state in the current set of adiabatic states, computes a range of x% around the current eigenenergy 
    and coefficient of its initial state, searches all the eigenvals and vecs stored in vals and vecs
    if it finds a state which if within the bounds of both value and coeffieicnt adds
    it too a candidate state list. If this list only has a single entry it adds this vector to the 
    adiaabatic state and places the vector in a taken list to stop it being compared to future 
    states.
    """
    taken_list = [] #stores the index of vals which have been assigned
    for state in state_list:
        candidate_list =[]
        predicted_energy = state.get_current_value()
        upperbound_energy = predicted_energy * (1+x)
        lowerbound_energy = predicted_energy * (1-x)
        energy_range = [lowerbound_energy,upperbound_energy]

            
        for i,val in enumerate(vals):
            if i not in taken_list: 
                vec_coeff = abs(vecs[state.index,i])
                if val < np.max(energy_range) and val > np.min(energy_range):
                            candidate_list.append(i)  

        if len(candidate_list) > 0:
            vec_index = candidate_list[0]
            state.add(vals[vec_index],vecs[:,vec_index],param)
            taken_list.append(vec_index)

                
    return state_list


#########################
#                       #
# circular  algoithim   #
#                       # 
#                       #
#########################

@jit 
def circular_algorithim(state_list,vals,vecs,param):
    """
    alogorithim specifically written to find circular states, where the 
    s state above is also being calculated.
    Assumes the largest returned eigenstate is always a given state.s
    """
    
    idx = vals.argsort()  #first sort the returned eigens from biggest to largest
    eigenValues = vals[idx]
    eigenVectors = vecs[:,idx]
    
    for i,state in enumerate(state_list):
        state.add(vals[i],vecs[:,i],param)
        
    return state_list