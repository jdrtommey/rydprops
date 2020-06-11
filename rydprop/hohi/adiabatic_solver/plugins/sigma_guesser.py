from scipy.sparse.linalg import eigs
from numpy.linalg import eigh
import numpy as np

class Sigma_gen:
    def __init__(self,method,target,ratio):
        """
        class which determines where the 'sigma' guess should move for the next iteration
        of the program. If 'target' is chosen must give a target index which will be tracked. 
        
        Parameters
        ----------
        method: string
            chose which matching algorithim to apply
            
        """          
        self.available_methods = {'static':self.static,'target':self.target}  
        try:
            self.func = self.available_methods[method]        
        except:
            raise KeyError(method + " is not an avaivable method, current dictionary of avaiable methods: " + str(self.available_methods.keys()))
            
        self.target_list = target
        self.ratio_list = ratio
        
    def __call__(self,sigma,state_list):   
            sigma = self.func(sigma,state_list)
            return sigma
    
    def static(self,sigma,state_list):
    
        return sigma
    
    def target(self,sigma,state_list):
            
        new_sigma = target_logic(self.target_list,self.ratio_list,state_list,sigma)
        
        return new_sigma
    
    
# FUNCTIONS TO HELP COMPUTE THE TARGET ENERGY   

def target_logic(target_list,ratio_list,state_list,old_sigma):
    """
    takes the target adiabatic states and the current tracked states. If all the target states are present will 
    compute the sigma based on ratios, if not all the states are present will take the average of the largest and
    smallest adiabatic eigenenergies still being tracked. If all the state are lost will leave the sigma static.
    """
    
    energy_list,index_list = target_get_energy_list(state_list,target_list,ratio_list)
    sigma = target_get_average_energy(energy_list,index_list,ratio_list,old_sigma)
    
    return sigma

    
def target_get_energy_list(state_list,target_list,ratio_list):
    """
    given list of adibatic states and a index list will return the current energies of the states, and
    a matching list of their indices
    """
    found_list = []      #the found states
    founds_ratio = []   #the ratio beloning to this state
    
    for j,index in enumerate(target_list):
        for state in state_list:
             if state.index ==index:
                found_list.append(state)
                founds_ratio.append(ratio_list[j])
                
                
    energy_list = []
    for found_state in found_list:
        energy_list.append(found_state.get_current_value())
    return np.asarray(energy_list),np.asarray(founds_ratio)  
    
def target_get_average_energy(energy_list,matching_ratio,ratio_list,old_sigma):
    """
    helper function for target method. takes a list of energies and a list of ratios
    and returns the weighted average. takes two numpy arrays
    """
    
    if len(energy_list) == len(ratio_list):   
        sigma = np.sum(energy_list * matching_ratio)
    elif len(energy_list)>0:
        sigma = (np.max(energy_list) + np.min(energy_list)) / 2.0
    else:
        sigma = old_sigma
    
    return sigma


    