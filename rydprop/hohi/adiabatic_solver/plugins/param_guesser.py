from scipy.sparse.linalg import eigs
from numpy.linalg import eigh
import numpy as np

class Param_gen:
    def __init__(self,method,p_max,num_iters,max_fails = 5, min_states = 1):
        """
        Determines what the next parameter should be.  
        
        Parameters
        ----------
        method: string
            chose which matching algorithim to apply
            
        """          
        self.available_methods = {'constant':self.constant_step,'variable':self.variable_step}  
        try:
            self.func = self.available_methods[method]        
        except:
            raise KeyError(method + " is not an avaivable method, current dictionary of avaiable methods: " + str(self.available_methods.keys()))
            
        self.min_states = min_states
        self.p_max = p_max
        self.step = p_max/num_iters
        self.max_fails = max_fails
        self.current_fails = 0
        
    def __call__(self,state_list,param):   
            param = self.func(state_list,param)
            return param
    
    def constant_step(self,state_list,param):
    
        return param + self.step
    
    def variable_step(self,state_list,param):
        """
        if fails to find a adiabatic state it will make the step size half as big and log the fact that
        it failed. if fails self.max_fails in a row it will raise an error.
        """
        if len(state_list) < self.min_states :
            print(len(state_list),self.step)
            self.step = self.step/2
            param = param - (self.step)
            self.current_fails = self.current_fails + 1
            
            if self.current_fails == self.max_fails:
                raise RuntimeError("reached maximum number of failed attempts " + str(self.max_fails))
            
        else:
            param = param + self.step
            self.current_fails = 0
            
        return param
    
    

    