from .state import State_nlm
from scipy.sparse import coo_matrix,csr_matrix,diags
import numpy as np

class Space:
    def __init__(self,atom):
        """
        wrapper around a python list for use with state objects. A list which can only hold single
        type of object. Index returns an index if found and None otherwise. Can only add a single
        instance of a state to the space, ie n=10,l=0,ml=0 can only be added once. 
        """
        self.atom = atom
        self.basis_type = self.atom.basis
        self.states = []
        
    def __len__(self):
        return len(self.states)
    
    
    def __getitem__(self,key):
        try:
            return self.states[key]
        except:
            raise IndexError("Index "+str(key) +" could not be found in space of dimension" + len(self))
            
    def __iter__(self):
        return self.states.__iter__()
    
    def append(self,state):
        """
        try to add to list, checks that the 
        """

        if type(state) == self.basis_type:
            if self.index(state) == None:
                self.states.append(state)
            else:
                raise ValueError("Cannot add duplicate states to space.")
        else:
            raise TypeError("Cannot add state basis type " +str(type(state)) + "to space of basis " + str(self.basis_type) )
        
    def index(self,test_state):
        """
        locate the index of a state in the list. loops through the states and checks if the state has a match
        """
        try:
            return self.states.index(test_state)
        except:
            return None
        
    def H0(self):
        """
        returns the field-free hamiltonian for the space.
        """
        energies = []
        for state in self.states:
            energies.append(self.atom.energy(state))
        
        return diags(energies,format='coo')

            
            
            