from ..adiabatic_state import Adiabatic
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix,isspmatrix
from numpy.linalg import eigh
import numpy as np


class State_initialiser:
    def __init__(self,method,input_list):
        """
        generates a list of adiabatic states to track. WORKS ON ASSUMPTION H0 is diagonal and parameter
        starts at 0. Can either initialise a set of adiabatic states within an energy list, by providing specfic
        index numbers, or just initialise all possible states.
        
        Parameters
        ----------
        energy_list: array default=None
            contains a lower and upper initial eigenvalue list 
        index_list: list default=None
            contains a list of indices for which to initialise states.
        """
        self.input_list = input_list

        self.available_methods = {'index':index_based,\
                                  'energy':energy_based,\
                                  'all':return_all}    #maintains a list of the current ways to initialise
        try:
            self.logic = self.available_methods[method]        
        except:
            raise KeyError("Not avaivable method, current dictionary of avaiable methods: " + str(self.available_methods.keys()))
        

        #sets the logic switch for what states to generate
         
    def __call__(self,H0):
        """
        takes the diagonal of a matrix and uses these to determine if an adibatic state should
        be gnerated for for eigenvalue
        """
        diags = self.get_diagonals(H0)
        return self.add_states(diags)
    
    def get_diagonals(self,H0):
        """
        strips the diagonals from a matrix
        """
        try:
            diags = H0.diagonal()
        except:
            raise TypeError("Could not take diagonal of matrix.")
        return diags
    
    def add_states(self,diagonals):
        """
        loops through the states and initialises an adiabatic state.
        called a logic question to determine if state should be generated.
        """
        state_list=[]
        for index,value in enumerate(diagonals):
            if self.logic(index,value,self.input_list) == True:
                vector = np.zeros(len(diagonals))
                vector[index] = 1.0
                parameter = 0.0
                state_list.append(Adiabatic(index,value,parameter,vector))                
        return state_list    
    
def energy_based(index,value,input_list): 
    """
    if the value is inside the energy list create a adiabatic state.
    """
    if value >= np.min(input_list) and value <= np.max(input_list):
        return True
    else:
        return False
                                                 
def index_based(index,value,input_list):  
    """
    only create a state if the eigenvalue is in the index list.
    """
        
    if index in input_list:
        return True
    else:
        return False
                                                                    
def return_all(index,value,input_list):
    """
    create an adiabatic state for all initial eigenstates
    """
    return True