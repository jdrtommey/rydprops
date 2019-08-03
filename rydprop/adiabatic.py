from .state import State_nlm
from scipy.sparse import coo_matrix,csr_matrix,diags
import numpy as np
from tqdm import tqdm
"""
module which creates an object based around a 'space' which contains the evolution of 
an eigenvector and eigenvalue with a field parameter.
"""

class Adiabatic:
    def __init__(self,space,vals,vecs,parameters):
        """
        Object which holds an adibataic state as a function of a static field parameter. 
        can locate states 
        """
        self.space = space   #space has the field-free states which make up this adiabatic state
        self.vals = vals
        self.vecs = vecs
        self.parameters = parameters
        
            
    def __len__(self):
        return len(self.space)
    
    def value(self,parameter):
        """
        interpolates the eigen energy of this state
        """     
        return np.interp(parameter,self.parameters,self.vals)
        
    def vector(self,parameter):
        """
        interpolates the eigenvector at a given parameter
        """
        vector = np.zeros(len(self))   #generate new vector of correct length
        
        for i in range(len(vector)):
            vector[i] = np.interp(parameter,self.parameters,self.vecs[:,i])
        
        return vector

    def index(self,state):
        """
        returns the index of a given field-free state
        """
        return self.space.index(state)
    
    def state(self,index):
        """
        given an index will return the field-free state object
        """
        return self.space.states[index]
    
    def coefficient(self,state,parameter):
        """
        returns the coeffieicent of a given state at a given parameter
        """
        vecs = self.vector(parameter)
        index = self.index(state)
        return vecs[index]
    
    
# SINGLE STATE FUNCTIONS

def character(adi_state1,state,parameter):
    """
    computes how much state 'character' the adiabatic state has at a given parameter value.
    """    
    return adi_state1.coefficient(state,parameter)
    

def transition_dipole_ground(adi_state1,ground_state,parameter,parallel):
    """
    computes the transition dipole moment between a state and the ground state. in units of ea0.
    """
    
    vector = adi_state1.vector(parameter)
    
    transition = 0.0
    
    for i in len(adi_state1):
        coeff = vector[i]
        state = adi_state1.state(i)
        
        transition = transition + (coeff*atom.matrix_element(state,ground_state,parallel))
            
    return transition
def polarizability(adi_state1,parameter):
    """
    gets the gradient of the state at a given parameter
    """
    
    foo = 0.0 
    
def angular_wf(adi_state1,parameter):
    """
    returns the angular wavefunction of a given state evaluated at a given
    parameter value
    """
    
    foo = 0.0
    
# FUNCTIONS FOR GETTING EVOLUTION OF THE TRANSITION DIPOLE MOMENT BETWEEN TWO STATES. CURRENTLY REQUIRES BOTH TO HAVE THE SAME
# SPACE.

def get_dipole_coupling_matrix(adi_state1,adi_state2,parallel):
    """
    computes the dipole coupling matrix between two spaces.
    
    Parameters
    ----------
    adi_state1: adiabatic state class
        first state
    adi_state2: adiabatic state class
        second state
    para: boolean
        if the field is parallel to the space.
        
    returns
    -------
    dipole_matrix: coomatrix array
        a matrix with matrix[i][j] corresponding to the 
        dipole matrix element between state i in adi_state1 space and 
        state j in adi_state2 space.
    """
    
    if parallel == True:
        selection_rules={'dl':1,'dml':0}
    else:
        selection_rules={'dl':1,'dml':1}
    
    space1_index_list=[]
    space2_index_list=[]
    value=[]
    atom =adi_state1.space.atom
    for i in range(len(adi_state1.space)):
        for j in range(len(adi_state2.space)):
            
            state1 = adi_state1.space[i]
            state2 = adi_state2.space[j]
            dn = abs(state1.n - state2.n)
            dl = abs(state1.l - state2.l)
            dml = abs(state1.ml-state2.ml)
            
            if dl == selection_rules['dl'] and dml ==selection_rules['dml']:
                matrix_value = atom.matrix_element(state1,state2,parallel)
                if matrix_value != 0.0:
                    space1_index_list.append(i)
                    space2_index_list.append(j)
                    value.append(matrix_value)
    dipole_matrix = coo_matrix( (value,(space1_index_list,space2_index_list)),\
                               shape = (len(adi_state1.space),len(adi_state2.space)))
    print(np.sum(dipole_matrix.toarray()[0][:]))
    return dipole_matrix

def _get_transition_dipole(adi_state1,adi_state2,coupling_matrix,param):
    """
    given two adiabatic states a coupling matrix and a parameter value will compute
    the transition moment
    """
    
    state1_vector = adi_state1.vector(param)
    state2_vector = adi_state2.vector(param)
    
    dipole = 0.0
    
    for i,j,v in zip(coupling_matrix.row, coupling_matrix.col, coupling_matrix.data):
        coeff1 = state1_vector[i]
        coeff2 = state2_vector[j]
        dipole = dipole + coeff1*coeff2*v

    return dipole

def adibatic_transition_dipole(adi_state1,adi_state2,para,coupling_matrix = None):
    """
    computes the new transition dipole at each of the calculated points
    
    Parameters
    ----------
    adi_state1: Adiabatic state object
        the first state
    adi_state2: Adiabatic state object
        the second state
    para: boolean
        the direction of the field
    coupling_matrix: matrix. optional
        for efficiency can provide the coupling matrix between the 
        two spaces, which records the transition dipole moments between
        all states in each space.
        
    Returns
    -------
    adiabatic_transition_dipoles: numpy array
        an array of the transition dipole moment between the two states 
        as for each parameter value contained in adi_state1.
        
    """
    adiabatic_transition_dipoles = np.zeros(len(adi_state1.vals))
    
    if coupling_matrix == None:
        coupling_matrix = get_dipole_coupling_matrix(adi_state1,adi_state2,para)
    
    for i in tqdm(range(len(adiabatic_transition_dipoles))):
        param = adi_state1.parameters[i]
        adiabatic_transition_dipoles[i] = _get_transition_dipole(adi_state1,adi_state2,coupling_matrix,param)
    return adiabatic_transition_dipoles
    
# FUNCTIONS FOR THE ENERGY DIFFERENCE BETWEEN TWO ADIABATIC STATES

def transition_energy(adi_state1,adi_state2,param):
    """
    return the transition energy between two states.
    """
    
    return adi_state1.value(param) - adi_state2.value(param)

    
    