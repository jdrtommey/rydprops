from .state import State_nlm
from scipy.sparse import coo_matrix,csr_matrix,diags
import numpy as np
from hohi import floquet_matrix,floquet_basis,Adiabatic_solver
from .interaction import interaction
from . import ureg
from . adiabatic import Adiabatic
"""
module to take a space of rydberg states and convert it into a floquet hamiltonian. set-up this way
sop that an efficient algorithim in hohi floquet_matrix can generate the floquet hamiltonains
without the need to compute the dipole moments between states more than once.Floquet_space
is a class which contains the floquet matrices and a way to locate floquet states based
on the Space class provided.
"""

class Floquet_space:
    @ureg.wraps(None,(None,None,ureg('hartree'),None,None,None))
    def __init__(self,space,freq,q_bands,parallel=True,h1_static=None):
        """
        takes a state space and an interaction hamiltonian and generates a new floquet hamiltonian. The 
        floquet interaction is an oscillating electric field.
    
        Parameters
        ----------
        space: space 
            the set of states which are being used
        freq: float
            the frequency of the driving oscillation. Provide in GHz.
        parallel: boolean
            whether the oscillating electric field is parallel or perpendicular.
        h1_static: sparse/numpy array
            an additional static interaction which will remain constant
        """
        self.space = space
        self.parallel = parallel
        self.h1_static = h1_static
        self.freq = freq
        self.q_bands = q_bands
    
        self.h0,self.h1 = self._hamiltonian()
    
    def locate(self,state,q):
        """
        given a state and a floquet sideband number will determine location in the floquet hamiltonian    
        """
        index = self.space.index(state)
        floq_index = floquet_basis(index,q,len(self.space),self.q_bands)
    
        return floq_index
    
    def energy(self,state,q):
        """
        returns the diagonal element for this state
        """
        index = self.locate(state,q)
        return self.h0.diagonal()[index]

    def _floquet_interaction(self):
        """
        generates the floquet interaction. Same as a static electric interaction hamiltonian 
        except for a factor of 1/2 to account for time averaging.
        """
    
        return interaction(self.space,'osc',self.parallel)
                
    def _hamiltonian(self):
        """
        generate the h0 and h1 hamiltonians in the floquet picture.
        """
    
        h0 = self.space.H0()
        if self.h1_static != None:
            try:
                h0= h0 + h1_static
            except:
                raise ValueError("could not combine the h0 and static interaction")
            
        h1 = self._floquet_interaction()
    
        h0_floq,h1_floq = floquet_matrix(h0,h1,self.q_bands,self.freq)
        return h0_floq,h1_floq
    
    
    
@ureg.wraps(None,(None,ureg('au_e'),None))
def floquet_solver_range(floq_space,field_vals,q_bands):
    """
    function which takes a floquet space and computes the adiabatic states from it. provide the 
    number of q_bands as a range of adibatic states to track. e.g only normally interested in the 
    states nearest q=0
    """
        
    h0 = floq_space.h0
    h1 = floq_space.h1
    parameters = field_vals
    sigma = (floq_space.energy(floq_space.space[0],0)+floq_space.energy(floq_space.space[1],0))/2
    
    num_eigs = (q_bands*2 +1)*len(floq_space.space)
    algorithim='vec'
    #energy list has the smallest eigenvalue of the lowest sideband to the largest value of the highest
    energy_list = [floq_space.energy(floq_space.space[0],-q_bands) , floq_space.energy(floq_space.space[-1],q_bands)]

    solver = Adiabatic_solver(h0,h1,parameters,sigma,num_eigs = num_eigs,algorithim=algorithim,\
                           energy_list=energy_list)
    states = solver.run()
    
    adiabatic_states= []
    for s in states:
        adiabatic_states.append(Adiabatic(floq_space.space,s.vals,s.vecs,s.parameter))
        
    return adiabatic_states


        
