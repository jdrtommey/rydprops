"""
Basic class to compute atom properties. Using GHz as the energy. 
"""
from .numerov import radial_overlap
from .angular import angular_overlap_analytical
from .state import basis_options

import scipy.constants as consts
from numba import jit
m_e = consts.physical_constants['electron mass'][0]
R_0 = consts.physical_constants['Rydberg constant'][0]
En_h =  consts.physical_constants['Hartree energy'][0]
a_0 =  consts.physical_constants['Bohr radius'][0]
m_atomic = consts.physical_constants['atomic mass constant'][0]
c = consts.physical_constants['speed of light in vacuum'][0]
h = consts.physical_constants['Planck constant'][0]
e = consts.physical_constants['elementary charge'][0]



class RydbergAtom:
    def __init__(self,mass = 1.00794,defects={},basis = 'nlm',additional_states = None,energy_units='ghz'):
        """
        Base class for atom, needs mass and defects to be able to compute all the properties of Rydberg states.
        Routines not accurate at low states so if try and pass a state with a low n will first try and find
        it in the additional_state dictionary.
        
        Parameters
        ----------
        
        mass: float
            mass of the particle in au.
        defects:
            as a nested dictionary in format {0 : {0:[x,y,z]}, 1: {0:[a,b,c], 2:[d,e,f] }}
        additional_states: dictionary
            contain energies of additional states at low energies in n,l,ml {n: {l:{ml:energy}   }  }
        """
        self.mass = mass * m_atomic
        self.defects = defects
        self.additional_states = additional_states
        
        # the basis in which calculations are to be made. feature to be extended to spin-orbit basis
        if basis in basis_options:
            self.basis = basis_options[basis]
        else:
            raise KeyError("couldnt find the desired basis option, choose from "+str(basis_options))
        
        #compute the scaled rydberg constant
        self.mass_core = self.mass - m_e
        self.reduced_mass = (self.mass_core*m_e)/ (self.mass)
        self.scalefactor = (self.reduced_mass/m_e)        
        self.rydberg_const = R_0 * self.scalefactor
        self.bohr = a_0 / self.scalefactor
        self.hartree= 2.0*self.rydberg_const*h*c
        
        #Basic energies computed in the scaled Rydberg units of a specific atom
        #this dictionary with the reduced units allows for conversion to other types.
                
    def n_eff(self,n,l):
        """
        returns the effective n for a given basis
        """
        defect = get_defect(self.defects,n,l)
        return n - defect
    
    @jit
    def energy(self,n,l,ml):
        """
        gets the energy of a state in units of H'_e, convert to the non-scaled version H_e.
        """
        neff = self.n_eff(n,l)
        return energy_au(neff) /self.scalefactor

    @jit
    def radial_overlap(self,n1,l1,n2,l2,order=1.0):
        """
        compute the radial overlap of two states. returns in units of a'0, convert to non-scaled a0
        """
        n1_eff = self.n_eff(n1,l1)
        n2_eff = self.n_eff(n2,l2)
        
        return radial_overlap(n1_eff,l1,n2_eff,l2,order) *self.scalefactor

    @jit
    def angular_overlap(self,l1,ml1,l2,ml2,para):
        """
        compute the angular overlap between two states.
        """
        return angular_overlap_analytical(l1,l2,ml1,ml2,para)
    
    @jit
    def matrix_element(self,n1,l1,ml1,n2,l2,ml2,para,order=1.0):
        """
        computes the matrix element between two states units of ea0.
        """
        angular_integral = self.angular_overlap(l1,ml1,l2,ml2,para)
        if angular_integral == 0.0:
            return 0.0
        else:
            radial_integral = self.radial_overlap(n1,l1,n2,l2,order)
            
        return angular_integral * radial_integral
    
# FUNCTIONS FOR USE BY ATOM CLASS   
    
def get_defect(defects,n,l):
    """
    routine which takes a dictionary of defects and computes the defect at higher n.
        
    Returns
    -------
    defect: float
        The defect of the state
    """
    if l in defects:
        if l+1 in defects[l]:    
            coeffs = defects[l][l+1]
            
            ritz_sum = coeffs[0]    
            for i,coeff in enumerate(coeffs[1:]):
                power = (i+1)*2
                ritz_sum = ritz_sum + coeff / ((n-coeffs[0])**power)
            return ritz_sum 
    else:
        return 0.0
@jit
def get_neff(n,defect):
    """
    returns
    --------
    n^{*}: int
        n*, the effective principle quantum  number
    """
    return n - defect
@jit
def energy_au(neff):     #energy in atomic units
    """
    returns the energy of the state in atomic units.
    """    
    return -(0.5 * (1.0 / ((neff)**2) ))

@jit
def get_radial_wf(neff,l):
    """
    returns the radial wavefunction in numpy array. Computed using the Numerov method.
    
    Returns
    --------
    radial wavefunction: numpy array
        array with the waevfunction value
    positions: numpy array
        corresponding positions
    """
    return wf(neff,l,neff)