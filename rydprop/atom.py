"""
Basic class to compute atom properties. Using GHz as the energy.
"""
from .numerov import radial_overlap,wf
from .angular import angular_overlap_analytical
from .state import basis_options

import scipy.constants as consts
from numba import jit
import numpy as np
m_e = consts.physical_constants['electron mass'][0]
R_0 = consts.physical_constants['Rydberg constant'][0]
En_h =  consts.physical_constants['Hartree energy'][0]
a_0 =  consts.physical_constants['Bohr radius'][0]
m_atomic = consts.physical_constants['atomic mass constant'][0]
c = consts.physical_constants['speed of light in vacuum'][0]
h = consts.physical_constants['Planck constant'][0]
e = consts.physical_constants['elementary charge'][0]
epsilon_0 = consts.epsilon_0
hbar = consts.hbar

class RydbergAtom:
    def __init__(self,mass = 1.00794,defects={},basis = 'nlm',additional_states = None):
        """
        Base class for atom, needs mass and defects to be able to compute all the properties of Rydberg states.
        Routines not accurate at low states so if try and pass a state with a low n will first try and find
        it in the additional_state dictionary. Acts on Basis objects.

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


    def n_eff(self,state1):
        """
        returns the effective n for a given basis
        """
        defect = get_defect(self.defects,state1.n,state1.l)
        return state1.n - defect

    def energy(self,state1):
        """
        gets the energy of a state in units of H'_e, convert to the non-scaled version H_e.
        """
        neff = self.n_eff(state1)
        return energy_au(neff) *self.scalefactor

    def radial_overlap(self,state1,state2,order=1.0):
        """
        compute the radial overlap of two states. returns in units of a'0, convert to non-scaled a0
        """
        n1_eff = self.n_eff(state1)
        n2_eff = self.n_eff(state2)

        return radial_overlap(n1_eff,state1.l,n2_eff,state2.l,order) *self.scalefactor

    def angular_overlap(self,state1,state2,para):
        """
        compute the angular overlap between two states.
        """
        return angular_overlap_analytical(state1.l,state2.l,state1.ml,state2.ml,para)

    def matrix_element(self,state1,state2,para,order=1.0):
        """
        computes the matrix element between two states units of ea0.
        """
        angular_integral = self.angular_overlap(state1,state2,para)
        if angular_integral == 0.0:
            return 0.0
        else:
            radial_integral = self.radial_overlap(state1,state2,order)

        return angular_integral * radial_integral

    def inglis_teller(self,n):
        """
        F_it = F0  /3 *n^{5}

        F0 = 2*h*c*R_m / (e*a_0)
        units of volt/m
        """
        F0 = (2*h*c*self.rydberg_const)/(e*self.bohr)
        return F0 / (3*(n**5))

    def transition(self,state1,state2):
        """
        gets the transition energy between two states. in units of Hartrees
        """
        return self.energy(state1) - self.energy(state2)

    def Rabi_beam(self,state1,state2,power,waist=1e-2,para = True):
        """
        get the rabi frequency between two states, from the power of a
        laser with a power of W, and a waist in m.

        Parameters
        ----------

        state1: state class
            state driving from
        state2: state class
            state driving to
        field_strength: float
            field strength in V/m
        para: boolean
            default True, the polarisation of the field
        """


        intensity = (2*power)/ (np.pi*(waist**2))
        field_strength = np.sqrt( (2*intensity)/(epsilon_0*c)) #in V/m
        transition_dipole = self.matrix_element(state1,state2,para,order = 1.0) * a_0 * e

        return (field_strength *transition_dipole) / hbar

    def Rabi_intensity(self,state1,state2,intensity,para = True):
        """
        get the rabi frequency between two states, from the power of a
        laser with a intensity in W/m^{2}

        Parameters
        ----------

        state1: state class
            state driving from
        state2: state class
            state driving to
        field_strength: float
            field strength in V/m
        para: boolean
            default True, the polarisation of the field
        """

        field_strength = np.sqrt( (2*intensity)/(epsilon_0*c)) #in V/m
        transition_dipole = self.matrix_element(state1,state2,para,order = 1.0) * a_0 * e

        return abs((field_strength *transition_dipole) / hbar)

    def get_radial_wf(self,state1):
        neff = self.get_neff(state1)
        return wf(neff,state1.l,neff)

    def Lifetime(self,temp):
        """
        get the lifetime of a state in seconds at a given temperature.
        """


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
