"""
module to compute the spectral intensities of excitation

tools for the convolution of outcomes

tools for the comparision with experimental peaks

A vector of couplings <n,l,ml|er|groundstate> is passed into the function which sums up the eigenvector
weightings and the transition moment. Take care passing in a Floquet space. 
"""
from numba import jit
import numpy as np

def get_coupling_ground(space,groundstate,para):
    """
    function which acts on the space object. Returns an array of the coupling 
    strength between groundstate and all states in space.
    
    Parameters
    ----------
    space: rydprop Space object
        the space of states to act on
    groundstate: rydprop state object
        the ground state to compute coupling to
    para: Boolean
        the polarization of excitation laser
        
    Returns
    --------
    couplings: np array
        array holding the coupling strengths
    """
    couplings= []
    for state in space:
        couplings.append(space.atom.matrix_element(groundstate,state,para))
        
    return np.asarray(couplings)

def get_floq_coupler(space,qs,groundstate,para):
    """
    If the space has been floqified, the vector space contains 2*qs+1 copies
    of the Space. Exciation laser only excites to the q=0 states so
    pad the coupling vector with zeros for all the q!=0 states.
    
    Parameters
    ----------
    space: rydprop Space object
        the space of states to act on
    qs: int
        the number of sidebands on either side
    groundstate: rydprop state object
        the ground state to compute coupling to
    para: Boolean
        the polarization of excitation laser
        
    Returns
    --------
    couplings: np array
        array holding the coupling strengths
    """
    
    bare_coupler = get_coupling_ground(space,groundstate,para)
    coupler = np.zeros( (2*qs+1)*len(space) )
    initial = len(space)*qs
    final = len(space)*qs + len(space)
    coupler[initial:final] = bare_coupler
    return coupler

def get_spec(coupls,eigenvec):
    """
    given a coupling vector and an eigenvector with the weightings of each bare state
    will compute the spectral intensity (\sum(C_i<state_i|ground> ))^{2}
    
    Parameters
    ----------
    coupls: rydprop Space object
        the space of states to act on
    eigenvec: int
        the number of sidebands on either side
        
    Returns
    --------
    spectral: float
        the spectral weighting to this eigenstate
    """
    
    spectral = np.sum(coupls * np.asarray(eigenvec))**2
    return spectral

def get_spectrals(vals,vecs,coupler):
    """
    function takes the vals and vecs which are output from a numpy eigensolver and 
    computes a list of the spectral intensitites
    
    Parameters
    ----------
    vals: numpy array
        output from numpy eigensolvers
    vecs: numpy array
        output from numpy eigensolvers
    coupler: numpy array
        the coupling vector to the groundstate
    
    Returns
    -------
    spectrals: numpy array
        the spectral intensities for all the eigenstates
    """
    spectrals =[]
    for i,v in enumerate(vals):
        spectral = get_spec(coupler,vecs[:,i])
        spectrals.append(spectral)
    spectrals = np.asarray(spectrals)
    return spectrals

def plot_spectral_intensity(x_range,vals,spectrals,std,minimum = 0.0):
    """
    given a domain on which to evaluate and a list of eigenvalues and their
    spectral intensities generate a spectral intensity plot.
    
    Parameters
    ----------
    x_range: numpy array
        domain over which to evaluate the intensities
    vals: numpy array
        eigenvalue output from numpy eigensolvers
    spectrals: numpy array
        the computed spectral intensity for each eigenvector
    std: float
        the standard deviation of each gaussian
    minimum: float (default 0.0)
        the minimum spectral intensity needs to be for a state to include it 
    
    Returns
    -------
    y_range: numpy array
        the resultant spectral intensity
    """
    y_range = np.zeros(len(x_range))
    for i,s in enumerate(spectrals):
        if s > minimum:
            y_range = y_range + generate_guassian(x_range,vals[i],std,spectrals[i])
    return y_range

def generate_guassian(x_range,mu,std,height):
    """
    function which will evalute a gaussian over x_range.
    
    Parameters
    ----------
    x_range: numpy array
        a vector of x positions on which to evalute the gaussian function
    mu: float
        the mean of gaussian
    std: float
        the standard deviation of gaussian
    height:
        the maximum height of the gaussian

    Returns
    -------
    y_range: numpy array
        the spectral intensities for all the eigenstates   
    """
    y_range=[]
    for x in x_range:
        y_range.append(_gaussian(x,mu,std,height))
        
    return np.asarray(y_range)
    
@jit
def _gaussian(x,mu,sig,height):
    """
    helper function which computes a guassian
    """
    return height*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))