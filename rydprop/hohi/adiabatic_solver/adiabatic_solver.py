from scipy.sparse import isspmatrix
from .plugins.state_init import State_initialiser   #gnerates the initial adiabatic states to track
from .plugins.sigma_guesser import Sigma_gen        #determines how the eigenstate guess should change
from .plugins.diag_engine import Diagonaliser       #determines which diagonalisation routine to use
from .plugins.state_matcher import Matcher          #algorithims to determine if adibatic state has been found
from .solver import Solver
import numpy as np

class Adiabatic_solver:
    def __init__(self,h0,h1,parameters,sigma = 0.0,num_eigs = None,return_vecs=True,algorithim='basic',return_failed = False,\
                 energy_list = None,index_list = None,sig_method = 'static',target=None,ratio=None):
        """
        The interface which user interacts with. It generates all the classes which are
        needed to be given to the solver to impliment the desired tracking. 
        performs all the checks and conversions necessary to attempt to convert the problem into
        the optimal form for the solver, then calls the solver and provides methods for 
        the return of the adiabatic states.
        
        Parameters
        ----------
        h0: sparse or dense matrix
            diagonal matrix which dosent change with parameter
        h1: sparse or dense matrix
            interaction matrix which changes with parameter
        parameters: numpy array
            the values of the interaction parameter for which eigenvalues and vectors are computed
        sigma: float
            the initial eigenvalue region to search in. see ARPACK diagonalisation routine.
        num_eigs: float
            the number of eigenvalues to compute. If not provided will compute all
        return_vecs: boolean
            Tells solver if eigenvectors should be computed. default = True
        algorithim: string
            Which algorithim to use. Currently has:
            'a' - match adiabatic states by using both value and coefficient.
        return_failed: boolean:
            whether the solver should return a list containing the partially tracked adibatic states
        energy_list: list
            contains an upper and lower bound which adiabatic states to track
        index_list: list
            contains the parameter free indices of adiabatic states to track
        sig_method: string
            how the eigenstate value guess should change each interaction
            'static' - remains constant throughout
            'target' - tracks a single adiabatic state
        target: list
            optional, list of the adibatic states whose energies should be tracked
        ratio: float 
            ratio of the target states to guess eigenvalue. same length as target,
            if not provided will set to equal
        """
        dimension = h0.shape[0]       
        self.Diag_engine = init_diagonalizer(return_vecs,num_eigs,h0,h1)
        self.Sigma_logic = init_sigma_checker(sig_method,target,ratio,dimension)
        self.Adibatic_generator = init_adibatic_state(energy_list,index_list,dimension)
        self.Matcher_logic = init_matcher(algorithim,dimension,num_eigs)
                        
        self.Solver = init_solver(self.Adibatic_generator,self.Matcher_logic,self.Sigma_logic,self.Diag_engine\
                                       ,parameters,sigma)

        self.return_failed = return_failed
    def run(self):
        """
        runs the solver and returns two lists, of successful /and failed states.
        """
        success,fail = self.Solver.mainloop()
        if self.return_failed == True:
            return success,fail
        else:
            return success
        
        
def init_solver(Adibatic_generator,Matcher_logic,Sigma_logic,Diag_engine,parameters,sigma):
    """
    generates the solver
    """
    if type(Adibatic_generator) != State_initialiser:
        raise TypeError("Adibatic_generator must be type: "+  str(type(State_initialiser)))
    if type(Matcher_logic) != Matcher:
         raise TypeError("Matcher_logic must be type: "+  str(type(Matcher)))
    if type(Sigma_logic) != Sigma_gen:
        raise TypeError("Sigma_logic must be type: " + str(type(Sigma_gen)))
    if type(Diag_engine) != Diagonaliser:
        raise TypeError("Diag_engine must be type: " + str(type(Diagonaliser)))
                       
    if type(parameters) != np.ndarray:
        try:
            parameters = np.asarray(parameters)
        except:
            raise TypeError("Could not convert parameters to numpy array:"  + str(type(parameters)))
        
    return Solver(Adibatic_generator,Matcher_logic,Sigma_logic,Diag_engine,parameters,sigma)
    
def init_sigma_checker(sig_method,target,ratio,dimension):
    """
    generates the class which will make a decision on where to 
    move the eigenstate range to probe
    """
    #check method is a string
    if type(sig_method) != str:
        raise ValueError("sigma_checker method must be string, not:" + str(type(sig_method)))
            
    #if target provided but ratio not make ratio equal amount of each
    if type(target)  != type(None) and ratio == None:
        try:
            ratio = (1/len(target)) * np.ones(len(target))
        except:
            raise RuntimeError("failed to generate ratios")
            
    # check ratio and target are same length 
    if type(target) != type(None) and type(ratio) != type(None):
        if len(target) != len(ratio):
            raise ValueError("target and ratio lists must be same length")
    

    
    #check sum of ratios basically one.
    if type(ratio) != type(None):
        if abs(1 - np.sum(ratio)) > 0.00001:
            raise ValueError("Ratios must sum to 1.0, " + str(np.sum(ratio)))
        
    target = np.asarray(target)
    ratio = np.asarray(ratio)
    return Sigma_gen(sig_method,target,ratio)
        
def init_adibatic_state(energy_list,index_list,dimension):
    """
    generates the class which is used to determine which
    adiabatic states are going to be searched for.
    """
    #check only one of energy_list and index_list are not None
    if type(energy_list) != type(None) and type(index_list) != type(None):
        raise ValueError("Provide either either an energy_list or and index_list not both")
            
    #check no indexs are given beyond dimension of matrix
    if type(index_list) != type(None):
        for index in index_list:
            if index >= dimension:
                raise IndexError("cannot create adiabatic state for index outside dimension of hamiltonian: " +str(dimension))
                
    if type(energy_list) != type(None):
        method = 'energy'
        input_list = energy_list
    elif type(index_list) != type(None):
        method = 'index'
        input_list = index_list
    else:
        method = 'all'
        input_list = None
    return State_initialiser(method,input_list)
    
def init_matcher(algorithim,dimension,num_eigs):
    """
    generates the class which will apply the logic for if an adibatic state has been
    found by the diagonaliser. Given it an algorithim to perform.
    """
    #check algorithim given as a string.
    if type(algorithim) != str:
        raise TypeError("method must be a string")
        
    """
    check basic algorithim only used when performing dense operation.
    """
    if algorithim == 'basic' and num_eigs != None:
        if num_eigs != dimension:
            raise ValueError("currently, 'basic' algorithim only supports dense mode where num_eigs = dimenison of space " +str(dimension))
    return Matcher(algorithim)
        
def init_diagonalizer(return_vecs,num_eigs,h0,h1):
    """
    initialises which method is going to be used to perform the diagonalisation, i.e if
    a sparse or dense method is required.
    """        
    #check matrices either scipy sparse or numpy dense.        
    if isspmatrix(h0) == True and isspmatrix(h1) == True:
        matrix_fine = True
    elif type(h0) == np.ndarray and type(h1) == np.ndarray:
        matrix_fine = True
    else:
        raise TypeError("Matrices must be provided in either scipy sparse or numpy format.")
            
    #check matrices are square
    if h0.shape[0] != h0.shape[1] or h1.shape[0] != h1.shape[1]:
        raise ValueError("Matrices must be square")
        
    #check h0 and h1 matrices have same dimension       
    if h0.shape[0] != h1.shape[0]:
        raise ValueError("H0 and H1 must be same dimension.")

    #if the number of eigenstates not specificed will compute all of them.
    if num_eigs == None:
        num_eigs = h0.shape[0]
            
    #check num_eigs is bounded by the dimension of the matrix
    if num_eigs > h0.shape[0]:
        raise ValueError("number of eigenstates to find must be <= dimension of system " + str(h0.shape))
        
    #returns a function which given a h0,h1 and parameter gives the eigens.
    return Diagonaliser(return_vecs,num_eigs,h0,h1)
        