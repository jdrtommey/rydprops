import numpy as np
from scipy.interpolate import  interp1d
class Adiabatic:
    def __init__(self,index,initial_value,initial_parameter,initial_vector):
        """
        contains the evolution of an adiabatic state. methods to compare a given eigenvector and value
        to this to see if it is part of the same state. initialise it with the 
        
        Parameters
        ----------
        index: int
            when italised records which eigenstate of the unperturbed hamiltonian this
            adiabatic state belongs to.
            
        initial_value: float
            initial eigenvalue
        initial_parameter: float
            initial value of the interaction parameter
        initial_vector: numpy array:
            the initial vector
        """
        self.index = index
        self.parameter = np.asarray([initial_parameter])
        self.vals = np.asarray([initial_value])
        self.vecs = np.asarray([initial_vector])
        
    def _add_vector(self,vector):
        """
        adds a new vector by stacking it under the list of vectors.
        """
        #if not already a numpy array try and convert
        if type(vector) != np.ndarray:
            try:
                vector = np.asarray(vector)
            except:
                raise AttributeError("could not convert vector to correct format for adibatic state, type provided: " \
                                    + type(vector))                   
        try:
            self.vecs = np.append(self.vecs,[vector],axis=0)
        except:  
            raise AttributeError("Could not add value to adiabatic state " + str(vector))
        
    def _add_value(self,value):
        """
        adds a new value to end of values array
        """
        
        try:
              self.vals = np.append(self.vals,value)
        except:
            raise AttributeError("Could not add value to adiabatic state")
        
    def _add_parameter(self,parameter):  #test
        """
        adds a new parameter value to the array
        """
        try:
            self.parameter = np.append(self.parameter,parameter)
        except:
            raise AttributeError("could not add parameter to adiabatic state")
        
        
    def add(self,val,vec,param):
        """
        given a vec/val/parameter will add them.
        """
        self._add_value(val)
        self._add_vector(vec)
        self._add_parameter(param)
    
    #METHODS FOR RETREIVING INFORMATION ON THE ADIABATIC STATE
        
        
    def get_length(self):
        """
        gets the current number of values this state holds of the adibatic state.
        """
        return len(self.parameter)
    
    def get_current_parameter(self):
        """
        returns the most recent parameter of this state
        """
        try:
            return self.parameter[-1]
        except:
            raise IndexError("Couldnt access the last value of values of length " + str(len(self.parameter)))
        
    def get_current_value(self):    #test
        """
        returns the current energy of this state
        """
        try:
            return self.vals[-1]
        except:
            raise IndexError("Couldnt access the last value of values of length " + str(len(self.vals)))
    
    def get_current_coefficient(self,index =None):   #test
        """
        gets the current coefficient of the target state,defaults to
        the states own adiabatic coefficient
        """
        
        if index == None:
                return self.vecs[-1][self.index]
        else:
            try:
                return self.vecs[-1][index]
            except:
                raise IndexError("Could not return value for index: " + str(index))
        