from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix,isspmatrix
from numpy.linalg import eigh,eigvalsh
import numpy as np

class Diagonaliser: 
    def __init__(self,return_vecs,num_eigs,h0,h1):
        """
        Determines which routine to use for the diagonalisation and return of vectors and values.
        has a toggle if vectors are returned. If all eigenvectors are requested uses the numpy dense
        routine with dense matrices, else will use the scipy ARPACK routine and csr sparse matrices 
        
        Parameters
        ----------
        return_vecs: boolean
            if the program should return vectors
        num_eigs: int
            number of eigenstates to return
        h0: matrix
            the parameter-free matrix
        h1: matrix
            the interaction matrix
        """
        if num_eigs == None:
            self.num_eigs = h0.shape[0]
        else:
            self.num_eigs = num_eigs
        self.return_vecs = return_vecs
        
        # based on the number of eigenvalues requested determine if possible to use sparse
        if self.num_eigs <= (h0.shape[0] - 2):
            self.sparse = True
        else:
            self.sparse = False
            
        self.h0,self.h1 = self.matrix_converter(h0,h1,self.sparse)
            
    def matrix_converter(self,h0,h1,sparse_engine):
        """
        determines the type of matrix which has been given to the diagonaliser, 
        and sees if it can be converted into sparse format. Assumes either a
        dense numpy array was provided or a scipy sparse array of some format
        Works for sparse and numpy arrays.
        """
        matrix_type = type(h0)
        
        if sparse_engine == True and matrix_type!=csr_matrix:
            try:
                h0_converted = csr_matrix(h0)
                h1_converted = csr_matrix(h1)
            except:
                raise ValueError("Failed to convert matrices into csr format.")
            
        elif sparse_engine == False and matrix_type != np.ndarray:
            try:
                h0_converted = h0.toarray()
                h1_converted = h1.toarray()
            except:
                raise ValueError("Failed to convert matrices into dense format.")
        else:
            h0_converted = h0
            h1_converted = h1
            
        return h0_converted,h1_converted
        
        
    def __call__(self,param,sigma):
        return self.get_values(param,sigma)
        
    def get_values(self,param,sigma):
        """
        given a parameter will return the eigenvalues/vectors
        """
            
        total_matrix = self.h0 + self.h1 * param
        vals = None
        vecs = None
        if self.sparse == True:
                vals,vecs = eigs(total_matrix,k=self.num_eigs,sigma=sigma,return_eigenvectors=self.return_vecs)
        elif self.sparse == False:
            if self.return_vecs == True:
                vals,vecs = eigh(total_matrix)
            elif self.return_vecs == False:
                vals = eigvalsh(total_matrix)
            
        return vals,vecs