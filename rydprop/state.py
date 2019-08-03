from numba import jit
default_basis = 'nlm'

class State_nlm:
    @jit
    def __init__(self,n,l,ml):
        """
        class which holds quantum numbers in the |n,l,ml> basis.
        """
        self._n = n
        
        if l >= 0 and l < n:
            self._l = l
        else:
            raise ValueError("l must be between 0 and n. l provided l = " +str(l)+", provided n = "+str(n))
            
        if abs(ml) <= l:    
            self._ml = ml
        else:
            raise ValueError("ml must be between -l and l. provided ml = "+str(ml)+", provided l = " + str(l))
            
    @property
    def n(self):
        return self._n
    
    @property
    def l(self):
        return self._l
    
    @property
    def ml(self):
        return self._ml
    
    def __members(self):
        return (self._n,self._l,self._ml)
        
    def __eq__(self,other_state):
        """
        checks if another state_nml is equal to this state or not.
        """
        
        if type(self) == type(other_state):
            return self.__members() == other_state.__members()
        else:
            return False

        
class State_nlj:
    def __init__(self,n,l,j,mj):
        """
        class which holds quantum numbers in the |n,l,ml> basis.
        """
        self._n = n
        
        if l >= 0 and l < n:
            self._l = l
        else:
            raise ValueError("l must be between 0 and n. l provided l = " +str(l)+", provided n = "+str(n))
            
        if j in [l-0.5,l+0.5]:
            self._j = j
        else:
            raise ValueError("j must be either l-0.5 or l+0.5. provided j = "+str(j)+", provided l = " + str(l))
            
        if abs(ml) <=j:
            self._mj = mj
        else:
            raise ValueError("mj must be between -j and j. provided j = "+str(j)+", provided mj = " + str(mj))
            
    @property
    def n(self):
        return self._n
    
    @property
    def l(self):
        return self._l
    
    @property
    def j(self):
        return self._j
    
    @property
    def mj(self):
        return self._mj
    
    def __members(self):
        return (self._n,self._l,self._j,self.mj)
        
    def __eq__(self,other_state):
        """
        checks if another state_nlj is equal to this state or not.
        """
        
        if type(self) == type(other_state):
            return self.__members() == other_state.__members()
        else:
            return False
        
        
basis_options = dict({'nlm':State_nlm,'nlj':State_nlj})

