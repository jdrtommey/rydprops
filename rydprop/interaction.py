from .space import Space
from scipy.sparse import coo_matrix
"""
functions which act on space objects to return a matrix in requested format.
""" 
def interaction(space,int_type=None,parallel = True):
    """
    Parameters
    ==========
    int_type: string
        Name of the interaction
        'elec' - static electric interaction
        'mag' - static magnetic interaction
        'osc' - oscillating electric interaction, same base matrix as elec. 
    """
    if type(space) != Space:
        raise TypeError("Interactions act on " +str(type(Space)) + "objects." )
    
    if int_type in interactions:
        self.function = interactions[int_type]
    else:
        raise KeyError("couldnt find the interaction asked for. Options are " + str(interactions.keys))
        
    return self.function(space,parallel)
        

def electric_interaction(space,parallel):
    """
    electric field interaction, in units of ea_0.
    """
    if parallel == True:
        selection_rules={'dl':1,'dml':0}
    else:
        selection_rules={'dl':1,'dml':1}
        
    row_list=[]
    col_list=[]
    value_list=[]
    
    for row in range(len(space)):  #only loops over upper triangle of space as symmetric
        for col in range(row):
            state1 = space[row]
            state2 = space[col]
            dn = abs(state1.n - state2.n)
            dl = abs(state1.l - state2.l)
            dml = abs(state1.ml-state2.ml)
            if dl == selection_rules['dl'] and dml ==selection_rules['dml']:
                matrix_element = space.atom.matrix_element(state1.n,state1.l,state1.ml,state2.n,state2.l,state2.ml,parallel)
                
                row_list.append(row)
                col_list.append(col)
                value_list.append(matrix_element)
                
                col_list.append(row)
                row_list.append(col)
                value_list.append(matrix_element)
                
                
    return coo_matrix((value_list,(row_list,col_list)),shape=(len(space), len(space)))
            
    
def magnetic_interaction(space,parallel):
    """
    magnetic interaction, currently only works if para if True.
    returns in atomic units (e*hbar/m_e)
    """
    
    
    if parallel == False:
        raise ValueError("Only parallel magnetic fields supported at present")
        
    row_list =[]
    col_list=[]
    value_list=[]
    
    for i,state in enumerate(space):
        row_list.append(i)
        col_list.append(i)
        value_list.append(state.ml * 0.5)
        
    return coo_matrix((value_list,(row_list,col_list)),shape=(len(space), len(space)))


interactions = {'elec':electric_interaction,'mag':magnetic_interaction,'osc':electric_interaction}

