#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions to compute angular integrals
"""
import sqlite3
from numba import jit
@jit
def angular_overlap_analytical(L_1, L_2, M_1, M_2,para):
    """ 
    Angular overlap `<l1, m| f(theta,phi) |l2, m>`. can compute parallel and perpendicular interactions. Taken from A. Morgan. 
    
    """
    dL = L_2 - L_1
    dM = M_2 - M_1
    L, M = int(L_1), int(M_1)
    overlap = 0.0
    if para == True:
        if (dM == 0):
            if dL == +1:
                overlap =  (+(((L+1)**2-M**2)/((2*L+3)*(2*L+1)))**0.5)
            elif dL == -1:
                overlap =  (+((L**2-M**2)/((2*L+1)*(2*L-1)))**0.5)
        elif (dM == +1):
            if dL == +1:
                overlap =  (-((L+M+2)*(L+M+1)/(2*(2*L+3)*(2*L+1)))**0.5)
            elif dL == -1:
                overlap =  (+((L-M)*(L-M-1)/(2*(2*L+1)*(2*L-1)))**0.5)
        elif (dM == -1):
            if dL == +1:
                overlap =  (+((L-M+2)*(L-M+1)/(2*(2*L+3)*(2*L+1)))**0.5)
            elif dL == -1:
                overlap =  (-((L+M)*(L+M-1)/(2*(2*L+1)*(2*L-1)))**0.5)

    if para == False:
        if dM == +1:
            if dL == +1:
                overlap = (+(0.5*(-1)**(M-2*L))  * (((L+M+1)*(L+M+2))/((2*L+1)*(2*L+3)))**0.5)
            elif dL == -1:
                overlap =  (-(0.5*(-1)**(-M+2*L)) * (((L-M-1)*(L-M))  /((2*L-1)*(2*L+1)))**0.5)
        elif dM == -1:
            if dL == +1:
                overlap = (+(0.5*(-1)**(M-2*L))  * (((L-M+1)*(L-M+2))/((2*L+1)*(2*L+3)))**0.5)
            elif dL == -1:
                overlap = (-(0.5*(-1)**(-M+2*L)) * (((L+M-1)*(L+M))  /((2*L-1)*(2*L+1)))**0.5)
    return overlap
