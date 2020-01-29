# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:50:15 2019

@author: Lucio
"""

import numpy as np
from numpy import linalg
from scipy import constants
import scipy.linalg as scilinalg
from math import sin ,cos, sqrt as scalar_sqrt

import quenchingsimulator_toolbox as qtool

#### Hamiltonian parameters #####
#Constants
g_gs= 2
bohr_mag, _, _ = constants.physical_constants["Bohr magneton"]
hplanck = constants.Planck
hplanck_par = constants.hbar
gamma2pi = g_gs * bohr_mag  / hplanck # Hz/T

# Change units to GHz
gamma2pi = gamma2pi

Dgs = 2.87e9
Des = 1.42e9
Sz = np.array( [[1,0,0],
                [0,0,0],
                [0,0,-1]], dtype = complex)
Sx = np.array([[0,1,0],
               [1,0,1],
               [0,1,0]], dtype = complex) / scalar_sqrt(2)
Sy = np.array( [[0.,-1j,0.],
                [1j,0,-1j],
                [0.,1j,0.]], dtype = complex ) / scalar_sqrt(2)

#S_array = np.stack([Sx, Sy, Sz], axis = 2)
S_array = np.stack([Sx, Sy, Sz], axis = 0)

Sz_sq = np.dot(Sz,Sz)
Sx_sq_Sy_sq = np.dot(Sx,Sx) - np.dot(Sy,Sy)

H0_gs =  Dgs * Sz_sq# + Egs * Sx_sq_Sy_sq
H0_es =  Des * Sz_sq

#Hmagnetic = lambda Bvec: gamma2pi * (Bvec[0]*Sx + Bvec[1]*Sy + Bvec[2]*Sz)

#Htot_gs = lambda Bvec: H0_gs + Hmagnetic(Bvec)
#Htot_es = lambda Bvec: H0_es + Hmagnetic(Bvec)

def Hmagnetic_fast(Bvec):
    """
    Bvec is (Bfield_num,3)-shaped array
    """
    return gamma2pi * np.einsum("njk,mn->mjk",S_array,Bvec)


def Htot_gs_fast(Bvec):
    return H0_gs + Hmagnetic_fast(Bvec)

def Htot_es_fast(Bvec):
    return H0_es[np.newaxis,:,:] + Hmagnetic_fast(Bvec)

def zerofield_kraus(k30, k41, k52, #Radiative decay rates
                        k36, k46,  k56, #Nonrad. decays to the intersystem crossing
                        k60, k61, k62, #Nonrad. decays from the intersystem crossing
                        laser_pump = 0
                        ):
    the_lindblads = np.zeros((12,7,7), dtype = complex)
    the_lindblads[0,1,4] = scalar_sqrt(k30)
    the_lindblads[1,2,5] = scalar_sqrt(k41)
    the_lindblads[2,0,3] = scalar_sqrt(k52)
    the_lindblads[3,4,1] = scalar_sqrt(laser_pump)
    the_lindblads[4,5,2] = scalar_sqrt(laser_pump)
    the_lindblads[5,3,0] = scalar_sqrt(laser_pump)
    the_lindblads[6,6,4] = scalar_sqrt(k36)
    the_lindblads[7,6,5] = scalar_sqrt(k46)
    the_lindblads[8,6,3] = scalar_sqrt(k56)
    the_lindblads[9,1,6] = scalar_sqrt(k60)
    the_lindblads[10,2,6] = scalar_sqrt(k61)
    the_lindblads[11,0,6] = scalar_sqrt(k62)
    
    return the_lindblads

def lindblad_quench_simulator(Bfields = None, rate_dictionary = None,
                              nv_theta = 54.7, nv_phi = 0, rate_coeff = 1e-3,
                              Bias_field = None,correct_for_crossing = False):
    
    default_rate_dictionary = {'kr' : 32.2e6,          # The radiative decay rate
                               'k36' : 12.6e6,          # Non-radiative to shelving, ms=0
                               'k45_6' : 80.7e6,       # Non-radiative to shelving, ms=+-1
                               'k60' : 3.1e6,          # Non-radiative from shelving to ms=0
                               'k6_12' : 2.5e6,        # Non-radiative from shelving to ms=+-1
                               'mw_rate' : 0,        # Microwave driving
                               'laser_pump' : 0.1}   # Laser driving, percentage of kr 
    if rate_dictionary is not None:
        for key, value in rate_dictionary.items():
            if key in default_rate_dictionary:
                default_rate_dictionary[key] = value
    
    input_shape = Bfields.shape
    if input_shape == (3,):
        Bfields = np.array([Bfields])
        bfield_num = 1
        input_shape = (1,)
    elif len(input_shape) == 2:
        bfield_num = input_shape[0]
    elif len(input_shape) > 2:
        bfield_num = 1
        for ii in range(0,len(input_shape)-1):
            bfield_num = bfield_num * input_shape[ii]
        Bfields = np.reshape(Bfields, (bfield_num,3))
    
    
    kr = default_rate_dictionary['kr']
    k36 = default_rate_dictionary['k36']
    k45_6 = default_rate_dictionary['k45_6']
    k60 = default_rate_dictionary['k60']
    k6_12 =default_rate_dictionary['k6_12']
    laser_pump = default_rate_dictionary['laser_pump'] * kr
    
    zerofield_krausop = zerofield_kraus(kr ,kr , kr,
                                        k36, k45_6, k45_6,
                                        k60, k6_12, k6_12,
                                        laser_pump = laser_pump)
    if (Bias_field is None):
        Bias_field = 0
    elif (linalg.norm(Bias_field) == 0):
        Bias_field = 0
    Bfields = Bfields + Bias_field
    Bfields = rotate2nvframev_fast(vectors2transform = Bfields[:,:,np.newaxis], nv_theta = nv_theta, nv_phi = nv_phi)
    norm_is_zero = (linalg.norm(Bfields,axis = 1) == 0)
    
    
    kraus_decaypaths, kraus_rows, kraus_cols = zerofield_krausop.shape
    """
    Generate the Hamiltonians for gs and es, then combine them into
    the six-level system Hamiltonian
    """
    full_Htot_gs = Htot_gs_fast(Bfields)
    full_Htot_es = Htot_es_fast(Bfields)
    
    master_hamiltonian = np.zeros((bfield_num,7,7), dtype = complex)
    master_hamiltonian[:,0:3,0:3] = full_Htot_gs
    master_hamiltonian[:,3:6,3:6] = full_Htot_es
    master_hamiltonian[:,-1,-1] = 1
    
    """
    Generate the unitary transformation matrix
    """
    coefficient_matrix = find_eigens_and_compose_fast(Htot_gs = full_Htot_gs, Htot_es=full_Htot_es,
                                                      correct_for_crossing = correct_for_crossing)
    coefficient_matrix[norm_is_zero,:,:] = np.eye(kraus_rows)
    
    
    #Generate adjoints for code readability
    #coefficient_matrix_adj = herm_conj(coefficient_matrix,[0,2,1])
    zerofield_krausop_adj = zerofield_krausop.conj().transpose([0,2,1])
    
    """
    Generate the matrices for the vectorised lindbladian
    """
    
    identity = np.zeros((kraus_rows,kraus_cols), dtype = complex)
    identity[np.diag_indices(kraus_rows)] = 1 + 0j
    
    sum_Lkdag_Lk = np.einsum("ijk,ikm->jm",zerofield_krausop_adj,zerofield_krausop)
    
    sum_Lkdag_Lk_rho = np.einsum("ij,km->ikjm", identity, sum_Lkdag_Lk)
    sum_rho_Lkdag_Lk = np.einsum("ij,km->jkim", sum_Lkdag_Lk, identity)
    
    sum_Lk_rho_Lkdag = np.einsum("ijk,imn->kmjn",zerofield_krausop_adj,
                                 zerofield_krausop)
    
    decoherence_lindblad = (sum_Lk_rho_Lkdag - 0.5*(sum_Lkdag_Lk_rho + sum_rho_Lkdag_Lk)
                            ).reshape(kraus_rows**2,kraus_cols**2)
    
    ham_commutator = (np.einsum("ij,mlk->miljk",
                               identity,master_hamiltonian
                               ) 
    
                     -np.einsum("ijk,mn->ijnkm",
                                master_hamiltonian,
                                identity
                                )
                     ).reshape(bfield_num, kraus_rows**2, kraus_cols**2)              
    
    lindbladian = (-1j*2*np.pi)*ham_commutator + decoherence_lindblad
    
    lindbladian[:,0,:] = create_solution_row(kraus_rows)
    
    solution_column = np.zeros(kraus_rows**2)
    solution_column[0] = 1
    solution_column = np.repeat(solution_column[None,:,None], repeats = bfield_num, axis = 0)
    

    final_solution =  (np.linalg.solve(lindbladian,solution_column).
                       reshape(bfield_num, kraus_rows, kraus_rows).
                       transpose([0,2,1])
                       )
   
    
    #final_solution = np.einsum("ijk,ikn,inq->ijq",coefficient_matrix_adj,final_solution,coefficient_matrix)
    
    
    pops = np.einsum("ijj->ij",final_solution)
    #pops = final_solution[:,(create_solution_row(kraus_rows)!=0)]
    
    #Order the populations along in standard order (e.g. |0>,|1>,|2>,...)
    pops = pops[:,[1,2,0,4,5,3,6]]
    
    return pops, final_solution

def herm_conj(matrix, *args, **kwargs):
    """
    Short form for the Hermitian conjugate, can take also N-dimensional matrices
    """
    
    return matrix.conj().transpose(*args, **kwargs)

def create_solution_row(density_op_dim):
    mat = np.eye(density_op_dim)
    return mat.reshape(density_op_dim**2).astype(complex)

# Matrix for the NV decay rates
#
#
#
#       ______ |5>
#       ______ |4>
#
#   ______ |3>
#
#
#                   _____ |6>
#
#
#       ______ |2>
#       ______ |1>
#
#   ______ |0>

def find_eigens_and_compose_fast(Htot_gs = None, Htot_es = None,
                                 correct_for_crossing = False):
    b_field_num = Htot_gs.shape[0]
   
    _, eistates_gs = linalg.eigh( Htot_gs )
    _, eistates_es = linalg.eigh( Htot_es )
    
    if correct_for_crossing:
        
        GS_ground_state_coeff_pos = np.argmax(np.square(np.abs( eistates_gs[:,1,:] )),axis = 1)
        sub = eistates_gs[GS_ground_state_coeff_pos>0,:,:]
        sub[:,:,[0,1]] = sub[:,:,[1,0]]
        eistates_gs[GS_ground_state_coeff_pos>0,:,:] = sub
        
        ES_ground_state_coeff_pos = np.argmax(np.square(np.abs( eistates_es[:,1,:] )), axis = 1)
        sub = eistates_es[ES_ground_state_coeff_pos>0,:,:]
        sub[:,:,[0,1]] = sub[:,:,[1,0]]
        eistates_es[ES_ground_state_coeff_pos>0,:,:] = sub
        
   #The returned eigenstates are columns, ordered from lowest to highest energy
   #With the next, they are arranged according to the Pauli operators basis
    eistates_gs = eistates_gs[:,:, [2,0,1]]
    eistates_es = eistates_es[:,:,[2,0,1]]
    
    full_matrix = np.zeros( (b_field_num,) + (7,7), dtype = complex )
    full_matrix[:,0:3,0:3] = eistates_gs
    full_matrix[:,3:6,3:6] = eistates_es
    full_matrix[:,-1,-1] = 1
    
    return full_matrix

def rotate2nvframev_fast(vectors2transform = np.zeros((2,3,1)), nv_theta = 0, nv_phi = 0):
    reps = vectors2transform.shape[0]
    
    rot_matrix = np.array([ [cos(nv_phi)*cos(nv_theta), sin(nv_phi)*cos(nv_theta), -sin(nv_theta)],
                            [-sin(nv_phi), cos(nv_phi), 0],
                             [sin(nv_theta)*cos(nv_phi), sin(nv_phi)*sin(nv_theta), cos(nv_theta)] ] )
    
    rot_matrix = np.repeat(rot_matrix[np.newaxis,:,:],reps,axis = 0)
    
    result = np.matmul( rot_matrix,  vectors2transform )
    
    return result[:,:,0]

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    norms = np.linspace(0,Dgs/gamma2pi+10e-3,5000) #np.array([Dgs/gamma2pi])
    normsplot = np.abs(norms)
    ang =  10 * np.pi/180 #np.linspace(0,10*np.pi/180,1000)#np.linspace(0,np.pi/2,1000)
    ang2 = 0 * np.pi/180
    phinv = 0* np.pi/180
    
    rate_coeff = np.linspace(1e-3,1e9,100)
    
    plotaxis = norms#ang * 180/np.pi
    
    Bfields = np.stack((norms*cos(ang2)*np.sin(ang),norms*sin(ang2)*np.sin(ang),np.cos(ang)*norms),axis = 1)
    
    lp = 0.01
    coeff = 1e6
    def rate_dictionary_func(coeff):
        r8= {'kr' : 32.2*coeff,          # The radiative decay rate
             'k36' : 12.6*coeff,          # Non-radiative to shelving, ms=0
             'k45_6' : 80.7*coeff,       # Non-radiative to shelving, ms=+-1
             'k60' : 3.1*coeff,          # Non-radiative from shelving to ms=0
             'k6_12' : 2.5*coeff,        # Non-radiative from shelving to ms=+-1
             'mw_rate' : 0,        # Microwave driving
             'laser_pump' : lp}   # Laser driving, percentage of kr 
        return r8
    
    
    
    poppy , hey = lindblad_quench_simulator(Bfields, nv_theta = 0, nv_phi=phinv,
                                            rate_dictionary = rate_dictionary_func(coeff))
    
    #fig,ax = plt.subplots(figsize = (12,7))
    for line in poppy[:,:].T:
        plt.plot(plotaxis,line)
    plt.show()

    