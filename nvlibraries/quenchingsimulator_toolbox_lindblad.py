# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 10:50:15 2019

@author: Lucio Stefan, lucio.stefan@gmail.com


This is a temporary library. Bugs need to be fixed! The main issue arises when
applying a magnetic field which has a component along y. In that case, populations
can be negative or more than 1. Most likely, the solver has too many degrees of 
freedom, in the sense that no condition over the off-diagonal elements of the
density operator is imposed. The only imposed condition is that the Tr(rho)=1

The problem is fixed by simply calculating the magnetic field components 
parallel and orthogonal nv axis, and then using them in the Zeeman Hamiltonian
as H_zeeman(B)= B_ort * Sx + B_par * Sz. Since the NV is symmetric around the
NV axis, it's not a restrictive approximation.

In this case the solution converges to the solution calculated using the
Qutip python library. Qutip library otherwise shows that there are no changes
when rotating the equatorial angle.

The solution also matches Tetienne solution used in the quenching_simulator
(https://iopscience.iop.org/article/10.1088/1367-2630/14/10/103033), but behaves
correctly at the level anticrossings for small angles between B field and NV.

"""

# Notation for the NV decay rates
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


import numpy as np
from numpy import linalg
from scipy import constants
from math import sin ,cos, sqrt as scalar_sqrt

#### Hamiltonian parameters #####
#Constants
"""
g_gs= 2
bohr_mag, _, _ = constants.physical_constants["Bohr magneton"]
hplanck = constants.Planck
hplanck_par = constants.hbar
gamma2pi = g_gs * bohr_mag  / hplanck # Hz/T
"""


hplanck = constants.Planck
hplanck_par = constants.hbar
g_gs,_,_ = constants.physical_constants['electron g factor']
g_gs = abs(g_gs)
bohr_mag,_,_ = constants.physical_constants["Bohr magneton"]
gamma2pi,_,_ = constants.physical_constants['electron gyromag. ratio over 2 pi']
gamma2pi *= 1e6 #Is given in MHz/T, convert in Hz/T

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

Sy = np.zeros((3,3))
"""
Sy = np.array( [[0.,-1j,0.],
                [1j,0,-1j],
                [0.,1j,0.]], dtype = complex ) / scalar_sqrt(2)
"""

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
    return H0_es + Hmagnetic_fast(Bvec)

def zerofield_kraus(k30, k41, k52, #Radiative decay rates
                        k36, k46,  k56, #Nonrad. decays to the intersystem crossing
                        k60, k61, k62, #Nonrad. decays from the intersystem crossing
                        laser_pump = 0,
                        kmw_01 = 0, kmw_02 = 0
                        ):
    the_lindblads = np.zeros((16,7,7), dtype = complex)
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
    the_lindblads[12,0,1] = scalar_sqrt(kmw_02)
    the_lindblads[13,1,0] = scalar_sqrt(kmw_02)
    the_lindblads[14,1,2] = scalar_sqrt(kmw_01)
    the_lindblads[15,2,1] = scalar_sqrt(kmw_01)
    
    
    return the_lindblads

def lindblad_quench_simulator(Bfields = None, rate_dictionary = None,
                              nv_theta = 54.7, nv_phi = 0,
                              Bias_field = None):
    """
    Calculates the pl rate of an nv and the steady state populations using 
    Lindblad-GKS equation.
    Rate_dictionary contains: {'kr','k36',k45_6','k60','k6_12','kmw_01','kmw_02','laser_pump'}
    mw_01, mw_02, laser_pump represent incoherent optical and microwave driving.
    'laser_pump' is given in terms of percentage of kr.
    
    Inputs:
        Bfields: array with shape (3,) or (N1,...N2,3)
        rate_dictionary: a dictionary with the decay rates, including laser pump rate
        nv_theta: the NV azimuthal angle in the lab frame
        nv_phi: the NV equatorial angle in the lab frame
        Bias_field: a (3,)-shaped array which represent a Bias field in the lab
                    frame
    Returns:
        pl_rate_out: array with the same shape as Bfields up to the second to 
                     last axis. It contains the 
                     not-normalised PL of the NV.
        steady_states_out: an array with shape Bfields.shape[0:-1] + (7,). E.g., if
                           Bfields.shape = (10,20,3) -> steady_states_out.shape
                           = (10,20,7). It contains the steady states populations,
                           normalised to sum(n_i, i = {0,6}) = 1 and ordered from
                           |0> to |6>
        final_solution: an array with shape Bfields.shape[0:-1] + (7,7),
                        containing the density matrices
    
    ###########################################
    Warning1: has (somewhat minor) bugs concerning the numerical solution of the lindbladian, read
              the description in the doc of the quenchingsimulator_toolbox_lindblad library
    Warning2: ~10x slower than the original quenching simulator in quenchingsimulator_toolbox!
              Might be fixed taking advantage of the Lindbladian symmetries and a better
              (and vectorised) solver.
    ###########################################
    """
    
    default_rate_dictionary = {'kr' : 65,             # The radiative decay rate
                               'k36' : 11,            # Non-radiative to shelving, ms=0
                               'k45_6' : 80,          # Non-radiative to shelving, ms=+-1
                               'k60' : 3,             # Non-radiative from shelving to ms=0
                               'k6_12' : 2.6,         # Non-radiative from shelving to ms=+-1
                               'kmw_01'  : 0,           # Microwave driving (0->1)
                               'kmw_02'  : 0,           # Microwave driving (0->2)
                               'laser_pump' : 0.1}    # Laser driving, percentage of kr 
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
    k6_12 = default_rate_dictionary['k6_12']
    kmw_01 = default_rate_dictionary['kmw_01']
    kmw_02 = default_rate_dictionary['kmw_02']
    laser_pump = default_rate_dictionary['laser_pump'] * kr
    
    zerofield_krausop = zerofield_kraus(kr ,kr , kr,
                                        k36, k45_6, k45_6,
                                        k60, k6_12, k6_12,
                                        laser_pump = laser_pump,
                                        kmw_01=kmw_01, kmw_02=kmw_02)
    if (Bias_field is None):
        Bias_field = 0
    elif (linalg.norm(Bias_field) == 0):
        Bias_field = 0
    Bfields = Bfields + Bias_field
    Bfields = calculate_projection_nvaxis(fields2project = Bfields,
                                          nv_theta = nv_theta, nv_phi = nv_phi)
    
    
    kraus_decaypaths, kraus_rows, kraus_cols = zerofield_krausop.shape
    """
    Generate the Hamiltonians for gs and es, then combine them into
    the seven-level system Hamiltonian
    """
    full_Htot_gs = Htot_gs_fast(Bfields)
    full_Htot_es = Htot_es_fast(Bfields)
    
    master_hamiltonian = np.zeros((bfield_num,7,7), dtype = complex)
    master_hamiltonian[:,0:3,0:3] = full_Htot_gs
    master_hamiltonian[:,3:6,3:6] = full_Htot_es
    master_hamiltonian[:,-1,-1] = 1
    
    #Generate adjoints for code readability
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
    
    decoherence_lindblad = sum_Lk_rho_Lkdag - 0.5*(sum_Lkdag_Lk_rho + sum_rho_Lkdag_Lk)
                            
    
    ham_commutator = (np.einsum("ij,mlk->miljk",
                               identity,master_hamiltonian
                               ) 
    
                     -np.einsum("ijk,mn->ijnkm",
                                master_hamiltonian,
                                identity
                                )
                     )
    
    lindbladian = ((ham_commutator*(-1j*2*np.pi) + decoherence_lindblad).
                    reshape(bfield_num, kraus_rows**2,kraus_cols**2)
                    )
    
    
    """
    Solve the lindbladian
    """
    solution_row = np.eye(kraus_rows).reshape(kraus_rows**2).astype(complex)
    lindbladian[:,0,:] = solution_row #This row is the trace of the density matrix
    
    solution_column = np.zeros(kraus_rows**2)
    solution_column[0] = 1 # Sets the trace of the density matrix to one
    solution_column = np.repeat(solution_column[None,:,None], repeats = bfield_num, axis = 0)
    

    final_solution =  (np.linalg.solve(lindbladian,solution_column).
                       reshape(bfield_num, kraus_rows, kraus_rows).
                       transpose([0,2,1])
                       )

    steady_states_out = np.einsum("ijj->ij",final_solution)
    
    
    
    #Order the populations along in standard order (e.g. |0>,|1>,|2>,...)
    steady_states_out = np.real(steady_states_out[:,[1,2,0,4,5,3,6]])
    """
    I take the real as a dirty trick to get rid of the imaginary part
    stemming from numerical errors. Hopefully with a better (constraneid)
    solver this problem will be fixed.
    """
    
    #The pl rate is proportional to the excited populations in this picture
    pl_rate_out = np.sum(steady_states_out[:,3:6], axis = 1)
    
    
    #Reshape the results according to the shape of the initial B field matrix
    pl_rate_out = np.reshape(pl_rate_out, input_shape[:-1])
    steady_states_out = np.reshape(steady_states_out, input_shape[:-1] + (7,))
    final_solution = np.reshape(final_solution, input_shape[:-1] + (7,7))
    
    
    return pl_rate_out, steady_states_out, final_solution

def calculate_projection_nvaxis(fields2project = None, nv_theta = 0, nv_phi = 0):
    """
    Cheaty function that fixes the issues with the unconstrained Lindblad solver.
    Only calculates the projection of the field along the NV axis and the orthogonal
    component. The B_ort is used as the field on x, B_par is the field on z
    
    fields2project must have shape (N,3)
    """
    bfields_num, _ = fields2project.shape
    
    nv_unit_vec = np.array([cos(nv_phi)*sin(nv_theta),sin(nv_phi)*sin(nv_theta),cos(nv_theta)])
    B_par = np.einsum("ji,i->j",fields2project,nv_unit_vec)
    B_ort = np.square(np.linalg.norm(fields2project,axis=1)) - np.square(B_par)
    B_ort[ B_ort < 0 ] = 0 #This number can never be negative
    B_ort = np.sqrt(B_ort)
    return np.hstack( (B_ort[:,None],np.zeros((bfields_num,1)),B_par[:,None]) )


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt


    norms = np.linspace(0,2000e-3,1000) #np.array([Dgs/gamma2pi])
    normsplot = np.abs(norms)
    ang = 0.1 * np.pi/180 #np.linspace(0,10*np.pi/180,1000)#np.linspace(0,np.pi/2,1000)
    ang2 = 0 * np.pi/180
    
    rate_coeff = np.linspace(1e-3,1e9,100)
    
    plotaxis = norms*1e3#ang * 180/np.pi
    
    Bfields = np.stack((norms*cos(ang2)*np.sin(ang),norms*sin(ang2)*np.sin(ang),np.cos(ang)*norms),axis = 1)
    
    lp = 0.01
    coeff = 1e6
    def rate_dictionary_func(coeff):
        r8= {'kr' : 32.2*coeff,          # The radiative decay rate
             'k36' : 12.6*coeff,          # Non-radiative to shelving, ms=0
             'k45_6' : 80.7*coeff,       # Non-radiative to shelving, ms=+-1
             'k60' : 3.1*coeff,          # Non-radiative from shelving to ms=0
             'k6_12' : 2.5*coeff,        # Non-radiative from shelving to ms=+-1
             'kmw_01': 0,
             'kmw_02': 0,
             'laser_pump' : lp}   # Laser driving, percentage of kr 
        return r8
    
    poppytot = np.zeros((1000,7))
    pltot = np.zeros((1000,))
    powers = np.linspace(1e-3,5,1000)
    for ii,laspump in enumerate(powers):
        pl, poppy , hey = lindblad_quench_simulator(Bfields[0,:], nv_theta = 0,
                                                    rate_dictionary = {'laser_pump':laspump})
        poppytot[ii,:] = poppy
        pltot[ii] = pl
        
    plotaxis = powers
    
    ii = 0
    for line in poppytot[:,:].T:
        plt.plot(plotaxis,line, label = str(ii))
        ii+=1
    plt.legend()
    plt.show()
   

    
    
    plt.plot(plotaxis,pltot/pltot.max())
    plt.show()
