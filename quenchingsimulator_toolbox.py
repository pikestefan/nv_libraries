# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:50:15 2019

@author: Lucio
"""

import numpy as np
from numpy import linalg
from scipy import constants
from math import sin,cos

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
gamma2pi = gamma2pi / 1e9 #GHz/T

Dgs = 2.87
Des = 1.42
Egs = 2.5e-3

Sz = np.array( [[1,0,0],
                [0,0,0],
                [0,0,-1]])
Sx = (1/np.sqrt(2)) * np.array([[0,1,0],
                                [1,0,1],
                                [0,1,0]])
Sy = (1j / np.sqrt(2) ) * np.array( [[0,-1,0],
                                     [1,0,-1],
                                     [0,1,0]] )

S_array = np.stack([Sx, Sy, Sz], axis = 0)

Sz_sq = np.dot(Sz,Sz)
Sx_sq_Sy_sq = np.dot(Sx,Sx) - np.dot(Sy,Sy)

H0_gs = Dgs * Sz_sq# + Egs * Sx_sq_Sy_sq
H0_es = Des * Sz_sq

Hmagnetic = lambda Bvec: gamma2pi * (Bvec[0]*Sx + Bvec[1]*Sy + Bvec[2]*Sz)

Htot_gs = lambda Bvec: H0_gs + Hmagnetic(Bvec)
Htot_es = lambda Bvec: H0_es + Hmagnetic(Bvec)

def Hmagnetic_fast(Bvec):
    """
    Bvec is (Bfield_num,3)-shaped array
    """
    return gamma2pi * np.einsum("njk,mn->mjk",S_array,Bvec)


def Htot_gs_fast(Bvec):
    return H0_gs + Hmagnetic_fast(Bvec)

def Htot_es_fast(Bvec):
    return H0_es + Hmagnetic_fast(Bvec)
#def Htot_gs_fast(Bvec):
"""
Bvec is (Bfield_num,3)-shaped array
"""

def quenching_calculator_fast(Bfields = None, rate_dictionary = None,
                              nv_theta = 54.7, nv_phi = 0, rate_coeff = 1e-3,
                              Bias_field = None,correct_for_crossing = False,
                              ):
    """
    
    Calculates the pl rate of an nv and the steady state populations.
    Rate_dictionary contains: {'kr','k36',k45_6','k60','k6_12','mw_rate','laser_pump'}
    
    Inputs:
        Bfields: array with shape (3,) or (N1,...N2,3)
        rate_dictionary: a dictionary with the decay rates, including pump rate
                         and mw_pump rate
        nv_theta: the NV azimuthal angle in the lab frame
        nv_phi: the NV equatorial angle in the lab frame
        rate_coeff: the coefficient which simulates the collection efficiency
        Bias_field: a (3,)-shaped array which represent a Bias field in the lab
                    frame
    Returns:
        pl_rate_out: array with the same shape as Bfields up to the second to 
                     last axis. It contains the 
                     not-normalised PL of the NV.
        steady_states_out: a array shape Bfields.shape[0:-1] + (7,). E.g., if
                           Bfields.shape = (10,20,3) -> steady_states_out.shape
                           = (10,20,7). It contains the steady states populations,
                           normalised to sum(n_i, i = {0,6}) = 1
                           
    A quenching calculator under steroids. All the functions and arrays are vectorised
    to N-dimensional matrices to take full advantage of the numpy functions speed.
    Rough benchmarking shows speedups from 5x to 10x compared to quenching_calculator.
        
    """
    
    default_rate_dictionary = {'kr' : 32.2,          # The radiative decay rate
                               'k36' : 12.6,          # Non-radiative to shelving, ms=0
                               'k45_6' : 80.7,       # Non-radiative to shelving, ms=+-1
                               'k60' : 3.1,          # Non-radiative from shelving to ms=0
                               'k6_12' : 2.5,        # Non-radiative from shelving to ms=+-1
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
    elif len(input_shape) > 1:
        bfield_num = np.prod( input_shape[:-1] )
        Bfields = np.reshape(Bfields, (bfield_num,3))
        
    
    kr = default_rate_dictionary['kr']
    k36 = default_rate_dictionary['k36']
    k45_6 = default_rate_dictionary['k45_6']
    k60 = default_rate_dictionary['k60']
    k6_12 =default_rate_dictionary['k6_12']
    mw_rate = default_rate_dictionary['mw_rate']
    laser_pump = default_rate_dictionary['laser_pump']

    
    #The unperturbed decay rates
    zero_rates = zeroB_decay_mat(kr ,kr , kr,
                                 k36, k45_6, k45_6,
                                 k60, k6_12, k6_12,
                                 laser_pump = laser_pump,
                                 k_mw_01 = mw_rate, k_mw_02 = mw_rate)
    
    new_rates = np.zeros( (bfield_num,) + zero_rates.shape  ) # matrix for decay rates
    rate_equation_matrix = np.zeros((bfield_num,) + zero_rates.shape) # matrix for the rate equations
    
    rate_rows, rate_cols = zero_rates.shape
    
    #Matrix to store the eigenstate coefficients
    #coefficient_matrix = np.array(new_rates, dtype = np.complex)
    
    #levels = np.zeros( (len(bnorm),3) ) Use it to store one level coefficients
    solution_vector = np.zeros( (rate_rows,) )
    solution_vector[0] = 1
    solution_vector = np.repeat(solution_vector[np.newaxis,:,np.newaxis], bfield_num, axis = 0)
    steady_states = np.zeros( (bfield_num, rate_rows) )
    
    if (Bias_field is None):
        Bias_field = 0
    elif (linalg.norm(Bias_field) == 0):
        Bias_field = 0
    Bfields = Bfields + Bias_field
    Bfields = rotate2nvframev_fast(vectors2transform = Bfields[:,:,np.newaxis], nv_theta = nv_theta, nv_phi = nv_phi)

    norm_is_zero = (linalg.norm(Bfields, axis = 1) == 0)

    #Calculate the B field dependence
    full_Htot_gs = Htot_gs_fast(Bfields)
    full_Htot_es = Htot_es_fast(Bfields)
    coefficient_matrix = find_eigens_and_compose_fast(Htot_gs = full_Htot_gs, Htot_es=full_Htot_es,
                                                      correct_for_crossing = correct_for_crossing)
    coefficient_matrix[norm_is_zero,:,:] = np.eye(rate_rows)
    
    
    new_rates = np.matmul(np.square( np.abs(coefficient_matrix) ),
                                     np.matmul(zero_rates,
                                               np.square( np.abs(np.transpose(coefficient_matrix, axes = [0,2,1])) )
                                               )
                                      )

    rate_equation_matrix = generate_rate_eq_fast(new_rates)
    steady_states = linalg.solve(rate_equation_matrix, solution_vector)[:,:,0]
    
    #Calculate the fluorescence rate from the three excited states and their decay rates
    excited_pops = steady_states[:,3:6,np.newaxis]
    excited_rates = new_rates[:,3:6,0:3]
    pl_rate = np.sum(np.matmul(excited_rates, excited_pops)[:,:,0], axis = 1)
    
    #Reshape the matrices to the original shape
    pl_rate_out = rate_coeff*np.reshape(pl_rate, input_shape[:-1])
    steady_states_out = np.reshape(steady_states, input_shape[:-1] + (7,))
    
    return pl_rate_out, steady_states_out

def quenching_calculator(Bfields = None, rate_dictionary = None, 
                         nv_theta = 55.6, nv_phi = 0, rate_coeff = 1e-3,
                         Bias_field = None):
    
    """
    Calculates the pl rate of an nv and the steady state populations.
    
    Inputs:
        Bfields: array with shape (3,) or (N1,...N2,3)
        rate_dictionary: a dictionary with the decay rates, including pump rate
                         and mw_pump rate
        nv_theta: the NV azimuthal angle in the lab frame
        nv_phi: the NV equatorial angle in the lab frame
        rate_coeff: the coefficient which simulates the collection efficiency
        Bias_field: a (3,)-shaped array which represent a Bias field in the lab
                    frame
    Returns:
        pl_rate_out: array with the same shape as Bfields up to the second to 
                     last axis. It contains the 
                     not-normalised PL of the NV.
        steady_states_out: a array shape Bfields.shape[0:-1] + (7,). E.g., if
                           Bfields.shape = (10,20,3) -> steady_states_out.shape
                           = (10,20,7). It contains the steady states populations,
                           normalised to sum(n_i, i = {0,6}) = 1
        
    """
    default_rate_dictionary = {'kr' : 32.2,
                               'k36' : 12.6,
                               'k45_6' : 80.7,
                               'k60' : 3.1,
                               'k6_12' : 2.5,
                               'mw_rate' : 0,
                               'laser_pump' : 0.1}
    
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
    mw_rate = default_rate_dictionary['mw_rate']
    laser_pump = default_rate_dictionary['laser_pump']
    
    #The unperturbed decay rates
    zero_rates = zeroB_decay_mat(kr ,kr , kr,
                                 k36, k45_6, k45_6,
                                 k60, k6_12, k6_12,
                                 laser_pump = laser_pump,
                                 k_mw_01 = mw_rate, k_mw_02 = mw_rate)
    
    new_rates = np.zeros( zero_rates.shape + (bfield_num,) ) # matrix for decay rates
    rate_equation_matrix = np.array(new_rates) # matrix for the rate equations
    
    rate_rows, rate_cols = zero_rates.shape
    
    #Matrix to store the eigenstate coefficients
    coefficient_matrix = np.array(new_rates, dtype = np.complex)
    
    #levels = np.zeros( (len(bnorm),3) ) Use it to store one level coefficients
    solution_vector = np.zeros( (rate_rows,) )
    solution_vector[0] = 1
    steady_states = np.zeros( (bfield_num, rate_rows) )
    
    if (Bias_field is None):
        Bias_field = 0
    elif (linalg.norm(Bias_field) == 0):
        Bias_field = 0
    
    #Calculate the B field dependence
    for kk, field in enumerate(Bfields):
        field = field + Bias_field
        field = rotate2nvframe(vector2transform = field, nv_theta = nv_theta, nv_phi = nv_phi)
        #print(field)
        #print(linalg.norm(field))
        if (linalg.norm(field) == 0):
            matrix = np.eye(rate_rows)
        else:
            matrix = find_eigens_and_compose(Htot_gs(field), Htot_es(field))
        coefficient_matrix[:,:,kk] = matrix
        
        for ii in range(rate_rows):
            for jj in range(rate_cols):
                new_rates[ii,jj,kk] = np.dot( np.square( np.abs(matrix[ii,:]) ),
                                              np.dot(zero_rates,
                                                     np.square( np.abs(matrix[jj,:]))
                                                    )
                                            )                                      
        rate_equation_matrix[:,:,kk] = generate_rate_eq(new_rates[:,:,kk])
        steady_states[kk,:] = linalg.solve(rate_equation_matrix[:,:,kk], solution_vector)
        
                                      
    #print(new_rates[:,:,0] == new_rates_alt[:,:,0])
    #Calculate the fluorescence rate from the three excited states and their decay rates
    pl_rate = np.zeros( (bfield_num,) )
    for ii, levels_at_B in enumerate(steady_states):
        excited_pops = levels_at_B[3:6]
        excited_rates = np.array( new_rates[3:6,0:3, ii] )
        pl_rate[ii] = rate_coeff*np.sum(  np.dot(excited_rates.T, excited_pops) )
    pl_rate_out = np.reshape(pl_rate, input_shape[:-1])
    steady_states_out = np.reshape(steady_states, input_shape[:-1] + (7,))
    
    return pl_rate_out, steady_states_out


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

def zeroB_decay_mat(k30, k41, k52, #Radiative decay rates
                    k36, k46,  k56, #Nonrad. decays to the intersystem crossing
                    k60, k61, k62, #Nonrad. decays from the intersystem crossing
                    laser_pump = 0,
                    k_mw_01 = 0, k_mw_02 = 0
                    ):
    
    matrix = np.array( [
                        [0, k_mw_01, k_mw_02, laser_pump*k30, 0, 0, 0],
                        [k_mw_01, 0, 0, 0, laser_pump*k41, 0, 0],
                        [k_mw_02, 0, 0, 0, 0, laser_pump*k52, 0],
                        [k30, 0, 0, 0, 0, 0, k36],
                        [0, k41, 0, 0, 0, 0, k46],
                        [0, 0, k52, 0, 0, 0, k56],
                        [k60, k61, k62, 0, 0, 0, 0]
                         ])
    return matrix

def generate_rate_eq(decay_matrix):
    rate_matrix = np.array(decay_matrix.T)
    
    for ii in range(0, len(rate_matrix)):
        rate_matrix[ii,ii] +=  - np.sum( decay_matrix[ii,:] )
    rate_matrix[0,:] = 1
    return rate_matrix
    
def generate_rate_eq_fast(decay_matrix):
    rate_matrix = np.array(decay_matrix.transpose([0,2,1]))

    di_inds = np.diag_indices(decay_matrix.shape[1])
    
    rate_matrix[:,di_inds[0],di_inds[1]] = (rate_matrix[:,di_inds[0],di_inds[1]]+
                np.sum( -decay_matrix, axis = 2 )
                )
    rate_matrix[:,0,:] = 1
    return rate_matrix

# Warning: this works only for a strain-free zero-field Hamiltonian
# The Sz eigenstates commute with the hamiltonian hDSz^2, so 
def find_eigens_and_compose(Htot_gs = None, Htot_es = None):
    
    try:
        _, eistates_gs = linalg.eigh( Htot_gs )
        _, eistates_es = linalg.eigh( Htot_es )
        
    except:
        raise ValueError("find_eigs_and_compose: One of the two matrices is not defined!")
    
    eistates_gs = eistates_gs.T
    eistates_gs = eistates_gs[:, [1,2,0]]
    
    eistates_es = eistates_es.T
    eistates_es = eistates_es[:,[1,2,0]]    
    
    full_matrix = np.zeros( (7,7), dtype = complex )
    full_matrix[0:3,0:3] = eistates_gs
    full_matrix[3:6,3:6] = eistates_es
    full_matrix[-1,-1] = 1
    
    return full_matrix

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
        
        

    eistates_gs = np.transpose(eistates_gs, axes = [0,2,1])
    eistates_gs = eistates_gs[:,:, [1,2,0]]
    
    eistates_es = np.transpose(eistates_es, axes = [0,2,1])
    eistates_es = eistates_es[:,:,[1,2,0]]    
    
    full_matrix = np.zeros( (b_field_num,) + (7,7), dtype = complex )
    full_matrix[:,0:3,0:3] = eistates_gs
    full_matrix[:,3:6,3:6] = eistates_es
    full_matrix[:,-1,-1] = 1
    return full_matrix

def find_eigens_eivals(Htot = None):
    
    try:
        eivals, eistates = linalg.eigh( Htot )
        
    except:
        raise ValueError("find_eigens_eivals: matrix is not defined!")   
    
    return eivals, eistates.T

def rotate2nvframe(vector2transform = np.array([0,0,0]), nv_theta = 0, nv_phi = 0):
    rot_aboutz = np.array([ [np.cos(nv_phi), np.sin(nv_phi), 0],
                            [-np.sin(nv_phi), np.cos(nv_phi), 0],
                             [0, 0, 1] ])
    rot_aboutx = np.array( [ [np.cos(nv_theta), 0, -np.sin(nv_theta)],
                             [0, 1, 0],
                             [np.sin(nv_theta), 0, np.cos(nv_theta)] ] )
    return np.matmul( np.matmul(rot_aboutx, rot_aboutz),  vector2transform)


def rotate2nvframev_fast(vectors2transform = np.zeros((2,3,1)), nv_theta = 0, nv_phi = 0):
    reps = vectors2transform.shape[0]
    
    #Rot matrix is the product of a rotation about z followed by
    #a rotation about x
    rot_matrix = np.array([ [cos(nv_phi)*cos(nv_theta), sin(nv_phi)*cos(nv_theta), -sin(nv_theta)],
                            [-sin(nv_phi), cos(nv_phi), 0],
                             [sin(nv_theta)*cos(nv_phi), sin(nv_phi)*sin(nv_theta), cos(nv_theta)] ] )
    
    rot_matrix = np.repeat(rot_matrix[np.newaxis,:,:],reps,axis = 0)
    
    result = np.matmul( rot_matrix,  vectors2transform )
    
    return result[:,:,0]

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    
    norms = np.linspace(0,55e-3,5000) #np.array([Dgs/gamma2pi])
    normsplot = np.abs(norms)
    ang =0.1 *np.pi/180 #np.linspace(0,10*np.pi/180,1000)#np.linspace(0,np.pi/2,1000)
    ang2 = 0 * np.pi/180
    
    plotaxis = norms#ang * 180/np.pi
    
    Bfields = np.stack((norms*cos(ang2)*np.sin(ang),norms*sin(ang2)*np.sin(ang),np.cos(ang)*norms),axis = 1)
    
      
    
    pl,pops = quenching_calculator_fast(Bfields,nv_theta=0,correct_for_crossing=True)
    
    for pop in pops[:,:].T:
        plt.plot(plotaxis,pop)
    plt.show()
    
    plt.plot(plotaxis,pl/pl.max())
    """
    
    test_dic = {'kr' : 67.7,          # The radiative decay rate
               'k36' : 6.4,          # Non-radiative to shelving, ms=0
               'k45_6' : 50.7,       # Non-radiative to shelving, ms=+-1
               'k60' : 0.7,          # Non-radiative from shelving to ms=0
               'k6_12' : 0.6,        # Non-radiative from shelving to ms=+-1
               'mw_rate' : 0,        # Microwave driving
               'laser_pump' : 0.1}   # Laser driving, percentage of kr
    iterdic = dict.fromkeys(test_dic.keys())
    norms = Des/gamma2pi
    ang = 10*np.pi/180#np.linspace(0,np.pi/2,1000)
    ang2 = 0
    Bfields = np.array([norms*cos(ang2)*np.sin(ang),norms*sin(ang2)*np.sin(ang),np.cos(ang)*norms])
    
    coeffs = np.linspace(1e-20,1e6,1000)
    
    pls = np.zeros(len(coeffs))
    for ii,alpha in enumerate(coeffs):
        for key,val in test_dic.items():
            iterdic[key]=val*alpha
        pl,_ = quenching_calculator_fast(Bfields,nv_theta=0,correct_for_crossing=True,rate_dictionary=test_dic)
        pls[ii] = pl
    plt.plot(coeffs,pls)
    
    
    mw_rate = 0.1
    
    rate = 0.1
    
        
    curr_dic = {"laser_pump" : rate, "mw_rate" : mw_rate}
    curr_dic_nomw = {"laser_pump" : rate}
    
    with_mw, _ = quenching_calculator_fast(Bfields = Bfields,
                                           rate_dictionary = curr_dic)
    without_mw,pops = quenching_calculator_fast(Bfields = Bfields,
                                             rate_dictionary = curr_dic_nomw)
    
    contrasts = (without_mw - with_mw)/without_mw
    no_mw_pl = without_mw
    
    plt.plot(plotaxis, without_mw)
    
    fig, ax = plt.subplots()
    ax.plot(plotaxis, contrasts)
    """