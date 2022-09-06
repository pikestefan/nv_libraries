# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:50:15 2019

@author: Lucio Stefan, lucio.stefan@gmail.com
"""

import numpy as np
from numpy import linalg
from scipy import constants
from math import sin,cos

#### Hamiltonian parameters #####

#Constants
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


#### Functions #####

def Hmagnetic(Bvec):
    """
    Bvec is (Bfield_num,3)-shaped array
    """
    return gamma2pi * np.einsum("njk,mn->mjk",S_array,Bvec)

def Htot_gs(Bvec):
    return H0_gs + Hmagnetic(Bvec)

def Htot_es(Bvec):
    return H0_es + Hmagnetic(Bvec)

def quenching_simulator(Bfields, rate_dictionary = None,
                        nv_theta = 54.7*np.pi/180, nv_phi = 0,
                        rate_coeff = 1e-3, Bias_field = None,
                        correct_for_crossing = False,
                        ):
    """
    Calculates the pl rate of an nv and the steady state populations.
    Rate_dictionary contains: {'kr','k36',k45_6','k60','k6_12','k_01','k_02','laser_pump'}
    k_01 and k_02 are the microwave driving, laser_pump is in percentage of the
    radiative decay rate.
    
    Parameters
    ----------
    Bfields: np.ndarray or list
        array with shape (3,) or (N1,..., Nmax,3)
    rate_dictionary: dictionary
        A dictionary with the decay rates, including pump rate and mw_pump rate.
        If only few rates in the dictionary need to be modified, there's no need
        to input the whole rate dictionary. The remaining keys will retain the default values.
        Ex: quenching_simulator(..., rate_dictionary = {'kr' = 999, 'k_01': 123 })
    nv_theta: float or array_like, optional
        The NV azimuthal angle in the lab frame. It can be either a scalar value
        or a one-dimensional numpy array. Default is ~0.955 rad (54.7 deg).
    nv_phi: float or array_like, optional
        The NV equatorial angle in the lab frame. It can be either a scalar or 
        a numpy array. Default is 0.
    rate_coeff: float
        The coefficient which simulates the collection efficiency. 
    Bias_field: np.array or list
        A (3,)-shaped array which represent a Bias field in the lab frame.
    Returns
    -------
    pl_rate_out: np.ndarray
        Matrix containing the PL of the NV (in MHz or the unit chosen by the user). 
        If nv_theta and nv_phi are scalars, it has shape Bfields.shape[0:-1]. 
        If one of the two angles is an array, the output shape is angle.shape + Bfields.shape[0:-1].
        If both angles are arrays, the output shape is nv_theta.shape + 
        nv_phi.shape + Bfields.shape[0:-1].
    steady_states_out: np.ndarray
        Array containing the steady states populations, normalised to
        sum(n_i, i = {0,6}) = 1. It has shape Bfields.shape[0:-1] + (7,).
        E.g., if Bfields.shape = (10,20,3) -> steady_states_out.shape = (10,20,7).
        If either nv_theta and nv_phi or both are arrays, new dimensions are
        added as described for pl_rate_out.
    """
    #Default rates. All units are MHz
    default_rate_dictionary = {'kr' : 65,             # The radiative decay rate
                               'k36' : 11,            # Non-radiative to shelving, ms=0
                               'k45_6' : 80,          # Non-radiative to shelving, ms=+-1
                               'k60' : 3,             # Non-radiative from shelving to ms=0
                               'k6_12' : 2.6,         # Non-radiative from shelving to ms=+-1
                               'k_01'  : 0,           # Microwave driving (0->1)
                               'k_02'  : 0,           # Microwave driving (0->2)
                               'laser_pump' : 0.1}    # Laser driving, percentage of kr 
    
    #If Bfields or Bias_field are python lists and not numpy arrays, cast them
    #to numpy arrays
    if isinstance(Bfields, list ):
        Bfields = np.array(Bfields)
    if Bias_field is not None and isinstance(Bias_field, list):
        Bias_field = np.array(Bias_field)
    
    #If some keys of the dictionary are fed into the function, replace them into
    #the dictionary
    if rate_dictionary is not None:
        for key, value in rate_dictionary.items():
            if key in default_rate_dictionary:
                default_rate_dictionary[key] = value
                
    kr = default_rate_dictionary['kr']
    k36 = default_rate_dictionary['k36']
    k45_6 = default_rate_dictionary['k45_6']
    k60 = default_rate_dictionary['k60']
    k6_12 =default_rate_dictionary['k6_12']
    k_01 = default_rate_dictionary['k_01']
    k_02 = default_rate_dictionary['k_02']
    laser_pump = default_rate_dictionary['laser_pump']
    
    input_shape = Bfields.shape
    if input_shape == (3,):
        Bfields = np.array([Bfields])
        bfield_num = 1
        input_shape = (1,)
    #If the magnetic field is a multidimensional matrix, reshape it to a  
    # matrix with 3 columns representing the 3 field components and with as many rows
    # as the product of all the dimensions but the last one (which contains the 
    # field components). Allows to handle any matrix dimensionality.
    elif len(input_shape) > 1:
        if input_shape[-1] != 3:
            raise ValueError("The last matrix dimension must contain the 3 " +
                             "magnetic field components.")
        bfield_num = np.prod( input_shape[:-1] )
        Bfields = np.reshape(Bfields, (bfield_num,3))
        
    if (Bias_field is None) or (linalg.norm(Bias_field) == 0):
        Bias_field = 0
        
    Bfields = Bfields + Bias_field
    init_shape = Bfields.shape
    
    #Rotate the fields to the NV reference frame
    Bfields = rotate2nvframe(vectors2transform = Bfields,
                             nv_theta = nv_theta, nv_phi = nv_phi)
    
    if Bfields.shape != init_shape:
        #This means that an array of phi or thetas was requested.
        #Reshape again the Bfields, with the goal of obtaining a
        # ( (len(theta), ) + (len(phi),), input_shape +  )-shaped matrix
        input_shape = Bfields.shape[:-2] + input_shape
        
        Bfields = np.reshape( Bfields, ( np.prod(Bfields.shape[:-1]), 3 ) )
        bfield_num = Bfields.shape[0]
        
    #The unperturbed decay rates (a 7x7 matrix)
    zero_rates = zeroB_decay_mat(kr ,kr , kr,
                                 k36, k45_6, k45_6,
                                 k60, k6_12, k6_12,
                                 laser_pump = laser_pump,
                                 k_mw_01 = k_01, k_mw_02 = k_02)
    
    rate_rows, rate_cols = zero_rates.shape

    #Calculate the B field dependence
    full_Htot_gs = Htot_gs(Bfields)
    full_Htot_es = Htot_es(Bfields)
    
    #Obtain the coefficients of the new eigenstates, which are expressed as linear
    #combinations of the zero-field eigenbasis. The coefficient matrix has shape of
    #(bfield_num, 7, 7)
    coefficient_matrix = find_eigens_and_compose(Htot_gs = full_Htot_gs,
                                                 Htot_es=full_Htot_es,
                                                 correct_for_crossing = correct_for_crossing)
    
    #This is to avoid numerical errors, if the magnetic field vector is zero 
    #then force the transformation matrix to be exactly the identity
    is_norm_zero = (linalg.norm(Bfields, axis = 1) == 0)
    coefficient_matrix[is_norm_zero,:,:] = np.eye(rate_rows)
                                                                            
    #Apply the coefficient matrix, which is also a transformation matrix, to the
    #zero-field decay rates. new_rates = |coeff_mat|^2 . zero_rates . transpose(|coeff_mat|^2).
    #The multiplication is applied along the first dimension of coefficient_matrix
    new_rates = np.matmul(np.square( np.abs(coefficient_matrix) ),
                                     np.matmul(zero_rates,
                                               np.square( np.abs(np.transpose(
                                                   coefficient_matrix, axes = [0,2,1])
                                                                 )
                                                        )
                                               )
                         )

    #Create the solution column vector
    solution_vector = np.zeros( (rate_rows,) )
    solution_vector[0] = 1 #Impose that the sum of the populations needs to be 1
    
    #Create as many copies of the solution column vector as the magnetic fields
    #vectors, creating a (bfield_num, 7, 1)-shaped matrix
    solution_vector = np.repeat(solution_vector[np.newaxis,:,np.newaxis],
                                bfield_num, axis = 0)
    
    #Take the rates, and combine them to form a matrix representing the system of
    #linear equations, with shape (bfield_num, 7, 7). In each of the slices along
    #the first dimensions, the first row is just ones. This imposes that the 
    #sum of the populations is one (see above) and removes the first row (although
    #any row is fine) which is linearly dependent on the others
    rate_equation_matrix = generate_rate_eq(new_rates)
    steady_states = linalg.solve(rate_equation_matrix, solution_vector)[:,:,0]
    
    #Calculate the fluorescence rate from the three excited states and their decay rates
    excited_pops = steady_states[:,3:6,np.newaxis]
    excited_rates = new_rates[:,3:6,0:3]
    pl_rate = np.sum(np.matmul(excited_rates, excited_pops)[:,:,0], axis = 1)
    
    #Reshape the matrices to the original shape
    pl_rate_out = rate_coeff*np.reshape(pl_rate, input_shape[:-1])
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
    
    #Matrix to store the zero-field (both excitationa and decay) rates (without signs),
    #stored as k_ij,  with 0<=i<=6 and 0<=j<=6
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
    #Take the decay matrix, and rearrange it to generate the matrix representing
    #the rate equations (understanding this step is easier with pen and paper!)
    rate_matrix = np.array(decay_matrix.transpose([0,2,1]))
    
    di_inds = np.diag_indices(decay_matrix.shape[1])
    
    rate_matrix[:,di_inds[0],di_inds[1]] = (rate_matrix[:,di_inds[0],di_inds[1]]+
                np.sum( -decay_matrix, axis = 2 )
                )
    rate_matrix[:,0,:] = 1
    return rate_matrix


def find_eigens_and_compose(Htot_gs = None, Htot_es = None,
                                 correct_for_crossing = False):
    
    #This function calculates the two (bfield_num, 3, 3) coefficient matrices
    #of the new ground and excited eigenstates, expressed in the zero-field
    #eigenbasis. After this, place them  into a (bfield_num, 7, 7) block matrix.
    #The top-left block contains the ground state coefficients, the middle block
    #the excited state coefficients, the last block (one element) is the coefficient
    #for the shelving state, which never changes with the magnetic field (spin singlet!)
    b_field_num = Htot_gs.shape[0]

    _, eistates_gs = linalg.eigh( Htot_gs )
    _, eistates_es = linalg.eigh( Htot_es )
    
    if correct_for_crossing:
        
        GS_ground_state_coeff_pos = np.argmax(np.square(np.abs( eistates_gs[:,1,:] )),
                                              axis = 1)
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

def rotate2nvframe(vectors2transform, nv_theta = 0, nv_phi = 0):
    """
    Rot matrix is the product of a rotation about z followed by a rotation about x.
    If both theta and phi are scalars, it returns an array with the same shape.
    
    Parameters
    ----------
    vectors2transform: np.ndarray
        Array with shape (N, 3), representing N 3D vectors which are going to be
        rotated in the NV reference frame.
    nv_theta: float or array_like
        NV azimuthal angle. It can be either a float or a 1D array of angles.
        Default is 0.
    nv_phi: float or array_like
        NV equatorial angle. It can be either a float or a 1D array of angles.
        Default is 0.
    
    Returns
    -------
    rotated: np.ndarray
        The rotated vectors. It both theta and phi are scalars, it has the original
        shape. Otherwise, if one of them is a np.array, the output shape is
        angle.shape + vectors2transform.shape. If both are arrays, then the
        output matrix is nv_theta.shape + nv_phi.shape + vectors2transform.shape.
    """
    reps = vectors2transform.shape[0]
    
    #I distinguish between the case of two scalar angles and the rest to keep
    #computation speed at its max. If they are not arrays, I avoid extra matrix
    #operations, e.g. the use of np.cos (slower than cos) and np.einsum.
    if (not isinstance(nv_theta, (np.ndarray, list)) ) and (not isinstance(nv_phi, (np.ndarray, list)) ):
        #The rotation is already the product of two rotation matrices. The product
        #is calculated by hand to save the unnecessary time spent into multiplying
        #two matrices.
        rot_matrix = np.array([ [cos(nv_phi)*cos(nv_theta),
                                 sin(nv_phi)*cos(nv_theta),
                                 -sin(nv_theta)],
                                [-sin(nv_phi),
                                 cos(nv_phi),
                                 0],
                                [sin(nv_theta)*cos(nv_phi),
                                 sin(nv_phi)*sin(nv_theta),
                                 cos(nv_theta)]
                                ] )
        rot_matrix = np.repeat(rot_matrix[np.newaxis,:,:],reps,axis = 0)
        rotated = np.matmul( rot_matrix,  vectors2transform[:,:,None] )
        rotated = rotated[:,:,0]
        
    else:
        #This part handles the cases in which nv_theta and nv_phi can be arrays
        #(or python lists).
        #Casts them to np.ndarray instances if they are scalars, then add
        #dimensions according to the dimensionality of nv_theta and nv_phi
        thetascalar = False
        if ( not isinstance(nv_theta, (np.ndarray, list) ) ):
            nv_theta = np.array([nv_theta])
            thetascalar = True
        elif isinstance(nv_theta, list ):
            nv_theta = np.array(nv_theta)
            
        phiscalar = False  
        if ( not isinstance(nv_phi, (np.ndarray, list) ) ):
            nv_phi = np.array([nv_phi])
            phiscalar = True
        elif isinstance(nv_phi, list ):
            nv_phi = np.array(nv_phi)
            
            
        rot_matrix = np.array([
                        [np.cos(nv_phi[None,:]) * np.cos(nv_theta[:,None]),
                         np.sin(nv_phi[None,:]) * np.cos(nv_theta[:,None]),
                         -np.sin(nv_theta[:,None]) * np.ones((1,) + nv_phi.shape)],

                        [-np.sin(nv_phi[None,:]) * np.ones( nv_theta.shape + (1,) ),
                          np.cos(nv_phi[None,:]) * np.ones( nv_theta.shape + (1,) ),
                          np.zeros(nv_theta.shape + nv_phi.shape)],
                        
                        [np.sin(nv_theta[:,None])*np.cos(nv_phi[None,:]), 
                         np.sin(nv_phi[None,:])*np.sin(nv_theta[:,None]),
                         np.cos(nv_theta[:,None]) * np.ones((1,) + nv_phi.shape)]
                      ] )
        rotated = np.einsum("nojk,mo->jkmn", rot_matrix, vectors2transform)
        
        if phiscalar and not thetascalar:
            rotated = rotated[:,0]
        if thetascalar and not phiscalar:
            rotated = rotated[0,:]
        
    return rotated



if __name__ == '__main__':
    """
    Code testing section!
    """
    import matplotlib.pyplot as plt
    
    
    norms = np.linspace(0,55e-3,5000) #np.array([Dgs/gamma2pi])
    normsplot = np.abs(norms)
    ang =0.1 *np.pi/180 #np.linspace(0,10*np.pi/180,1000)#np.linspace(0,np.pi/2,1000)
    ang2 = 0 * np.pi/180
    
    plotaxis = norms#ang * 180/np.pi
    
    Bfields = np.stack((norms*cos(ang2)*np.sin(ang),norms*sin(ang2)*np.sin(ang),np.cos(ang)*norms),axis = 1)

    pl,pops = quenching_simulator(Bfields, Bias_field = [0,0,1e-3],nv_theta=[0,0.1],correct_for_crossing=True)
    
    """
    for pop in pops[:,:].T:
        plt.plot(plotaxis,pop)
    plt.show()
    
    plt.plot(plotaxis,pl/pl.max())
    """