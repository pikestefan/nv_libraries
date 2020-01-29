# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:23:59 2018

@author: ls746_2
"""

import numpy as np
from scipy import optimize

def remove_polynomial_bg(data, degree_x, degree_y, init_params = None ):
    """
    Removes polynomial background from image
    """
    if init_params is None:
        init_params = np.ones( (degree_x + degree_y + 2, ) )
    
    Y, X = np.indices(data.shape)
    
    def poly_surf( pars, x, y ):
            if (len(pars) != degree_x + degree_y + 2):
                raise Exception( "Parameters dimension and degrees must match!" )
                
            x_coeffs = pars[0:degree_x + 1]
            y_coeffs = pars[-(degree_y + 1):]
            x_polyplane = np.array( [ coeff*np.power(x, index) for index, coeff in enumerate(x_coeffs) ] ).sum(axis = 0)
            y_polyplane = np.array( [ coeff*np.power(y, index) for index, coeff in enumerate(y_coeffs) ] ).sum(axis = 0)
            return x_polyplane + y_polyplane
       
    minimising_func = lambda pars: np.ravel( data - poly_surf( pars, X, Y ) )
    
    fitted, _ = optimize.leastsq( minimising_func, x0 = init_params ) 
    return poly_surf( fitted, X, Y )

def median_correction(data_2d, add_mean = False):
    """
    Corrects line error in AFM scans by removing the median of the slow axis
    """
    return np.array( [line - np.median(line) for line in data_2d] ) + add_mean * data_2d.mean()

def polyline_correction( data_2d, degree = 1,  add_mean = False, fit_pars = None):
    
    if fit_pars is None:
        fit_pars = np.ones( (degree + 1,) )
    
    output_matrix = np.array( data_2d )
    
    _, columns = data_2d.shape
    xaxis = np.arange(0, columns)
    
    def polyfunc(pars, x):
        output = np.array( [ np.power(x, index)*coeff for index, coeff in enumerate(pars) ] ).sum(axis = 0)
        return output
    
    def minimisation_problem( pars, x,  data ):
        return data - polyfunc( pars, x )
    
    for index, line in enumerate(data_2d):
        optimal_pars, _ = optimize.leastsq( minimisation_problem, x0 = fit_pars, args = ( xaxis, line ) )
        output_matrix[ index, : ] -= polyfunc( optimal_pars, xaxis )
        
    return output_matrix + add_mean * data_2d.mean()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    direc = r"S:\low_temp_afm\data\2018_10_22\magscan_002"
    topography = np.loadtxt(direc + r"\magscan_zout.txt")
    
    corrected = polyline_correct(topography, degree = 2)
    
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    p = plt.pcolor(corrected, cmap = cm.Greys)
    cb = fig.colorbar(p)