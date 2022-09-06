# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:23:59 2018

@author: Lucio Stefan, lucio.stefan@gmail.com
"""

import numpy as np
from scipy import optimize
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.colors as mplcol
from math import floor, ceil
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
    
    return data_2d - np.median(data_2d, axis=0) + add_mean * data_2d.mean()

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

def plot2d_with_size_label(Zdata, Xdata = None, Ydata = None,
                           rect_coords = (0,0), rect_thick = 0.02, length = 1,
                           pad = 0.02, axis_handle = None, fig_handle = None,
                           *args, **kwargs):
    
    """
    Function to generate a length label on 2D plots. Using the specified length,
    the function makes a 2D plot without axes displaying a bar and text which
    indicate the dimensions. Label sizes and coordinates are given in percentages
    of the x and y coordinates.
    
    Parameters
    ----------
    Zdata: 2D np.array
        The grid of data making the 2D plot.
    Xdata: 2D np.array, optional
        Needs to have the same shape as Zdata, it is the grid of x coordinates.
        Default is None.
    Ydata: 2D np.array, optional
        Needs to have the same shape as Zdata, it is the grid of y coordinates.
        Default is None.
    rect_coords: tuple of floats, optional
        Coordinates of the label relative to the image coordinates. Default is (0,0).
    rect_thick: float, optional
        How much the label background extends around the text, relative, to the 
        image coordinates. Default is 0.02.
    length: float, optional
        The length represented by the bar and displayed in the text. Default is 1.
    pad: float, optional
        How far the text is from the length bar. Default is 0.02.
    axis_handle:
        The handle of the axis where the plot is made. If not given, the axis
        is taken with plt.gca().
    fig_handle:
        The handle of the axis where the plot is made. If not given, the axis
        is taken with plt.gcf().
        
    Returns
    ------
    plot_handle: 
        The handle of the plot.
        
    Optional parameters
    -------------------
    **kwargs:
    unit: str
        The string appendend to indicate the length units in the label. Default is 'nm'.
    label_precision: str
        The float precision of the number displayed in the label. The precision
        indicator follows the python string format method syntax. Default is ":0f".
    label_textcolor: str or RBGA tuple
        The color of the text. Default is 'k'.
    fontsize: int
        The size of the text in pts. Default is 12.
    with_bg: bool
        Display a background behind the text. Default is True.
    which_plot_method: str
        Which 2D plotting to use. The options are 'imshow', 'pcolor' and 'pcolormesh'.
        Default is 'imshow'.
    bg_color: 'w' or RGBA tuple
        The color of the label background.
    bg_extra_size: float
        Add extra background around the label. Default is 0.01.
    return_txt_handle: bool
        Require to return both the plot and text handles. Default si False.
    txt_settings: dict
        A dictionary to pass to the **kwargs of plt.text()
        
    Other *args or **kwargs are passed to the plotting function.
    """
    
    unit = kwargs.pop('unit', 'nm')
    label_precision = kwargs.pop('label_precision', ":.0f")
    label_textcolor = kwargs.pop('label_textcolor', 'k')
    fontsize = kwargs.pop('fontsize', 12)
    with_bg = kwargs.pop('with_bg', True)
    which_plot_method = kwargs.pop('which_plot_method', 'imshow')
    bg_color = kwargs.pop('bg_color', 'w')
    bg_extra_size = kwargs.pop('bg_extra_size', 0.01)
    return_txt_handle = kwargs.pop('return_txt_handle', False)
    txt_settings = kwargs.pop('txt_settings',{})
    
    pixely,pixelx = Zdata.shape
    textstring = "{" + label_precision + "}"
    textstring = textstring.format(length)
    texstring = r"$\mathrm{" + textstring + "\," + unit +  "}$"
    
    if (Xdata is None) or (Ydata is None):
        Xdata, Ydata = np.meshgrid(np.arange(pixelx),np.arange(pixely))
        
    rect_hei = rect_thick*(Ydata.max() - Ydata.min())
    text_pad = pad*(Ydata.max() - Ydata.min())
    
    if axis_handle is None:
        axis_handle = plt.gca()
    if fig_handle is None:
        fig_handle = plt.gcf()
        
    if which_plot_method == 'imshow':
        plot_handle = axis_handle.imshow(Zdata,
                                         extent = (Xdata.min(),Xdata.max(),Ydata.min(),Ydata.max()),
                                         *args, **kwargs)
    elif which_plot_method == 'pcolormesh':
        plot_handle = axis_handle.pcolormesh(Xdata,Ydata,Zdata, *args, **kwargs)
    elif which_plot_method == 'pcolor':
        plot_handle = axis_handle.pcolor(Xdata,Ydata,Zdata, *args, **kwargs)
        
    axis_handle.axis('off')
    
    text_y_coord = rect_coords[1] + (text_pad + rect_hei)
    text_object = axis_handle.text(rect_coords[0]+length/2,
                                   text_y_coord,
                                   texstring, 
                                   fontsize = fontsize, 
                                   ha = 'center', color = label_textcolor, **txt_settings)
    
    
    #This code section gets the width and height of the text box
    renderer = fig_handle.canvas.get_renderer()
    bbox = text_object.get_window_extent(renderer = renderer) #bbox in display coordinates
    display_coord2data_coord_transform = axis_handle.transData.inverted() #get the coord transform from display to data
    bbox = bbox.transformed( display_coord2data_coord_transform ) #convert bbox into data coordinates
    
    text_w, text_h = bbox.width, abs(bbox.height)
    
    #Length bar
    size_indicator = Rectangle(rect_coords, width=length, height = rect_hei,
                               edgecolor = label_textcolor,
                               facecolor = label_textcolor)
    if with_bg:
        #Calculate the size of the background box
        tot_width = length if (length > text_w) else text_w
        
        #Calculate the bottom left corner position and the centre of the box

        tot_height = (bbox.y0 - rect_coords[1]) + text_h
        
        ratio = tot_width / tot_height
        
        tot_y0 = rect_coords[1]
        tot_x0 = rect_coords[0] if (rect_coords[0] < bbox.x0) else bbox.x0
        
        bbox_centre_x0 = tot_x0 + tot_width/2
        bbox_centre_y0 = tot_y0 + tot_height/2
        
        #Increase the size of the bbox
        tot_width = (1 + bg_extra_size) * tot_width
        tot_height = tot_height + bg_extra_size * tot_height * ratio #tot_height + bg_extra_size * tot_width
        
        #These are the coordinates of the enlarged background box
        tot_x0 = bbox_centre_x0 - tot_width/2
        tot_y0 = bbox_centre_y0 - tot_height/2
        
        bg = Rectangle((tot_x0, tot_y0), width=tot_width, height = tot_height,
                        edgecolor = bg_color,
                        facecolor = bg_color)
        
        axis_handle.add_patch(bg)
    axis_handle.add_patch(size_indicator)
    
    if return_txt_handle:
        return plot_handle, text_object
    else:
        return plot_handle

def add_size_label(xax = None, yax = None, matshape = None,
                   rect_coords = (0,0), rect_thick = 0.02, length = 1,
                   pad = 0.02, axis_handle = None, fig_handle = None, 
                   *args, **kwargs):
    unit = kwargs.pop('unit', 'nm')
    label_precision = kwargs.pop('label_precision', ":.0f")
    label_textcolor = kwargs.pop('label_textcolor', 'k')
    fontsize = kwargs.pop('fontsize', 12)
    with_bg = kwargs.pop('with_bg', True)
    bg_color = kwargs.pop('bg_color', 'w')
    bg_extra_size = kwargs.pop('bg_extra_size', 0.01)
    
    pixely,pixelx = matshape
    textstring = "{" + label_precision + "}"
    textstring = textstring.format(length)
    texstring = r"$\mathrm{" + textstring + "\," + unit +  "}$"
    
    if (xax is None) or (yax is None):
        xax, yax = np.arange(pixelx), np.arange(pixely)
        
    if axis_handle is None:
        axis_handle = plt.gca()
    if fig_handle is None:
        fig_handle = plt.gcf()
        
    rect_hei = rect_thick*(yax.max() - yax.min())
    text_pad = pad*(xax.max() - xax.min())
    
    text_y_coord = rect_coords[1] + (text_pad + rect_hei)
    text_object = axis_handle.text(rect_coords[0]+length/2,
                                   text_y_coord,
                                   texstring, 
                                   fontsize = fontsize, 
                                   ha = 'center', color = label_textcolor)
    
    
    #This code section gets the width and height of the text box
    renderer = fig_handle.canvas.get_renderer()
    bbox = text_object.get_window_extent(renderer = renderer) #bbox in display coordinates
    display_coord2data_coord_transform = axis_handle.transData.inverted() #get the coord transform from display to data
    bbox = bbox.transformed( display_coord2data_coord_transform ) #convert bbox into data coordinates
    
    text_w, text_h = bbox.width, abs(bbox.height)
    
    #Length bar
    size_indicator = Rectangle(rect_coords, width=length, height = rect_hei,
                               edgecolor = label_textcolor,
                               facecolor = label_textcolor)
    if with_bg:
        #Calculate the size of the background box
        tot_width = length if (length > text_w) else text_w
        
        #Calculate the bottom left corner position and the centre of the box

        tot_height = (bbox.y0 - rect_coords[1]) + text_h
        
        ratio = tot_width / tot_height
        
        tot_y0 = rect_coords[1]
        tot_x0 = rect_coords[0] if (rect_coords[0] < bbox.x0) else bbox.x0
        
        bbox_centre_x0 = tot_x0 + tot_width/2
        bbox_centre_y0 = tot_y0 + tot_height/2
        
        #Increase the size of the bbox
        tot_width = (1 + bg_extra_size) * tot_width
        tot_height = tot_height + bg_extra_size * tot_height * ratio #tot_height + bg_extra_size * tot_width
        
        #These are the coordinates of the enlarged background box
        tot_x0 = bbox_centre_x0 - tot_width/2
        tot_y0 = bbox_centre_y0 - tot_height/2
        
        bg = Rectangle((tot_x0, tot_y0), width=tot_width, height = tot_height,
                        edgecolor = bg_color,
                        facecolor = bg_color)
        
        axis_handle.add_patch(bg)
    axis_handle.add_patch(size_indicator)
    
    return None
    
def colorbar_arbitrary_position(mappable = None, side = "right", pad = 0.01, width = 0.01, height = 1,
                                orientation = "vertical", shift = (0,0),
                                *args, **kwargs):
    """
    Creates a colorbar that doesn't suffer from rescaling issues.
    Mappable is the plot handle. *args and **kwargs are passed to the colorbar
    """
    ax = mappable.axes
    fig = ax.figure
    locpos = ax.get_position()
    
    x_shift, y_shift = shift
    x_shift, y_shift = x_shift*locpos.width, y_shift*locpos.height
    
    if orientation == 'vertical':
        if side == 'right':
            coords = [locpos.x0 + locpos.width*(1+pad) + shift[0],
                      locpos.y0 + (1 -height)*locpos.height/2 + shift[1],
                      locpos.width*width, height*locpos.height]
        elif side == 'left':
            coords = [locpos.x0 - locpos.width*width - locpos.width*pad - shift[0],
                      locpos.y0 + (1 -height)*locpos.height/2 + shift[1],
                      locpos.width*width, height*locpos.height]
        else:
            raise(Exception("Side not compatible with orientation"))
        
        cbax = fig.add_axes(coords)
        cbar = fig.colorbar(mappable, cax = cbax, orientation = orientation, *args, **kwargs)
        cbar.ax.yaxis.set_ticks_position(side)
        cbar.ax.tick_params(direction = 'out')
    elif orientation == 'horizontal':
        if (width == 0.01) and (height == 1):
            width = 1
            height = 0.02
            
        if side == 'top':
            coords = [locpos.x0 + locpos.width*(1 - width)/2 + shift[0], locpos.y0 + (pad+1)*locpos.height + shift[1],
                      locpos.width*width, height*locpos.height]
        elif side == 'bottom':
            coords = [locpos.x0 + locpos.width*(1 - width)/2 + shift[0], locpos.y0 - (height+pad)*locpos.height - shift[1],
                      locpos.width*width, height*locpos.height]
        else:
            raise(Exception("Side not compatible with orientation"))
            
        cbax = fig.add_axes(coords)
        cbar = fig.colorbar(mappable, cax = cbax, orientation = orientation, *args, **kwargs)
        cbar.ax.xaxis.set_ticks_position(side)
        
    return cbar

def circular_color_legend( mappable = None, xy = (0,0), size = 0.5,
                           resolution = 500, circular_cmap = cm.hsv,
                           **kwargs):
    """
    Generates a circular colorbar, for use with colormaps highlighting vector
    directions, e.g. spin textures.
    
    Parameters
    ----------
    mappable:
        The mappable cooresponding to the plot which needs the circular legend.
    xy: float tuple, optional
        The shift of the legend in axes coordinates. The alignment is specified
        with the **kwargs. Default is 'left' and 'bottom'.
    size: float, optional
        The size of the legend, in relative coordinates. Default is 0.5.
    resolution: int, optional
        The amount of pixels defining the grid over which the legend is plotted.
        Default is 500.
    circular_cmap: optional
        The colormap used specify the colors for the in-plane direction.
        Default is cm.hsv
    
    Returns
    -------
    axis object
        The axis of the circular legend.
        
    Other parameters
    ----------------
    **kwargs
    rasterized: bool
        Set to True if image rasterization is required. Default is False.
    horizontalignment: str
        Can be 'left', 'center', 'right'. Default is 'left'.
    verticalalignment: str
        Can be 'bottom', 'center', 'top'. Default is 'bottom'.
    white_center: bool
        If True, the color representing the out-of-plane vector is white,
        black otherwise. Default is False.
    linewidth: float
        The linewidth of the circle in relative coordinates. Default is 0.01.
    edgecolor: RGBA list or str
        The color of the edge. Default is black.
    """
    
    horizontalalignment = kwargs.get('horizontalalignment', 'left')
    verticalalignment = kwargs.get('verticalalignment', 'left')
    white_center = kwargs.get('white_center', False)
    edgecolor = kwargs.get('edgecolor', 'k')
    linewidth = kwargs.get('linewidth', .01)
    rasterized = kwargs.get('rasterized', False)
    
    plt_axis = mappable.axes
    fig = plt_axis.figure
    figure_size= fig.get_size_inches()
    aspect = figure_size[0] / figure_size[1] 
    
    locpos = plt_axis.get_position()
    #smaller_dim = min( locpos.height, locpos.width )
    #circ_size = size * smaller_dim
    circ_size_x, circ_size_y = size * locpos.width, size * locpos.height * aspect
    
    multiplier = 0
    if horizontalalignment == 'left':
        pass
    elif horizontalalignment == 'center':
        multiplier = 0.5
    elif horizontalalignment == 'right':
        multiplier = 1
        
    #circle_x = locpos.x0 + xy[0] * (locpos.width - locpos.x0) - circ_size * multiplier
    circle_x = locpos.x0 + xy[0] * locpos.width- circ_size_x * multiplier
    
    multiplier = 0
    if verticalalignment == 'bottom':
        pass
    elif verticalalignment == 'center':
        multiplier = 0.5
    elif verticalalignment == 'top':
        multiplier = 1
    
    #circle_y = locpos.y0 + xy[1] * (locpos.height - locpos.y0) - circ_size * multiplier
    circle_y = locpos.y0 + xy[1] * locpos.height - circ_size_y * multiplier
    
    #leg_ax = fig.add_axes([circle_x, circle_y, circ_size, circ_size])
    leg_ax = fig.add_axes([circle_x, circle_y, circ_size_x, circ_size_y])
    
    
    X, Y = np.meshgrid( np.linspace(-1, 1, resolution),
                        np.linspace(-1, 1, resolution) )
    
    norm = np.linalg.norm( np.stack( (X, Y),axis = 2 ), axis = 2 )
    
    Z = np.sqrt(1 -np.square(X) - np.square(Y) )
    
    
    tangent = np.fliplr( np.arctan2(Y,X) )
    tangent -= tangent.min()
    tangent /= tangent.max()
    
    theta = np.arctan2( norm, Z )
    theta -= np.nanmin(theta)
    theta /= np.nanmax(theta)
    
    theta[np.isnan(theta)] = 0
    
    #gauss_grad =  np.exp(- np.square(norm) / (2 * central_color_rad**2) )
        
    
    the_map = circular_cmap(tangent.ravel())
    
    if white_center:
        the_map[:,:3] = (1-theta).ravel()[:, None] + the_map[:,:3] * theta.ravel()[:, None]#gauss_grad.ravel()[:, None] + the_map[:,:3] * (1-gauss_grad).ravel()[:, None]
    else:
        the_map[:,:3] = the_map[:,:3] * theta.ravel()[:, None]# the_map[:,:3] * (1-gauss_grad).ravel()[:, None]
    
    
    the_map = the_map.reshape( X.shape + (4, ) )
    
  
    if linewidth != 0:
        if  type(edgecolor) is str:
            color = mplcol.to_rgba(edgecolor)
            color = [x for x in color]
        the_map[ np.square(X) + np.square(Y) >=(1-linewidth)**2, :] = color
        
    the_map[ np.square(X) + np.square(Y) >=  1, 3] = 0

    circ_legend = leg_ax.imshow(the_map, aspect = 'auto', interpolation = 'spline36',
                                extent = (-1,1,-1,1), rasterized = rasterized)
    leg_ax.set_xlim(-1.05,1.05)
    leg_ax.set_ylim(-1.05,1.05)
    leg_ax.axis('off')
    
    return circ_legend, theta
    
def crosscorrelate2d(mat1, mat2, mode = 'same'):
    """
    Calculates the normalised cross-correlation (or autocorrelation) of two matrices.
    The method uses the Wiener-Kinchin theorem (fft, mulitply and then inverse fft).
    The function already corrects the data by mean and subtraction, so there is
    no need to provide corrected matrices. If the matrices shapes M and N are
    larger than 2, the cross-correlation is computed along the last two dimensions.
    M.shape[:-2] must be equal to N.shape[:-2]. 
    
    Parameters
    ----------
    mat1: np.ndarray
        The first matrix.
    mat2: np.ndarray
        The second matrix.
    mode: str
        'full' : computes the full correlation
        'same' : returns a matrix with the shape of the first input matrix
        'valid' : returns the correlation where both matrices are not padded.
                  One of the two matrices needs to be larger than the other in
                  the both last two dimensions.
    
    Returns
    -------
    correlated: np.ndarray
        The correlation matrix.
    """
    
    mat1 = (mat1 - mat1.mean()) / mat1.std()
    mat2 = (mat2 - mat2.mean()) / mat2.std()
    
    mat1shape = np.array(mat1.shape[-2:])
    mat2shape = np.array(mat2.shape[-2:])
    
    finalshape = mat1shape + mat2shape - 1
    
    def padding_generator( finalshape, matshape ):
        other_axes = ( (0,0), ) * len(matshape[:-2])
        last_two = ( ( floor( ( finalshape[0] - matshape[-2] )/2), ceil((finalshape[0] - matshape[-2] ) / 2) ),
                     ( floor( ( finalshape[1] - matshape[-1] )/2), ceil((finalshape[1] - matshape[-1] ) / 2) ),
                   )
        return other_axes + last_two

    pad_mat1 = padding_generator(finalshape, np.array(mat1.shape))

    pad_mat2 = padding_generator(finalshape, np.array(mat2.shape))
    
    image_center = [0,0]
    if finalshape[0]%2 != 0:
        image_center[0] = floor(finalshape[0]/2) + 1
    else:
        image_center[0] = int(finalshape[0]/2)
        
    if finalshape[1]%2 != 0:
        image_center[1] = floor(finalshape[1]/2) + 1
    else:
        image_center[1] = int(finalshape[1]/2)

    mat1 = np.pad(mat1, pad_mat1, mode = 'constant')
    mat2 = np.pad(mat2, pad_mat2, mode = 'constant')            
        
    correlated = np.roll( ifft2( fft2(mat1) * fft2(mat2).conj() ).real,
                         (image_center[0]-1, image_center[1]-1), axis = (-2,-1) )
    if mode == 'same':
        correlated = correlated[... ,
                                image_center[0] - floor(mat1shape[0]/2) : image_center[0] + ceil(mat1shape[0]/2),
                                image_center[1] - floor(mat1shape[1]/2) : image_center[1] + ceil(mat1shape[1]/2)]
    if mode == 'valid':
        if np.all(mat1shape >= mat2shape):
            outputshape = mat1shape - mat2shape + 1
        elif np.all(mat1shape >= mat2shape):
            outputshape = mat2shape - mat1shape + 1
        else:
            raise ValueError("Either mat1 or mat2 must be larger than the other in both dimensions")
            
        correlated = correlated[...,
                                image_center[0] - floor(outputshape[0]/2) - 1 : image_center[0] + ceil(outputshape[0]/2) - 1,
                                image_center[1] - floor(outputshape[1]/2) - 1: image_center[1] + ceil(outputshape[1]/2) - 1]

    correlated /= max(mat1shape[0],mat2shape[0]) * max(mat1shape[1],mat2shape[1])
    
    return correlated     

def parula_cmap():
    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
                 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
                 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
                  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
                 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
                  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
                 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
                  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
                 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
                  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
                 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
                  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
                 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
                  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
                  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
                 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
                  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
                 [0.0589714286, 0.6837571429, 0.7253857143], 
                 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
                 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
                  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
                 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
                  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
                 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
                  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
                 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
                  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
                 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
                 [0.7184095238, 0.7411333333, 0.3904761905], 
                 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
                  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
                 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
                 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
                  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
                 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
                  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
                 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
                 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
                 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
                  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
                 [0.9763, 0.9831, 0.0538]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    return parula_map

if __name__ == '__main__':
    X,Y = np.mgrid[0:10:100j,0:10:100j]
    Z = X* Y
    
    recwid = 0.3
    rechei = 0.1
    
    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(Z, extent = (0,10,0,10), origin = 'lower')
    add_size_label(xax = X[0,:], yax = Y[:,0], matshape = Z.shape, rect_coords=(0.5 * X.max(), 0.5 * Y.max()),
                   length = 2, bg_extra_size = 0.1)