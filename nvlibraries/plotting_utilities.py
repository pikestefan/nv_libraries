# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:00:37 2019

@author: Lucio Stefan, lucio.stefan@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import golden
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

def cm2inch(cm):
    return cm*0.393701

DEF_WIDTH = cm2inch(21)

DEF_ASPECT = golden

DEF_TICKLABELRATIO = 2

DEF_LABELPAD = 1

DEF_LABELRATIO = 2.5

def rgb_norm(red,green,blue):
    return red/255,green/255,blue/255

def _whiteblueblacklist():
    custom_map_res = 64;

    middle =0;
    
    cmap_first = np.column_stack((
        np.linspace(1,middle,custom_map_res),np.linspace(1,middle,custom_map_res)**(0.4),np.tile(1,custom_map_res) ))
    cmap_second = np.column_stack((
        np.tile(0,custom_map_res),np.tile(0,custom_map_res),np.linspace(1,0,custom_map_res)**(1.2) ));
    
    cmap = np.flipud(
        np.vstack((cmap_first, cmap_second)) )
    
    tup_cmap = [(cmap[ind,0], cmap[ind,1], cmap[ind,2]) for ind in range(cmap.shape[0])]
    return tup_cmap
whiteblueblack = LinearSegmentedColormap.from_list( 'whiteblueblack', _whiteblueblacklist(), N = 1028 )

def set_label_sizes(fig_width = DEF_WIDTH, ax  = None,
                    ticklabel_aspect = DEF_TICKLABELRATIO,
                    label_aspect = DEF_LABELRATIO,
                    label_padding = DEF_LABELPAD):
    
    ax.set_xlabel(ax.get_xlabel(), fontsize = label_aspect*fig_width, labelpad = label_padding*fig_width)
    ax.set_ylabel(ax.get_ylabel(), fontsize = label_aspect*fig_width, labelpad = label_padding*fig_width)
    
    ax.xaxis.set_tick_params(labelsize = ticklabel_aspect*fig_width)
    ax.yaxis.set_tick_params(labelsize = ticklabel_aspect*fig_width)

def plot2d_with_size_label(Zdata = None, Xdata = None, Ydata = None,
                           rect_coords = (0,0), rect_thick = 0.02,
                           length = 1, unit = "nm", label_precision = ":.0f",
                           size_label_color = 'k', pad = 0.02, fontsize = 12,
                           bg_color = 'w', bg_extra_size = 0.01, with_bg = True,
                           which_plot_method = 'imshow',
                           axis_handle = None, fig_handle = None,
                           *args, **kwargs):
    
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
        plot_handle = axis_handle.imshow(Zdata, *args, **kwargs, origin = 'lower',
                                         extent = (Xdata.min(),Xdata.max(),Ydata.min(),Ydata.max()),
                                         aspect = 'auto')
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
                                   ha = 'center', color = size_label_color)
    
    
    #This code section gets the width and height of the text box
    renderer = fig_handle.canvas.get_renderer()
    bbox = text_object.get_window_extent(renderer = renderer) #bbox in display coordinates
    display_coord2data_coord_transform = axis_handle.transData.inverted() #get the coord transform from display to data
    bbox = bbox.transformed( display_coord2data_coord_transform ) #convert bbox into data coordinates
    
    text_w, text_h = bbox.width, abs(bbox.height)
    
    #Length bar
    size_indicator = Rectangle(rect_coords, width=length, height = rect_hei,
                               edgecolor = size_label_color,
                               facecolor = size_label_color)
    if with_bg:
        #Calculate the size of the background box
        tot_width = length if (length > text_w) else text_w
        
        #Calculate the bottom left corner position and the centre of the box

        tot_height = (bbox.y0 -rect_coords[1]) + text_h
        
        tot_y0 = rect_coords[1]
        tot_x0 = rect_coords[0] if (rect_coords[0] < bbox.x0) else bbox.x0
        
        bbox_centre_x0 = tot_x0 + tot_width/2
        bbox_centre_y0 = tot_y0 + tot_height/2
        
        #Increase the size of the bbox
        tot_width = (1 + bg_extra_size) * tot_width
        tot_height = (1 + bg_extra_size) * tot_height #tot_height + bg_extra_size * tot_width
        
        #These are the coordinates of the enlarged background box
        tot_x0 = bbox_centre_x0 - tot_width/2
        tot_y0 = bbox_centre_y0 - tot_height/2
        
        bg = Rectangle((tot_x0, tot_y0), width=tot_width, height = tot_height,
                        edgecolor = bg_color,
                        facecolor = bg_color)
        
        axis_handle.add_patch(bg)
    axis_handle.add_patch(size_indicator)
    
    
    return plot_handle
    
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


if __name__ == '__main__':
    vec = np.linspace(0,100,100)
    x=  np.linspace(0,10,100)
    y = np.linspace(10,20,100)
    X, Y = np.meshgrid(x,y)
    mat = np.repeat(vec[:,np.newaxis], repeats = 100, axis = 1)
    fig,ax = plt.subplots()
    plot2d_with_size_label(mat,X,Y,rect_coords = (3,10), length = 5, which_plot_method = 'imshow',
                           size_label_color = 'k',bg_extra_size=0.05, pad = 0.1, fontsize = 40,
                           rasterized = True,
                           cmap = whiteblueblack)
    gni = "{"+":.2f"+"}"
    ue=gni.format(10)