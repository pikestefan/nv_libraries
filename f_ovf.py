# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:33:54 2020
@author: ant_t
"""
import numpy as np
import pandas as pd
import re

def get_stripe_Mz(Nx= None,Ny= None,Nz= None,eff_z=1,periodic_offset_x=0, 
                 periodic_offset_y=0,angle=0,interlayer_coupling=1,width_up= None,
                 width_down= None):
    M_data=np.zeros([3,Nx,Ny,Nz])
    period=width_up+width_down;
    
    xgrid,ygrid=np.meshgrid(np.arange(0,Nx),np.arange(0,Ny))
    
    sign_M=interlayer_coupling;
    for l in np.arange(0,eff_z): 
        M_data[2,:,:,l]=sign_M*(-np.ones([Nx,Ny])
        +2*(np.mod(xgrid-np.mod(l,2)*periodic_offset_x,period)<width_up))
        sign_M=sign_M*interlayer_coupling;
    
    return(M_data)
    
def get_circle_Mz(Nx= None,Ny= None,Nz= None,eff_z=1,offset_x=0, 
                 offset_y=0,interlayer_coupling=1,radius=0):
    M_data=np.zeros([3,Nx,Ny,Nz])
    
    xgrid,ygrid=np.meshgrid(np.arange(-Nx/2,Nx/2),np.arange(-Ny/2,Ny/2))
    rgrid=((xgrid+offset_x)**2+(ygrid+offset_x)**2)**0.5
    
    sign_M=interlayer_coupling;
    for l in np.arange(0,eff_z): 
        M_data[2,:,:,l]=sign_M*(-np.ones([Nx,Ny]) +2*(rgrid<radius))
        sign_M=sign_M*interlayer_coupling;
    
    return(M_data)
    
def write_ovf(filepath = None,lenx = None, leny = None, lenz=None, Nx = None, 
              Ny=None, Nz=None, dataset=None):
  
    if ((np.size(dataset)==(Nx*Ny*Nz*3)) and ((len(dataset.shape)==4) or (len(dataset.shape)==2))):
        #change to (Nx*Ny*Nz,3)
        if len(dataset.shape)==4:
            if dataset.shape[-1]!=3:
                dataset=np.transpose(dataset,(3,2,1,0))  
            dataset=np.reshape(dataset,(Nx*Ny*Nz,3))
        elif len(dataset.shape)==2:
            if dataset.shape[-1]!=3:
                dataset=np.transpose(dataset,(1,0))  
                
        #note that python writes \n as \r\n in windows
        #mumax throws an error with \r
        #newline=\n forces python to write \n as \n in windows
        with open(filepath,"w", newline='\n') as f:
            f.write(u"# OOMMF: rectangular mesh v1.0\n")
            f.write(u"# Segment count: 1 \n")
            f.write(u"# Begin: Segment  \n")
            f.write(u"# Begin: Header  \n")
            f.write(u"# Desc: Time (s) :0\n")
            f.write(u"# Title: m  \n")
            f.write(u"# meshtype: rectangular  \n") 
            f.write(u"# meshunit: m  ") 
            f.write(u"\n# xbase: "+np.format_float_scientific(lenx,trim='0')+"  ")
            f.write(u"\n# ybase: "+np.format_float_scientific(leny,trim='0')+"  ")
            f.write(u"\n# zbase: "+np.format_float_scientific(lenz,trim='0')+"  ")
            f.write(u"\n# xstepsize: "+np.format_float_scientific(lenx/Nx,trim='0')+"  ")
            f.write(u"\n# ystepsize: "+np.format_float_scientific(leny/Ny,trim='0')+"  ")
            f.write(u"\n# zstepsize: "+np.format_float_scientific(lenz/Nz,trim='0')+"  ")  
            f.write(u"\n# xmin: 0  ")
            f.write(u"\n# ymin: 0  ")
            f.write(u"\n# zmin: 0  ")
            f.write(u"\n# xmax: "+np.format_float_scientific(lenx,trim='0')+"  ") 
            f.write(u"\n# ymax: "+np.format_float_scientific(leny,trim='0')+"  ")  
            f.write(u"\n# zmax: "+np.format_float_scientific(lenz,trim='0')+"  ")  
            f.write(u"\n# xnodes: %i  "%Nx)
            f.write(u"\n# ynodes: %i  "%Ny)  
            f.write(u"\n# znodes: %i  "%Nz)  
            f.write(u"\n# ValueRangeMinMag: 1e-08  ")
            f.write(u"\n# ValueRangeMaxMag: 1  ")
            f.write(u"\n# valueunit:   ")
            f.write(u"\n# valuemultiplier: 1  ")
            f.write(u"\n# End: Header  ")
            f.write(u"\n# Begin: Data Text \n") 
            np.savetxt(f, dataset,fmt='%.2f')
            f.write(u"# End: Data Text ")
            f.write(u"\n# End: Segment\n")
                    
        display('File sucessfully saved')
    else:
        display('Size of dataset does not match simulation paramters')

    # replacement strings
    '''
    WINDOWS_LINE_ENDING = b'\r\n'
    UNIX_LINE_ENDING = b'\n'
    
    with open(filepath, 'rb') as open_file:
        content = open_file.read()
    
    content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)
    
    with open(filepath, 'wb') as open_file:
        open_file.write(content)
    '''    
def read_ovf(filepath = None, data_sep = ' ', comment = '#', engine = 'c'):
    """
    Function to read .ovf files.
    
    Parameters
    ----------
    filepath: string
        The path of the .ovf file. Default is none
    datasep: string
        The separator of the numbers. Default is ' '
    comment: string
        The symbol defining a commented line. Default is '#'
    engine: string, 'c' or 'python'
        The engine used by pandas to read the data. Default is 'c'
        
    Returns
    -------
    data: np.ndarray
        The 4d matrix containing the coordinates. It is arranged as (zcoord, xcoord, ycoord, component).
    header_dictionary: dict
        A dictionary containing the metadata in the .ovf header
    """
    
    with open(filepath) as file:
        keepreading = True
        header_dictionary = {}
        while keepreading:
            line = file.readline()
            #Keep reading as long as the line starts with the comment symbol
            if line.startswith(comment):
                line = line[1:].strip() #Remove trailing white spaces and the comment symbol
                split = line.split(':', maxsplit = 1) #Split on the first colon
                #Use the first list element as a dictionary key
                #and use the second element of the list as value
                key = split[0].strip()
                value = split[1].strip()
                try:
                    value = float(value) #Attempt to convert the string into float, if possible
                    #Lazy solution to cast to int if the float is an integer number.
                    #helps to feed the number straight into linspace without recasting by hand
                    if value.is_integer():
                        value = int(value)
                except:
                    pass
                header_dictionary[ key ]  = value
            else:
                keepreading = False
     
    #Load the data as a pandas dataframe and convert it into numpy array
    data = pd.read_csv(filepath,
                       sep = data_sep, comment = comment, usecols = [0, 1, 2],
                       skip_blank_lines = True, header = None, engine = engine).values
    
    
    #Reshape the columns into 3D matrices with shape (znodes, xnodes, ynodes, 3)
    shape = (header_dictionary['xnodes'], header_dictionary['ynodes'], header_dictionary['znodes'], 3)
    data = data.reshape(shape, order = 'F').transpose( 2, 1, 0, 3 )
    
    return data, header_dictionary

'''
class ovf_data:
    def __init__(self):
        self._Begin = None
        
    def read_ovf(self,ovf_file):
        
        print('loading file...')
        with open(ovf_file,'r') as ovf_file_handle:
            self._Begin=None
            
            #Retrieve header information
            while self._Begin != 'Data Text': #last line of header info
                line = ovf_file_handle.readline()
                self._assign_headers(line)                
            
            #create data array
            self.data=np.zeros((self.Nx*self.Ny*self.Nz,3))
            
            #Retrieve OVF data 
            for line_ind, line in zip(np.arange(0,self.Nx*self.Ny*self.Nz),ovf_file_handle):
                self.data[line_ind,:]=np.fromstring(line,dtype=float,count=3,sep=' ')
                
        self.data=self.data.reshape((self.Nx,self.Ny,self.Nz,3),order='F').transpose(1,0,2,3)    
        
        #Retrieve OVF data via pandas (slower?) 

        self.data=pd.read_csv(ovf_file,sep='\s+',skiprows=data_line, \
        skipfooter=2, index_col=None,header=None,engine='python').to_numpy()\
        .reshape((self.Nx,self.Ny,self.Nz,3),order='F').transpose(1,0,2,3)

        print('done!')               
           
    def _assign_headers(self, line):
        #regular expression pattern: # xyz: 123
        header_exp=re.compile(r'#\s(\D*):\s(.*)')
        info=header_exp.findall(line)
        
        if np.size(info) == 2:
            if info[0][0] == 'meshunit':
                self.xyzunit =info[0][1]
            elif info[0][0] ==  'xstepsize':
                self.xstep = float(info[0][1])
            elif info[0][0] ==  'ystepsize':
                self.ystep= float(info[0][1])
            elif info[0][0] ==  'zstepsize':
                self.zstep= float(info[0][1])
            elif info[0][0] ==  'xnodes':
                self.Nx= int(info[0][1])
            elif info[0][0] ==  'ynodes':
                self.Ny= int(info[0][1])
            elif info[0][0] ==  'znodes':
                self.Nz= int(info[0][1])
            elif info[0][0] ==  'valueunit':
                self.Dunit= info[0][1]
            elif info[0][0] ==  'ValueRangeMaxMag':
                self.Dmax= float(info[0][1])
            elif info[0][0] ==  'ValueRangeMinMag':
                self.Dmin= float(info[0][1])
            elif info[0][0] ==  'Begin':
                self._Begin=info[0][1]
 '''       
        