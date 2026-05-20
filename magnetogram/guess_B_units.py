

##############################################################
# Import packages, variables, functions...

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datetime as dt
from astropy.io import fits
import matplotlib.patches as patches
from scipy.odr import Model, RealData, ODR
from scipy.interpolate import interp1d
import matplotlib.lines as mlines
import sunpy.map
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
import astropy.units as u

import sys
import os
sys.path.append(os.path.abspath('..'))
from utils.data_path import path_data_soho 
from utils.SOHO_aux_functions import *
from utils.calibration_parameters__output import *
from utils.spectroheliogram_functions import *
from utils.solar_rotation_variables import *
from utils.aux_functions import *



##############################################################
# 

# We compare with the paper "On the outflow in an equatorial coronal hole", L. D. Xia et al. 2003

filename_mag = '991105M1'

header_mag = fits.getheader('data/'+filename_mag)
data_mag = fits.getdata('data/'+filename_mag)
data_mag = data_mag * header_mag['BSCALE'] + header_mag['BZERO'] #header: BSCALE  =       1.0000000000E0  /  REAL = TAPE*BSCALE + BZERO
data_mag_inverse = data_mag[::-1]




print('###')
print('Magnetogram:', header_mag['UTDATE'], "---", header_mag['UTSTART'], '[UT]')



v_max = 50.
fig, ax = plt.subplots(figsize=(9, 8.25))
ax.imshow(data_mag_inverse, vmin=-v_max, vmax=v_max, cmap='Greys_r')
#ax.axis('equal')
ax.set_xlabel('X direction (pixels)', fontsize=17)
ax.set_ylabel('Y direction (pixels)', fontsize=17)
plt.show(block=False)


# Crop in the same window as the Figure 2 of the paper
rows_range = 672, 932
cols_range = 475, 693 
data_mag_inverse_crop = data_mag_inverse[rows_range[0]:rows_range[1]+1, cols_range[0]:cols_range[1]+1]

v_max = 100. # v_max=100 is the value used in the map of the paper (Fig. 2) and the color coincides, so we can say that the values of the image here are magnetic field in Gauss, and that white is positive polarity and black is negative polarity. 
fig, ax = plt.subplots(figsize=(9, 8))
img = ax.imshow(data_mag_inverse_crop, vmin=-v_max, vmax=v_max, cmap='Greys_r')
cax = fig.add_axes([0.84, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, ax=ax, cax=cax, pad=0.01)
cbar.set_label(f'Magnetic field (Gauss)', fontsize=16)
#ax.axis('equal')
ax.set_xlabel('X direction (pixels)', fontsize=17)
ax.set_ylabel('Y direction (pixels)', fontsize=17)
plt.show(block=False)




