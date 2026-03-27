
# I DO NOT BIN THE MAP, but the name bin2 is to be together with the other bin2_ files.

"""
#  Inputs
eit_wavelength = 195 #171, 195, 284, or 304 [Angstrom]
eit_time = 'late' #'early' or 'late' (early: around 1 or 4 am; late: around 6 or 7 am)
"""

######################################################
# import packages

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
from utils.general_variables import *
from utils.NeVIII_rest_wavelength import *
from scale_hrts import *

######################################################
# Import EIT data

# Select the name of the EIT file according to the above inputs
if eit_time=='early':
    if eit_wavelength==171: filename_eit = 'SOHO_EIT_171_19991107T010032_L1.fits'
    elif eit_wavelength==195: filename_eit = 'SOHO_EIT_195_19991107T042103_L1.fits'
    elif eit_wavelength==284: filename_eit = 'SOHO_EIT_284_19991107T011231_L1.fits'
    elif eit_wavelength==304: filename_eit = 'SOHO_EIT_304_19991107T013601_L1.fits'

elif eit_time=='late':
    if eit_wavelength==171: filename_eit = 'SOHO_EIT_171_19991107T070017_L1.fits'
    elif eit_wavelength==195: filename_eit = 'SOHO_EIT_195_19991107T063706_L1.fits'
    elif eit_wavelength==284: filename_eit = 'SOHO_EIT_284_19991107T070704_L1.fits'
    elif eit_wavelength==304: filename_eit = 'SOHO_EIT_304_19991107T073030_L1.fits'

# Path of EIT file
filepath_eit = path_data_soho + 'eit/' + filename_eit

# Extract data and header
data_eit = fits.getdata(filepath_eit)[::-1]
header_eit = fits.getheader(filepath_eit)

######################################################

from utils.solar_rotation_variables import *
closest_index = closest_index_EIT_SUMER_dic[filename_eit]
closest_time_sumer = closest_time_SUMER_to_EIT_dic[filename_eit]
time_eit = time_EIT_dic[filename_eit]
hour_eit = hour_EIT_dic[filename_eit]
HPlon_rotcomp = HPlon_rotcomp_dic[filename_eit]
HPlon
HPlat
HPlat_croplat = HPlat[slit_top_px:slit_bottom_px+1]

print("Time of EIT image:.......................", time_eit)
print("Closest time of the SUMER raster:........", closest_time_sumer)
print("Index of that SUMER file in the list:....", closest_index)


######################################################
# Crop EIT data

# Find row index in EIT corresponding to these extremes
y_px_crop_top = int(np.round(Y__HP_to_pixel(y_HP=HPlat[slit_top_px], header_eit=header_eit)))
y_px_crop_bottom = int(np.round(Y__HP_to_pixel(y_HP=HPlat[slit_bottom_px], header_eit=header_eit)))
x_px_crop_left = int(np.round(X__HP_to_pixel(x_HP=HPlon_rotcomp[0], header_eit=header_eit)))
x_px_crop_right = int(np.round(X__HP_to_pixel(x_HP=HPlon_rotcomp[-1], header_eit=header_eit)))

# Crop EIT array
data_eit_crop = data_eit[y_px_crop_top:y_px_crop_bottom+1, x_px_crop_left:x_px_crop_right+1]

# Corrected alignment
dx_px = 0
dy_px = -6
data_eit_crop_corrected = data_eit[y_px_crop_top+dy_px : y_px_crop_bottom+dy_px, x_px_crop_left+dx_px : x_px_crop_right+dx_px]


#####################################################
# Extents

# Extents in pixels
## Image
extent_eit_px_uncorrected_image = [-0.5, data_eit_crop.shape[1]-1+0.5, data_eit_crop.shape[0]-1+0.5, -0.5]
extent_eit_px_image = [-0.5, data_eit_crop_corrected.shape[1]-1+0.5, data_eit_crop_corrected.shape[0]-1+0.5, -0.5]
## Contours
extent_eit_px_uncorrected_contours = [0., data_eit_crop.shape[1]-1, data_eit_crop.shape[0]-1, 0.]
extent_eit_px_contours = [0., data_eit_crop_corrected.shape[1]-1, data_eit_crop_corrected.shape[0]-1, 0.]


#####################################################
# 

print('--------------------------------------')
print('header_eit')
print('data_eit')
print('data_eit_crop_corrected')
print('')
print('extent_eit_px_uncorrected_contours')
print('extent_eit_px_contours')
print('')
print('extent_eit_sumer_arcsec_image')
print('extent_eit_sumer_arcsec_contours')
print('')
print('closest_index')
print('closest_time_sumer')
print('time_eit')
print('hour_eit')
print('HPlon_rotcomp')
print('HPlon')
print('HPlat')
print('HPlat_croplat')
print('--------------------------------------')

#####################################################
# 

