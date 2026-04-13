######################################################
# Inputs

# Main inputs
N_bins = 300

# Secondary inputs
line_label = 'NeVIII'
eit_wavelength = 195 #171, 195, 284, or 304 [Angstrom]
eit_time = 'late' #'early' or 'late' (early: around 1 or 4 am; late: around 6 or 7 am)
range_percentage_eit = [4., 10.]
threshold_value_type_eit = 'max'

######################################################
######################################################
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

########################################################
########################################################
########################################################
# Import SUMER intensity map

# Load the intensity map and uncertainties
intensitymap_loaded_dic = np.load('../outputs/intensity_map_'+line_label+'_interpolated.npz')
intensity_map = intensitymap_loaded_dic['intensity_map'] #2D-array
intensity_map_unc = intensitymap_loaded_dic['intensity_map_unc'] #2D-array
intensity_map_croplat = intensitymap_loaded_dic['intensity_map_croplat'] #2D-array
intensity_map_unc_croplat = intensitymap_loaded_dic['intensity_map_unc_croplat'] #2D-array
line_center_label = intensitymap_loaded_dic['line_center_label'] 
vmin_sumer, vmax_sumer = intensitymap_loaded_dic['vmin_vmax'] 

########################################################
########################################################
########################################################
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
# Crop EIT data

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

######################################################
# Extents

# Extents in pixels
## Image
extent_eit_px_uncorrected_image = [-0.5, data_eit_crop.shape[1]-1+0.5, data_eit_crop.shape[0]-1+0.5, -0.5]
extent_eit_px_image = [-0.5, data_eit_crop_corrected.shape[1]-1+0.5, data_eit_crop_corrected.shape[0]-1+0.5, -0.5]
extent_sumer_px_image = [-0.5, intensity_map_croplat.shape[1]-1+0.5, intensity_map_croplat.shape[0]-1+0.5, -0.5]
## Contours
extent_eit_px_uncorrected_contours = [0., data_eit_crop.shape[1]-1, data_eit_crop.shape[0]-1, 0.]
extent_eit_px_contours = [0., data_eit_crop_corrected.shape[1]-1, data_eit_crop_corrected.shape[0]-1, 0.]
extent_sumer_px_contours = [0., intensity_map_croplat.shape[1]-1, intensity_map_croplat.shape[0]-1, 0.]

vmin_eit, vmax_eit = 4e1, 3e3

######################################################
# Define intensity bin
lower_bound_eit, upper_bound_eit = get_bounds(intensitymap_croplat=intensity_map_croplat, range_percentage=range_percentage_eit, threshold_value_type=threshold_value_type_eit)
print('lower_bound_eit, upper_bound_eit =', lower_bound_eit, ',', upper_bound_eit)


########################################################
########################################################
########################################################
# Figures

########################################################
# Plot image and histogram of SUMER

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
label_size = 18
ax.imshow(intensity_map_croplat, norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer), cmap='Greys_r', aspect='auto', extent=extent_sumer_px_image)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
ax.set_ylabel('Latitude dimension (pixels)', fontsize=17)
plt.show(block=False)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
ax.hist(intensity_map.ravel(), bins=N_bins, color='blue')
ax.set_title('Histogram of the SUMER intensity map of Ne VIII 770 \u212B', fontsize=18)
ax.set_xlabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
ax.set_ylabel("Count")
plt.tight_layout()
plt.show(block=False)


########################################################
# Plot image and histogram of SUMER

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
label_size = 18
ax.imshow(data_eit_crop_corrected, norm=LogNorm(vmin=vmin_eit, vmax=vmax_eit), cmap='Greys_r', aspect='auto', extent=extent_eit_px_image)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
ax.set_ylabel('Latitude dimension (pixels)', fontsize=17)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
# EIT subplot (top) - contours from EIT data
contour_lower_eit = ax.contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_px_contours)
contour_upper_eit = ax.contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='blue', linewidths=2, extent=extent_eit_px_contours)
plt.show(block=False)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
ax.hist(data_eit_crop_corrected.ravel(), bins=N_bins, color='blue')
ax.axvline(x=lower_bound_eit, color='black', linestyle='--', linewidth=1.5, label='lower and upper bound')
ax.axvline(x=upper_bound_eit, color='black', linestyle='--', linewidth=1.5)
ax.set_title(f'Histogram of the EIT intensity map of {eit_wavelength}'' \u212B', fontsize=18)
ax.set_xlabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
ax.set_ylabel("Count")
ax.legend(fontsize=12)
plt.tight_layout()
plt.show(block=False)


########################################################
########################################################
########################################################
# 


