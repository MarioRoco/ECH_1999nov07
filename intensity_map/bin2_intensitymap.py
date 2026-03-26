# INPUTS

# Binning
bin_lat = 4
bin_lon = 1

save_intensity_map = 'yes'

show_spectral_image_binned = 'yes'
show_wavelength_range = 'yes'
show_intensitymap_binned = 'yes'

line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line':

filename_eit = 'SOHO_EIT_195_19991107T042103_L1.fits' #for the binning of the coordinates (where solar rotation has been corrected)

############################################################


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
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import matplotlib.lines as mlines

import sys
import os
sys.path.append(os.path.abspath('..'))
#from utils.data_path import path_data_soho 
from utils.SOHO_aux_functions import *
from utils.calibration_parameters__output import *
from utils.spectroheliogram_functions import *
from utils.solar_rotation_variables import *
from utils.aux_functions import *
from utils.general_variables import *
from utils.NeVIII_rest_wavelength import *
from utils.auxfuncs_binning_and_dopplermap import *
from scale_hrts import *

############################################################
# Bin data

data_binned_loaded = np.load(f'../data/data_modified/spectral_image_list_intepolated_binned_lon{bin_lon}_lat{bin_lat}.npz', allow_pickle=True)
spectral_image_interpolated_croplat_binned_list = data_binned_loaded['spectral_image_interpolated_croplat_binned_list']
spectral_image_unc_interpolated_croplat_binned_list = data_binned_loaded['spectral_image_unc_interpolated_croplat_binned_list']
pixelscale_list_croplat_binned = data_binned_loaded['pixelscale_list_croplat_binned']
pixelscale_unc_list_croplat_binned = data_binned_loaded['pixelscale_unc_list_croplat_binned']
pixelscale_intercept_list_croplat_binned = data_binned_loaded['pixelscale_intercept_list_croplat_binned']
pixelscale_intercept_unc_list_croplat_binned = data_binned_loaded['pixelscale_intercept_unc_list_croplat_binned']
x_HPlon_rotcomp_binned = data_binned_loaded['x_HPlon_rotcomp_binned']
y_HPlat_crop_binned = data_binned_loaded['y_HPlat_crop_binned']
lam_sumer = data_binned_loaded['lam_sumer']
lam_sumer_unc = data_binned_loaded['lam_sumer_unc']
row_reference = data_binned_loaded['row_reference']
row_reference_binned = data_binned_loaded['row_reference_binned']


# Or if you want to execute the file that bins the data: (but remember to change the inputs in that file)
#exec(open("bin_data_interpolated.py").read())
"""
# Outputs:
spectral_image_interpolated_list
spectral_image_unc_interpolated_list
spectral_image_interpolated_croplat_list
spectral_image_unc_interpolated_croplat_list
lam_sumer
lam_sumer_unc
row_reference
spectral_image_interpolated_croplat_binned_list
spectral_image_unc_interpolated_croplat_binned_list
pixelscale_list_croplat_binned
pixelscale_unc_list_croplat_binned
pixelscale_intercept_list_croplat_binned
pixelscale_intercept_unc_list_croplat_binned
x_HPlon_rotcomp_binned
y_HPlat_crop_binned
"""

############################################################

## Ranges of wavelength
if line_label == 'NeVIII':
    wavelength_range_intensity_map = [1540.45, 1541.2] #Angstroem
    wavelength_range_intensity_map_bckg = [1539.8, 1540.2] #Angstroem
    line_center_label = 'Ne VIII - 770.428 \u212B'
elif line_label == 'SiII':
    wavelength_range_intensity_map = [1533.075, 1533.805] #Angstroem
    #wavelength_range_intensity_map = [1533.17, 1533.725] #Angstroem
    wavelength_range_intensity_map_bckg = [1535.10, 1536.60] #Angstroem
    line_center_label = 'Si II - 1533.43 \u212B'
elif line_label == 'CIV':
    wavelength_range_intensity_map = [1547.90, 1548.66] #Angstroem
    wavelength_range_intensity_map_bckg = [1545.93, 1547.65] #Angstroem
    line_center_label = 'C IV - 1548.21 \u212B'
elif line_label == 'cold_line':
    wavelength_range_intensity_map = [1537.80, 1538.10] #Angstroem
    wavelength_range_intensity_map_bckg = [1538.30, 1538.39] #Angstroem
    line_center_label = 'Si I - 1537.94 \u212B'
    
############################################################

pixelscale_reference_croplat_binned = pixelscale_list_croplat_binned[row_reference//bin_lat]
pixelscale_intercept_reference_croplat_binned = pixelscale_intercept_list_croplat_binned[row_reference//bin_lat]

# For the wavelength ranges in the input: find the pixel range and the exact wavelength range (from the center of the pixels of the range)
## Line to analyze:
w_px_range, w_cal_range_pixcenter = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=wavelength_range_intensity_map, slope_cal=pixelscale_reference_croplat_binned, intercept_cal=pixelscale_intercept_reference_croplat_binned)
## Background to subtract:
w_px_range_bckg, w_cal_range_pixcenter_bckg = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=wavelength_range_intensity_map_bckg, slope_cal=pixelscale_reference_croplat_binned, intercept_cal=pixelscale_intercept_reference_croplat_binned)


############################################################

# Select one spectral image binned
spectral_image_interpolated_croplat_binned = spectral_image_interpolated_croplat_binned_list[0]
spectral_image_unc_interpolated_croplat_binned = spectral_image_unc_interpolated_croplat_binned_list[0]


N_rows = spectral_image_interpolated_croplat_binned.shape[0]

px_half = (lam_sumer[1] - lam_sumer[0])/2
extent = [lam_sumer[0]-px_half, lam_sumer[-1]-px_half, N_rows+0.5, 0-0.5]

if show_wavelength_range == 'yes':

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(spectral_image_interpolated_croplat_binned, cmap='Greys', aspect='auto', norm=LogNorm(), extent=extent)
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title(f'SOHO/SUMER, spectral image calibrated and binned, index {0}', fontsize=18) 
    ax.set_xlabel('Wavelength (\u212B)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)
    ax.axvline(x=w_cal_range_pixcenter[0], linestyle='-', color='blue', linewidth=2, label='Left of the range of Ne VIII to integrate')
    ax.axvline(x=w_cal_range_pixcenter[1], linestyle='-', color='cyan', linewidth=2, label='Right of the range of Ne VIII to integrate')
    ax.axvline(x=w_cal_range_pixcenter_bckg[0], linestyle='-', color='red', linewidth=2, label='Left of the range of continuum')
    ax.axvline(x=w_cal_range_pixcenter_bckg[1], linestyle='-', color='orange', linewidth=2, label='Right of the range of continuum')
    ax.legend(fontsize=10)
    plt.show(block=False)



    x_profile = lam_sumer[w_px_range[0]:w_px_range[1]+1]
    y_profile = spectral_image_interpolated_croplat_binned.mean(axis=0)[w_px_range[0]:w_px_range[1]+1]
    N_rows = spectral_image_interpolated_croplat_binned.shape[0]
    y_unc_profile = 1/(N_rows) * np.sqrt((spectral_image_unc_interpolated_croplat_binned**2).sum(axis=0))[w_px_range[0]:w_px_range[1]+1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.errorbar(x=x_profile, y=y_profile, yerr=y_unc_profile, color='blue', linewidth=1.)
    ax.set_title(f'SOHO/SUMER, averaged profile of Ne VIII in the range used for the intensity map', fontsize=16) 
    ax.set_xlabel('Wavelength (\u212B)', color='black', fontsize=16)
    ax.set_ylabel(f'Av. intensity [W/sr/m^2/Angstroem]', color='black', fontsize=16)
    plt.show(block=False)

############################################################

intensity_map_croplat_binned, intensity_map_unc_croplat_binned = create_spectroheliogram_from_interpolation(spectralimage_interpolated_list=spectral_image_interpolated_croplat_binned_list, spectralimage_unc_interpolated_list=spectral_image_unc_interpolated_croplat_binned_list, w_px_range=w_px_range, w_px_range_bckg=w_px_range_bckg, slope_list=pixelscale_list_croplat_binned, slope_unc_list=pixelscale_unc_list_croplat_binned)

############################################################

if show_intensitymap_binned == 'yes':
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 6))
    img = ax.imshow(intensity_map_croplat_binned, cmap='Greys_r', norm=LogNorm(), aspect='auto')
    cbar = fig.colorbar(img, ax=ax, pad=0.03)
    cbar.set_label(f'Integrated spectral radiance [W/sr/m^2]', fontsize=16)
    ax.set_title(f'SOHO/SUMER intensity map binned ({round(wavelength_range_intensity_map[0],2)} - {round(wavelength_range_intensity_map[1],2)}) \u212B', fontsize=18)
    ax.set_xlabel('Helioprojective longitude (pixels)', fontsize=16)
    ax.set_ylabel('Helioprojective latitude (pixels)', fontsize=16)
    ax.axis('auto') # Ensures equal scaling of axis x and y
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 6))
    img = ax.imshow(intensity_map_unc_croplat_binned, cmap='Greys_r', norm=LogNorm(), aspect='auto')
    cbar = fig.colorbar(img, ax=ax, pad=0.03)
    cbar.set_label(f'Integrated spectral radiance [W/sr/m^2]', fontsize=16)
    ax.set_title(f'SOHO/SUMER uncertainties of the intensity map binned ({round(wavelength_range_intensity_map[0],2)} - {round(wavelength_range_intensity_map[1],2)}) \u212B', fontsize=18)
    ax.set_xlabel('Helioprojective longitude (pixels)', fontsize=16)
    ax.set_ylabel('Helioprojective latitude (pixels)', fontsize=16)
    ax.axis('auto') # Ensures equal scaling of axis x and y
    plt.show(block=False)

############################################################
#

# Save intensity map and uncertainties as .npy
if save_intensity_map == 'yes':
    filename_profile = 'intensity_map_'+line_label+f'_binned_lon{bin_lon}_lat{bin_lat}.npz'
    foldepath_profile = '../outputs/'
    np.savez(foldepath_profile+filename_profile, intensity_map_croplat_binned=intensity_map_croplat_binned, intensity_map_unc_croplat_binned=intensity_map_unc_croplat_binned)


"""
In order to load the intensity map in another file (or this one), do the next:

intensitymap_loaded_dic = np.load('intensity_map_'+line_label+f'_binned_lon{bin_lon}_lat{bin_lat}.npz')
intensity_map_croplat_binned = intensitymap_loaded_dic['intensity_map_croplat_binned'] #[W/sr/m^2] 2D-array
intensity_map_unc_croplat_binned = intensitymap_loaded_dic['intensity_map_unc_croplat_binned'] #[W/sr/m^2] 2D-array
"""

############################################################


print('--------------------------------------')
print('line_label =', line_label)
print('wavelength_range_intensity_map')
print('wavelength_range_intensity_map_bckg')
print('line_center_label')
print('pixelscale_reference_croplat_binned')
print('pixelscale_intercept_reference_croplat_binned')
print('w_px_range')
print('w_cal_range_pixcenter')
print('w_px_range_bckg')
print('w_cal_range_pixcenter_bckg')
print('intensity_map_croplat_binned')
print('intensity_map_unc_croplat_binned')
print('--------------------------------------')


############################################################





