# INPUTS
"""
# Binning
bin_lat = 4
bin_lon = 1

# Line
line_label = 'CIV' #'NeVIII', 'SiII', 'CIV', 'cold_line'


show_spectral_ranges = 'no'
show_intensity_maps = 'yes'
"""


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
from auxiliar_functions.data_path import path_data_soho 
from auxiliar_functions.SOHO_aux_functions import *
from auxiliar_functions.calibration_parameters__output import *
from auxiliar_functions.spectroheliogram_functions import *
from auxiliar_functions.solar_rotation_variables import *
from auxiliar_functions.aux_functions import *
from auxiliar_functions.general_variables import *
from auxiliar_functions.NeVIII_rest_wavelength import *
from auxiliar_functions.auxfuncs_binning_and_dopplermap import *
from scale_hrts import *



## Ranges of wavelength
if line_label == 'NeVIII':
    wavelength_range_spectroheliogram = [1540.45, 1541.2] #Angstroem
    wavelength_range_spectroheliogram_bckg = [1539.8, 1540.2] #Angstroem
    line_center_label = 'Ne VIII - 770.428 \u212B'
elif line_label == 'SiII':
    wavelength_range_spectroheliogram = [1533.075, 1533.805] #Angstroem
    #wavelength_range_spectroheliogram = [1533.17, 1533.725] #Angstroem
    wavelength_range_spectroheliogram_bckg = [1535.10, 1536.60] #Angstroem
    line_center_label = 'Si II - 1533.43 \u212B'
elif line_label == 'CIV':
    wavelength_range_spectroheliogram = [1547.90, 1548.66] #Angstroem
    wavelength_range_spectroheliogram_bckg = [1545.93, 1547.65] #Angstroem
    line_center_label = 'C IV - 1548.21 \u212B'
elif line_label == 'cold_line':
    wavelength_range_spectroheliogram = [1537.80, 1538.10] #Angstroem
    wavelength_range_spectroheliogram_bckg = [1538.30, 1538.39] #Angstroem
    line_center_label = 'Si I - 1537.94 \u212B'



# x and y axes (helioprojective longitude and latitude)
filename_eit = 'SOHO_EIT_195_19991107T042103_L1.fits'
sumer_header_list, sumer_data_list, sumer_data_unc_list = SUMERraster_get_data_header_and_datauncertainties(sumer_filepath=path_data_soho+'sumer/', sumer_filename_list=filename_list, factor_fullspectrum=factor_fullspectrum, t_exp_sec=t_exp)
x_HPlon_rotcomp = HPlon_raster_rotcomp_dic[filename_eit]
y_HPlat_crop = s_image_pixels_to_helioprojective_latitude(s_px_img=np.arange(slit_top_px,slit_bottom_px+1), header_sumer=sumer_header_list[0])


#####################################
# Import SUMER data interpolated (wavelength calibrated)
data_interpolated_loaded = np.load('../auxiliar_functions/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)
spectral_image_interpolated_list = data_interpolated_loaded['spectral_image_interpolated_list']
spectral_image_unc_interpolated_list = data_interpolated_loaded['spectral_image_unc_interpolated_list']
spectral_image_interpolated_croplat_list = data_interpolated_loaded['spectral_image_interpolated_croplat_list']
spectral_image_unc_interpolated_croplat_list = data_interpolated_loaded['spectral_image_unc_interpolated_croplat_list']
lam_sumer = data_interpolated_loaded['reference_wavelength']          # scalar (0‑d array; use a_loaded.item() for Python float)
lam_sumer_unc = data_interpolated_loaded['unc_reference_wavelength'] #uncertainty of lam_sumer
row_reference = int(data_interpolated_loaded['row_reference'])        # becomes a NumPy array or object array, so I conver it to integer again



#####################################
# 
# Calculate range of pixels   # For the wavelength ranges in the input: find the pixel range and the exact wavelength range (from the center of the pixels of the range)
pixelscale_reference = pixelscale_list[row_reference]
pixelscale_intercept_reference = pixelscale_intercept_list[row_reference]
w_px_range, w_cal_range_pixcenter = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=wavelength_range_spectroheliogram, slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference) # Line to analyze
w_px_range_bckg, w_cal_range_pixcenter_bckg = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=wavelength_range_spectroheliogram_bckg, slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference) # Background to subtract




# Show spectral image with vertical lines showing the wavelength range, and averaged profile
if show_spectral_ranges == 'yes':
    spectral_image_interp_croplat_0 = spectral_image_interpolated_croplat_list[0]
    unc_spectral_image_interp_croplat_0 = spectral_image_unc_interpolated_croplat_list[0]


    # Plot the spectral image with these ranges
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    lam_sumer_px = np.arange(spectral_image_interp_croplat_0.shape[1]) 
    HPlat_crop_px = np.arange(spectral_image_interp_croplat_0.shape[0])
    img=ax.pcolormesh(lam_sumer_px, HPlat_crop_px, spectral_image_interp_croplat_0, cmap='Greys_r', norm=LogNorm())
    cax = fig.add_axes([0.99, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(r'Radiance [W/sr/m$^2$/''\u212B]', fontsize=16)
    ax.set_xlabel('wavelength direction (pixels)', fontsize=16)
    ax.set_ylabel('helioprojective latitude (pixels)', fontsize=16)
    ax.set_title(r'SUMER spectral image interpolated and cropped in $y$-axis, 'f' {line_center_label}''\n marking the wavelength ranges for the intensity', fontsize=18) 
    ax.set_aspect('auto') #'auto', 'equal'
    ax.axvline(w_px_range[0], color='blue', linewidth=1.2, label='Range for intensity map')
    ax.axvline(w_px_range[1], color='blue', linewidth=1.2)
    ax.axvline(w_px_range_bckg[0], color='red', linewidth=1.2, label='Range for background of the intensity map')
    ax.axvline(w_px_range_bckg[1], color='red', linewidth=1.2)
    # Define the forward (pixel -> wavelength) and inverse (wavelength -> pixel) mappings
    # Fit linear mapping λ = a*pix + b
    slope_x = (lam_sumer[-1]-lam_sumer[0]) / (lam_sumer_px[-1]-lam_sumer_px[0])
    intercept_x = lam_sumer[0]
    def pixel_to_lambda(x):
        return slope_x * x + intercept_x
    def lambda_to_pixel(l):
        return (l - intercept_x) / slope_x
    # Create top axis
    secax = ax.secondary_xaxis('top', functions=(pixel_to_lambda, lambda_to_pixel))
    secax.set_xlabel('Wavelength [Å]', fontsize=16)
    slope_y = (y_HPlat_crop[-1]-y_HPlat_crop[0]) / (HPlat_crop_px[-1]-HPlat_crop_px[0])
    intercept_y = y_HPlat_crop[-1] - slope_y*HPlat_crop_px[-1]
    def pixel_to_HPlat(x):
        return slope_y * x + intercept_y
    def HPlat_to_pixel(lt):
        return (lt - intercept_y) / slope_y
    secay = ax.secondary_yaxis('right', functions=(pixel_to_HPlat, HPlat_to_pixel))
    secay.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    plt.show(block=False)




    # Profile 
    x_profile = lam_sumer_px
    y_profile = spectral_image_interp_croplat_0.mean(axis=0)
    yerr_profile = (1/spectral_image_interp_croplat_0.shape[0]) * np.sqrt((unc_spectral_image_interp_croplat_0**2).sum(axis=0))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    ax.errorbar(x=x_profile, y=y_profile, yerr=yerr_profile, color='black', linewidth=1.5, label='Averaged profile along the slit')
    ax.set_xlabel('wavelength direction (pixels)', fontsize=16)
    ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', fontsize=16)
    ax.set_title('SUMER spectral profile averaged along\n the spatial direction of the interpolated spectral image', fontsize=18) 
    ax.set_aspect('auto') #'auto', 'equal'
    ax.axvspan(w_px_range[0], w_px_range[1], color='blue', alpha=0.15, label='Range for intensity map')
    ax.axvspan(w_px_range_bckg[0], w_px_range_bckg[1], color='red', alpha=0.15, label='Range for background of the intensity map')
    # Define the forward (pixel -> wavelength) and inverse (wavelength -> pixel) mappings
    # Fit linear mapping λ = a*pix + b
    slope_x = (lam_sumer[-1]-lam_sumer[0]) / (lam_sumer_px[-1]-lam_sumer_px[0])
    intercept_x = lam_sumer[0]
    def pixel_to_lambda(x):
        return slope_x * x + intercept_x
    def lambda_to_pixel(l):
        return (l - intercept_x) / slope_x
    # Create top axis
    secax = ax.secondary_xaxis('top', functions=(pixel_to_lambda, lambda_to_pixel))
    secax.set_xlabel('Wavelength [Å]', fontsize=16)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    plt.show(block=False)


#######################################
#######################################
#######################################
# Binned


# Bin spectral images interpolated (interpolated = wavelength calibrated) and cropped in lattitude
spectral_image_interpolated_croplat_list_binned = bin_2Darray_list_and_y_axis(arr2D_list=spectral_image_interpolated_croplat_list, y_bin=bin_lat, img_bin=bin_lon, average_sum='average')
spectral_image_unc_interpolated_croplat_list_binned = bin_2Darray_list_and_y_axis_unc(arr2D_list=spectral_image_unc_interpolated_croplat_list, y_bin=bin_lat, img_bin=bin_lon, average_sum='average')

# Bin 1D-arrays of pixel scale and intercept (of the calibration) of the list cropped in lattitude
pixelscale_list_croplat_binned = bin_1Darray(arr=pixelscale_list_croplat, N_bin=bin_lat, average_sum='average')
pixelscale_intercept_list_croplat_binned = bin_1Darray(arr=pixelscale_intercept_list_croplat, N_bin=bin_lat, average_sum='average')

# Bin arrays of longitude and latitude
x_HPlon_rotcomp_binned = bin_1Darray(arr=x_HPlon_rotcomp, N_bin=bin_lon, average_sum='average')
y_HPlat_crop_binned = bin_1Darray(arr=y_HPlat_crop, N_bin=bin_lat, average_sum='average')

# Binned. Show spectral image with vertical lines showing the wavelength range, and averaged profile
if show_spectral_ranges == 'yes':
    spectral_image_interp_croplat_binned_0 = spectral_image_interpolated_croplat_list_binned[0]
    unc_spectral_image_interp_croplat_binned_0 = spectral_image_unc_interpolated_croplat_list_binned[0]

    # Plot the spectral image with these ranges
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    lam_sumer_px = np.arange(spectral_image_interp_croplat_binned_0.shape[1]) 
    HPlat_crop_binned_px = np.arange(spectral_image_interp_croplat_binned_0.shape[0])
    img=ax.pcolormesh(lam_sumer_px, HPlat_crop_binned_px, spectral_image_interp_croplat_binned_0, cmap='Greys_r', norm=LogNorm())
    cax = fig.add_axes([0.99, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(r'Radiance [W/sr/m$^2$/''\u212B]', fontsize=16)
    ax.set_xlabel('wavelength direction (pixels)', fontsize=16)
    ax.set_ylabel('helioprojective latitude (pixels)', fontsize=16)
    ax.set_title(r'SUMER spectral image interpolated and cropped in $y$-axis, 'f'{line_center_label}''\n marking the wavelength ranges for the intensity. Binned', fontsize=18) 
    ax.set_aspect('auto') #'auto', 'equal'
    ax.axvline(w_px_range[0], color='blue', linewidth=1.2, label='Range for intensity map')
    ax.axvline(w_px_range[1], color='blue', linewidth=1.2)
    ax.axvline(w_px_range_bckg[0], color='red', linewidth=1.2, label='Range for background of the intensity map')
    ax.axvline(w_px_range_bckg[1], color='red', linewidth=1.2)
    # Define the forward (pixel -> wavelength) and inverse (wavelength -> pixel) mappings
    # Fit linear mapping λ = a*pix + b
    slope_x = (lam_sumer[-1]-lam_sumer[0]) / (lam_sumer_px[-1]-lam_sumer_px[0])
    intercept_x = lam_sumer[0]
    def pixel_to_lambda(x):
        return slope_x * x + intercept_x
    def lambda_to_pixel(l):
        return (l - intercept_x) / slope_x
    # Create top axis
    secax = ax.secondary_xaxis('top', functions=(pixel_to_lambda, lambda_to_pixel))
    secax.set_xlabel('Wavelength [Å]', fontsize=16)
    slope_y = (y_HPlat_crop_binned[-1]-y_HPlat_crop_binned[0]) / (HPlat_crop_binned_px[-1]-HPlat_crop_binned_px[0])
    intercept_y = y_HPlat_crop_binned[-1] - slope_y*HPlat_crop_binned_px[-1]
    def pixel_to_HPlat(x):
        return slope_y * x + intercept_y
    def HPlat_to_pixel(lt):
        return (lt - intercept_y) / slope_y
    secay = ax.secondary_yaxis('right', functions=(pixel_to_HPlat, HPlat_to_pixel))
    secay.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    plt.show(block=False)




    # Profile 
    x_profile = lam_sumer_px
    y_profile = spectral_image_interp_croplat_binned_0.mean(axis=0)
    yerr_profile = (1/spectral_image_interp_croplat_binned_0.shape[0]) * np.sqrt((unc_spectral_image_interp_croplat_binned_0**2).sum(axis=0))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    ax.errorbar(x=x_profile, y=y_profile, yerr=yerr_profile, color='black', linewidth=1.5, label='Averaged profile along the slit')
    ax.set_xlabel('wavelength direction (pixels)', fontsize=16)
    ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', fontsize=16)
    ax.set_title('SUMER spectral profile averaged along\n the spatial direction of the interpolated spectral image. Binned', fontsize=18) 
    ax.set_aspect('auto') #'auto', 'equal'
    ax.axvspan(w_px_range[0], w_px_range[1], color='blue', alpha=0.15, label='Range for intensity map')
    ax.axvspan(w_px_range_bckg[0], w_px_range_bckg[1], color='red', alpha=0.15, label='Range for background of the intensity map')
    # Define the forward (pixel -> wavelength) and inverse (wavelength -> pixel) mappings
    # Fit linear mapping λ = a*pix + b
    slope_x = (lam_sumer[-1]-lam_sumer[0]) / (lam_sumer_px[-1]-lam_sumer_px[0])
    intercept_x = lam_sumer[0]
    def pixel_to_lambda(x):
        return slope_x * x + intercept_x
    def lambda_to_pixel(l):
        return (l - intercept_x) / slope_x
    # Create top axis
    secax = ax.secondary_xaxis('top', functions=(pixel_to_lambda, lambda_to_pixel))
    secax.set_xlabel('Wavelength [Å]', fontsize=16)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    plt.show(block=False)


# Create intensity map (binned and not binned)
intensitymap_croplat = create_spectroheliogram_from_interpolated_data(spectralimage_interpolated_list=spectral_image_interpolated_croplat_list, w_px_range=w_px_range, w_px_range_bckg=w_px_range_bckg, slope_list=pixelscale_list_croplat, intercept_list=pixelscale_intercept_list_croplat)
intensitymap_croplat_binned = create_spectroheliogram_from_interpolated_data(spectralimage_interpolated_list=spectral_image_interpolated_croplat_list_binned, w_px_range=w_px_range, w_px_range_bckg=w_px_range_bckg, slope_list=pixelscale_list_croplat_binned, intercept_list=pixelscale_intercept_list_croplat_binned)

# Show intensity maps
if show_intensity_maps == 'yes':
    plot_2Darray(array_2D=intensitymap_croplat, x_axis='pixels', y_axis='pixels', title=f'Intensity map {line_center_label}', x_label='auto', y_label='auto', z_label=r'Integrated radiance [W/sr/m$^2$]', cmap='Greys_r', z_scale='log', vmin_vmax='auto')
    plot_2Darray(array_2D=intensitymap_croplat_binned, x_axis='pixels', y_axis='pixels', title=f'Intensity map binned {line_center_label}', x_label='auto', y_label='auto', z_label=r'Integrated radiance [W/sr/m$^2$]', cmap='Greys_r', z_scale='log', vmin_vmax='auto')


print('--------------------------------------')
print('Line:', line_label)
print('spectral_image_interpolated_list')
print('spectral_image_unc_interpolated_list')
print('spectral_image_interpolated_croplat_list')
print('spectral_image_unc_interpolated_croplat_list')
print('lam_sumer')
print('lam_sumer_unc')
print('row_reference')
print('w_px_range'+line_label, '= w_px_range')
print('w_cal_range_pixcenter'+line_label, '= w_cal_range_pixcenter')
print('w_px_range_bckg'+line_label, '= w_px_range_bckg')
print('w_cal_range_pixcenter_bckg'+line_label, '= w_cal_range_pixcenter_bckg')
print('spectral_image_interpolated_croplat_list_binned')
print('spectral_image_unc_interpolated_croplat_list_binned')
print('pixelscale_list_croplat_binned')
print('pixelscale_intercept_list_croplat_binned')
print('x_HPlon_rotcomp_binned')
print('y_HPlat_crop_binned')
print('intensitymap_croplat'+line_label, '= intensitymap_croplat')
print('intensitymap_croplat_binned'+line_label, '= intensitymap_croplat_binned')
print('--------------------------------------')


