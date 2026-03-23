#####################################
#####################################
#####################################
# Inputs

"""
# Threshold value: label (type) and range of percentageRange percentage of the threshold value
threshold_value_type = 'mean' #'max', 'min', 'mean', 'median', 'max_region', 'min_region', 'mean_region', 'median_region', 'CH_limits_rough'
range_percentage = [0., 50.]

# Wavelength ranges to crop spectra
wavelength_range_to_average_suggest = [1531.1147, 1551.7688]
"""


#####################################
#####################################
#####################################
# Import packages, functions, variables, data, spectral image...


#####################################
# Import packages, functions, variables...

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
import pandas as pd
from scipy.ndimage import gaussian_filter1d


from auxiliar_files.data_path import path_data_soho 
from auxiliar_files.SOHO_aux_functions import *
from auxiliar_files.calibration_parameters import *
from auxiliar_files.spectroheliogram_functions import *
from auxiliar_files.solar_rotation_variables import *
from auxiliar_files.aux_functions import *
from NeVIII_rest_wavelength import * #NeVIII_theoretical_wavelength_dic, NeVIII_theoretical_wavelength_color_dic 

from wcal_parameters import * #slit_bottom_px, slit_top_px


#####################################
# Import spectral images interpolated

# Intensities
loaded = np.load('spectral_image_list_intepolated_and_wavelength___.npz')
spectral_image_interp_list = [loaded[f'arr_{i}'] for i in range(240)] #W/sr/m$^2/Angstrom$
spectral_image_interp_croplat_list = [sp_img[slit_top_px:slit_bottom_px+1] for sp_img in spectral_image_interp_list] #W/sr/m$^2/Angstrom$

# Wavelength
lam_sumer_full = 10.*loaded['arr_240'][0,:] #Angstrom

# Intensity's uncertainties
loaded_unc = np.load('spectral_image_unc_list_intepolated___.npz')
spectral_image_unc_interp_list = [loaded_unc[f'arr_{i}'] for i in range(240)] #W/sr/m$^2/Angstrom$
spectral_image_unc_interp_croplat_list = [sp_img_unc[slit_top_px:slit_bottom_px+1] for sp_img_unc in spectral_image_unc_interp_list] #W/sr/m$^2/Angstrom$


#####################################
# Import intensity map of Ne VIII
spectroheliogram_data_name = 'spectroheliogram_NeVIII_interpolated___.npy'
intensity_map, intensity_map_nobg = np.load(spectroheliogram_data_name)
intensity_map_croplat = intensity_map[slit_top_px:slit_bottom_px+1, :]


#####################################
# Import dictionary of threshold values
threshold_value_NeVIII_dic = {'max': [0.473378338955035, 'maximum of full intensity map'], 'min': [0.009908022024473434, 'minimum of full intensity map'], 'mean': [0.057475782830626015, 'mean of full intensity map'], 'median': [0.052268053739969414, 'median of full intensity map'], 'max_region': [0.12810939316200312, 'maximum of region selected'], 'min_region': [0.026764306219467925, 'minimum of region selected'], 'mean_region': [0.0556772714485083, 'mean of region selected'], 'median_region': [0.052513185113243946, 'median of region selected'], 'CH_limits_rough': [0.043, 'approximate values of the limits of the coronal hole']}


#####################################
#####################################
#####################################
# Functions

#####################################
# Pixel addresses, range of intensity

def range_intensity_addresses_of_image(range_percentage, image_array, threshold_value='max'):
    """
    This function extracts all the addresses (row index, column index) of the pixels whose intensities are between a certain range of percentage (given in the input) of the image's maximum intensity. 
    """
    proportion_range0 = range_percentage[0]/100.
    proportion_range1 = range_percentage[1]/100.
    
    if threshold_value=='max': thr_val = np.max(image_array) #maximum of the intensity map
    elif threshold_value=='average' or threshold_value=='mean': thr_val = np.mean(image_array) # minimum of the intensity map
    else: thr_val = threshold_value # value defined by the user
    
    lower_bound = proportion_range0 * thr_val
    upper_bound = proportion_range1 * thr_val
    rowscols_inside_range_arr = np.argwhere((image_array>=lower_bound) & (image_array<=upper_bound))
    
    # convert to list
    rowscols_inside_range = []
    for [row, col] in rowscols_inside_range_arr:
        rowscols_inside_range.append([row, col])
    
    return [rowscols_inside_range, lower_bound, upper_bound] #[[lower percentage, upper percentage], float, float] of the maximum intensity    

def range_intensity_addresses_of_SUMER_spectroheliogram(range_percentage, spectroheliogram_croplat, slit_top_px, threshold_value='max'):
    rowscols_inside_range_croplat = range_intensity_addresses_of_image(range_percentage=range_percentage, image_array=spectroheliogram_croplat, threshold_value=threshold_value)[0]
    
    rowscols_inside_range = []
    for i, [row, col] in enumerate(rowscols_inside_range_croplat):
        rowscols_inside_range.append([row+slit_top_px, col])
    
    return [rowscols_inside_range, rowscols_inside_range_croplat]

#####################################
# Plot intensity map, contours, highlights...

def plot_intensitymap_NeVIII(intensity_map, vmin_sumer=8e-3, vmax_sumer=1e-0, title='auto'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(intensity_map, cmap='Greys_r', aspect='auto', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
    cax = fig.add_axes([0.9, 0.05, 0.03, 0.90])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Intensity [W/sr/m^2]', fontsize=16)
    if title=='auto': ax.set_title(f'SOHO/SUMER intensity map of Ne VIII 770.428 \u212B', fontsize=18)
    else: ax.set_title(title, fontsize=18)
    ax.set_xlabel('Wavelength dimension (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial dimension (pixels)', color='black', fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)


def plot_intensitymap_NeVIII_with_contours(intensity_map, threshold_value, range_percentage, vmin_sumer=8e-3, vmax_sumer=1e-0, title='auto'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(intensity_map, cmap='Greys_r', aspect='auto', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
    cax = fig.add_axes([0.9, 0.05, 0.03, 0.90])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Intensity [W/sr/m^2]', fontsize=16)
    if title=='auto': ax.set_title(f'SOHO/SUMER intensity map of Ne VIII 770.428 \u212B', fontsize=18)
    else: ax.set_title(title, fontsize=18)
    ax.set_xlabel('Wavelength dimension (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial dimension (pixels)', color='black', fontsize=16)
    
    lower_bound = range_percentage[0]/100. * threshold_value
    upper_bound = range_percentage[1]/100. * threshold_value

    contour_lower = ax.contour(intensity_map, levels=[lower_bound], colors='cyan', linewidths=2)
    contour_upper = ax.contour(intensity_map, levels=[upper_bound], colors='blue', linewidths=2)

    legend_elements = [
        mlines.Line2D([],[],color='cyan', label=f'{range_percentage[0]} %'),
        mlines.Line2D([],[],color='blue', label=f'{range_percentage[1]} %')
    ]

    ax.legend(handles=legend_elements, title=f'Threshold value: {np.round(threshold_value, 5)}')
    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)
    
    return [lower_bound, upper_bound]



def plot_intensitymap_NeVIII_with_contours_and_highlights(intensity_map, threshold_value, range_percentage, vmin_sumer=8e-3, vmax_sumer=1e-0, title='auto', contours_highlights_both='highlights'):
    
    # Thresholds
    lower_bound = range_percentage[0]/100. * threshold_value
    upper_bound = range_percentage[1]/100. * threshold_value

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    # Show image
    img = ax.imshow(intensity_map, cmap='Greys_r', aspect='auto', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
    cax = fig.add_axes([0.9, 0.05, 0.03, 0.90])
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Intensity [W/sr/m^2]', fontsize=16)
    
    if contours_highlights_both=='highlights' or contours_highlights_both=='both':
    
        # Highlight pixels within thresholds
        mask = (intensity_map >= lower_bound) & (intensity_map <= upper_bound)
        ys, xs = np.where(mask)
    
        # Compute marker size in points^2 so that each scatter square matches one imshow pixel
        fig_w, fig_h = fig.get_size_inches()*fig.dpi     # figure size in pixels
        ax_w, ax_h = ax.get_window_extent().width, ax.get_window_extent().height
        ny, nx = intensity_map.shape
        px_size_x = ax_w / nx   # pixel size in display units
        px_size_y = ax_h / ny
        px_size_points = 0.5*(px_size_x + px_size_y)# Convert to points (scatter uses area in points^2). Approximate with geometric mean of x and y pixel size
        marker_size = px_size_points**2  
    
        # Use scatter with square markers, size matches pixel size
        ax.scatter(xs, ys, marker='s', s=marker_size, facecolor='red', edgecolor='none', alpha=0.5, label=f'N pixels highlighted: {len(xs)}')
        
        
    if contours_highlights_both=='contours' or contours_highlights_both=='both':
        contour_lower = ax.contour(intensity_map, levels=[lower_bound], colors='cyan', linewidths=2)
        contour_upper = ax.contour(intensity_map, levels=[upper_bound], colors='blue', linewidths=2)

        legend_elements = [
            mlines.Line2D([],[],color='cyan', label=f'{range_percentage[0]} %'),
            mlines.Line2D([],[],color='blue', label=f'{range_percentage[1]} %')
        ]

    # Labels and title
    ax.set_title(f'SOHO/SUMER intensity map of Ne VIII 770.428 \u212B', fontsize=18)
    ax.set_xlabel('Wavelength dimension (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial dimension (pixels)', color='black', fontsize=16)
    ax.legend()

    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)


#####################################
# Functions to average spectra of the regions inside the intensity range
def w_image_pixels_to_calibrated_wavelength(w_px_img, slope_cal, intercept_cal): 
    """
    This function converts the pixel numbers of the x axis (wavelength direction) in the image to calibrated wavelength values (in Angstrom). 
    """
    w_cal = slope_cal*w_px_img + intercept_cal
    return w_cal #[Angstrom]


def w_calibrated_wavelength_to_image_pixels(w_cal, slope_cal, intercept_cal):
    """
    This function is the inverse of image_pixels_to_calibrated_wavelength()
    """
    m = 1/slope_cal
    b = -intercept_cal/slope_cal
    w_px_img = m * w_cal + b 
    return w_px_img #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(w_px) in case s_px is one number, or np.floor(w_px).astype(int) in case s_px is an array.


# Convert calibrated wavelength range to (closest) pixel range
def w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range, slope_cal, intercept_cal):
    # Calculate the pixel range (floats)
    w_px_range0_float = w_calibrated_wavelength_to_image_pixels(w_cal=w_cal_range[0], slope_cal=slope_cal, intercept_cal=intercept_cal)
    w_px_range1_float = w_calibrated_wavelength_to_image_pixels(w_cal=w_cal_range[1], slope_cal=slope_cal, intercept_cal=intercept_cal)
    
    # Round and convert to integer 
    w_px_range0 = int(np.round(w_px_range0_float))
    w_px_range1 = int(np.round(w_px_range1_float))
    w_px_range =  [w_px_range0,  w_px_range1]
    
    # Calculate the range of wavelength calibrated from the center of the pixels range
    w_cal_range0_pixcenter = w_image_pixels_to_calibrated_wavelength(w_px_img=float(w_px_range0), slope_cal=slope_cal, intercept_cal=intercept_cal)
    w_cal_range1_pixcenter = w_image_pixels_to_calibrated_wavelength(w_px_img=float(w_px_range1), slope_cal=slope_cal, intercept_cal=intercept_cal)
    w_cal_range_pixcenter = [w_cal_range0_pixcenter, w_cal_range1_pixcenter]
    
    return [w_px_range, w_cal_range_pixcenter]

def average_profiles_from_pixels_selected(wavelength_range_suggest, wavelength_array, spectra_interpolated_list, unc_spectra_interpolated_list, rows_cols_of_spectroheliogram):
    """
    Inputs:
        - wavelength_range_suggest: range of wavelength to analyze (to average)
        - wavelength_array: list or 1d-array of wavelength calibrated (corresponding to the center of the pixels)
        - spectra_interpolated_list: list of spectral images
        - unc_spectra_interpolated_list: uncertainties of "spectra_interpolated_list"
        - rows_cols_of_spectroheliogram: list of pixels addresses [row, col] of the spectroheliogram. Each row of the spectroheliogram corresponds to the same row in an spectral image, and each col (column) of the spectroheliogram corresponds to an spectral image.
    """
    
    # Parameters to convert pixels to wavelength
    pixelscale_reference = wavelength_array[1]-wavelength_array[0]
    pixelscale_intercept_reference = wavelength_array[0]
    
    # Calculate the range of pixels and Angstrom in the wavelength direction for each row
    w_px_range, w_cal_range_pixcenter = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=wavelength_range_suggest, slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference)
    col0, col1 = w_px_range
    N_pixels_range = col1 - col0 + 1
    
    profile_intensity_line_sum = np.zeros(N_pixels_range)
    unc_profile_intensity_line_sumsq = np.zeros(N_pixels_range)
    N_pixels_adresses = 0
    for row_lat, idx_spec in rows_cols_of_spectroheliogram:
        spectrum_array = spectra_interpolated_list[idx_spec]
        spectrum_array_unc = unc_spectra_interpolated_list[idx_spec]
        
        profile = spectrum_array[row_lat, col0:col1+1]
        profile_unc = spectrum_array_unc[row_lat, col0:col1+1]

        # Skip if NaNs present
        if np.any(np.isnan(profile)) or np.any(np.isnan(profile_unc)):
            continue

        profile_intensity_line_sum += profile
        unc_profile_intensity_line_sumsq += profile_unc**2
        N_pixels_adresses += 1
    
    # Wavelength array and average the summed intensities
    profile_wavelength_line = w_image_pixels_to_calibrated_wavelength(w_px_img=np.arange(col0, col1+1), slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference)
    profile_intensity_line_average = (1/N_pixels_adresses) * profile_intensity_line_sum
    profile_intensity_unc_line_average = (1/N_pixels_adresses) * np.sqrt(unc_profile_intensity_line_sumsq)
    
    return [profile_wavelength_line, profile_intensity_line_average, profile_intensity_unc_line_average]
    


def average_profiles_from_pixels_selected_normalized(wavelength_range_suggest, wavelength_array, spectra_interpolated_list, unc_spectra_interpolated_list, rows_cols_of_spectroheliogram, intensitymap_):
    """
    Inputs:
        - wavelength_range_suggest: range of wavelength to analyze (to average)
        - wavelength_array: list or 1d-array of wavelength calibrated (corresponding to the center of the pixels)
        - spectra_interpolated_list: list of spectral images
        - unc_spectra_interpolated_list: uncertainties of "spectra_interpolated_list"
        - rows_cols_of_spectroheliogram: list of pixels addresses [row, col] of the spectroheliogram. Each row of the spectroheliogram corresponds to the same row in an spectral image, and each col (column) of the spectroheliogram corresponds to an spectral image.
    """
    
    # Parameters to convert pixels to wavelength
    pixelscale_reference = wavelength_array[1]-wavelength_array[0]
    pixelscale_intercept_reference = wavelength_array[0]
    
    # Calculate the range of pixels and Angstrom in the wavelength direction for each row
    w_px_range, w_cal_range_pixcenter = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=wavelength_range_suggest, slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference)
    col0, col1 = w_px_range
    N_pixels_range = col1 - col0 + 1
    
    profile_intensity_line_sum = np.zeros(N_pixels_range)
    unc_profile_intensity_line_sumsq = np.zeros(N_pixels_range)
    N_pixels_adresses = 0
    for row_lat, idx_spec in rows_cols_of_spectroheliogram:
        spectrum_array = spectra_interpolated_list[idx_spec]
        spectrum_array_unc = unc_spectra_interpolated_list[idx_spec]
        
        factor_norm = profile * pixelscale_reference
        
        profile = spectrum_array[row_lat, col0:col1+1] / factor_norm
        profile_unc = spectrum_array_unc[row_lat, col0:col1+1] / factor_norm

        # Skip if NaNs present
        if np.any(np.isnan(profile)) or np.any(np.isnan(profile_unc)):
            continue

        profile_intensity_line_sum += profile
        unc_profile_intensity_line_sumsq += profile_unc**2
        N_pixels_adresses += 1
    
    # Wavelength array and average the summed intensities
    profile_wavelength_line = w_image_pixels_to_calibrated_wavelength(w_px_img=np.arange(col0, col1+1), slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference)
    profile_intensity_line_average = (1/N_pixels_adresses) * profile_intensity_line_sum
    profile_intensity_unc_line_average = (1/N_pixels_adresses) * np.sqrt(unc_profile_intensity_line_sumsq)
    
    return [profile_wavelength_line, profile_intensity_line_average, profile_intensity_unc_line_average]




