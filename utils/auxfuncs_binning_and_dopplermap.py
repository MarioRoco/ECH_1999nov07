##################################################
# Import things

import numpy as np
from scipy.optimize import curve_fit
from scipy.odr import Model, RealData, ODR
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from matplotlib.colors import LogNorm
import datetime as dt
from astropy.io import fits
import matplotlib.lines as mlines
from astropy.coordinates import SkyCoord


import sys
import os
sys.path.append(os.path.abspath('..'))
from auxiliar_functions.SOHO_aux_functions import *
from auxiliar_functions.calibration_parameters__output import *
from auxiliar_functions.spectroheliogram_functions import *
from auxiliar_functions.solar_rotation_variables import *
from auxiliar_functions.aux_functions import *
from auxiliar_functions.general_variables import *
from auxiliar_functions.NeVIII_rest_wavelength import *


##################################################
# binning functions

def bin_1Darray(arr, N_bin, average_sum='average'):
    """
    Bin a 1D array by averaging or adding every N_bin elements.
    
    Parameters:
        arr (ndarray): Input 1D array of shape (Nx)
        N_bin (int): Number of pixels to average together

    Returns:
        ndarray: Binned array of shape Nx//N_bin
    """
    arr = np.asarray(arr)
    # Trim array length to a multiple of N
    L = (len(arr) // N_bin) * N_bin
    arr_trimmed = arr[:L]
    
    if average_sum=='average': arr_binned = arr_trimmed.reshape(-1, N_bin).mean(axis=1)
    elif average_sum=='sum': arr_binned = arr_trimmed.reshape(-1, N_bin).sum(axis=1)
    
    return arr_binned


def bin_y_axis_of_2Darray(arr, Ny_bin, average_sum='average'):
    """
    Bin a 2D array along the y-axis by averaging or adding every Ny_bin rows.
    
    Parameters:
        arr (ndarray): Input 2D array of shape (Ny, Nx)
        Ny_bin (int): Number of pixels to average together along y-axis

    Returns:
        ndarray: Binned array of shape (Ny // Ny_bin, Nx)
    """
    Ny, Nx = arr.shape
    # Trim excess rows so Ny is divisible by Ny_bin
    Ny_trim = (Ny // Ny_bin) * Ny_bin
    arr_trimmed = arr[:Ny_trim, :]

    # Reshape and average along the new axis
    arr_reshaped = arr_trimmed.reshape(Ny_trim // Ny_bin, Ny_bin, Nx)
    
    if average_sum=='average': arr_binned = arr_reshaped.mean(axis=1)
    elif average_sum=='sum': arr_binned = arr_reshaped.sum(axis=1)
    
    return arr_binned

def bin_y_axis_of_2Darray_unc(arr, Ny_bin, average_sum='average'):
    Ny, Nx = arr.shape
    # Trim excess rows so Ny is divisible by Ny_bin
    Ny_trim = (Ny // Ny_bin) * Ny_bin
    arr_trimmed = arr[:Ny_trim, :]

    # Reshape and average along the new axis
    arr_reshaped = arr_trimmed.reshape(Ny_trim // Ny_bin, Ny_bin, Nx)
    
    if average_sum=='average': arr_binned = (1/Ny_bin) * np.sqrt((arr_reshaped**2).sum(axis=1))
    elif average_sum=='sum': arr_binned = np.sqrt((arr_reshaped**2).sum(axis=1))
    
    return arr_binned



def bin_2Darray_list(arr_list, N_bin, average_sum='average'):
    # Ensure arr_list is a NumPy array for easy slicing
    stacked = np.stack(arr_list)  # shape: (num_arrays, height, width)
    
    # Number of full groups
    num_groups = len(arr_list) // N_bin
    
    # Reshape and take the mean over the group axis
    grouped = stacked[:num_groups * N_bin].reshape(num_groups, N_bin, *arr_list[0].shape)
    
    if average_sum=='average': averaged = grouped.mean(axis=1)
    elif average_sum=='sum': averaged = grouped.sum(axis=1)
    
    # Return as a list of 2D arrays
    return [arr for arr in averaged]

def bin_2Darray_list_unc(arr_list, N_bin, average_sum='average'):
    # Ensure arr_list is a NumPy array for easy slicing
    stacked = np.stack(arr_list)  # shape: (num_arrays, height, width)
    
    # Number of full groups
    num_groups = len(arr_list) // N_bin
    
    # Reshape and take the mean over the group axis
    grouped = stacked[:num_groups * N_bin].reshape(num_groups, N_bin, *arr_list[0].shape)
    
    if average_sum=='average': 
        averaged = (1/N_bin) * np.sqrt((grouped**2).sum(axis=1))
    elif average_sum=='sum': 
        averaged = np.sqrt((grouped**2).sum(axis=1))
    
    # Return as a list of 2D arrays
    return [arr for arr in averaged]




def bin_2Darray_list_and_y_axis(arr2D_list, y_bin, img_bin, average_sum='average'):
    arr2D_list_grouped = bin_2Darray_list(arr_list=arr2D_list, N_bin=img_bin, average_sum=average_sum)
    
    arr2D_list_grouped_binned = []
    for arr2D_grouped in arr2D_list_grouped:
        arr2D_grouped_binned = bin_y_axis_of_2Darray(arr=arr2D_grouped, Ny_bin=y_bin, average_sum=average_sum)
        arr2D_list_grouped_binned.append(arr2D_grouped_binned)
    
    return arr2D_list_grouped_binned

def bin_2Darray_list_and_y_axis_unc(arr2D_list, y_bin, img_bin, average_sum='average'):
    arr2D_list_grouped = bin_2Darray_list_unc(arr_list=arr2D_list, N_bin=img_bin, average_sum=average_sum)
    
    arr2D_list_grouped_binned = []
    for arr2D_grouped in arr2D_list_grouped:
        arr2D_grouped_binned = bin_y_axis_of_2Darray_unc(arr=arr2D_grouped, Ny_bin=y_bin, average_sum=average_sum)
        arr2D_list_grouped_binned.append(arr2D_grouped_binned)
    
    return arr2D_list_grouped_binned
    
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

def plot_intensitymap_NeVIII_with_contours_and_highlights_v2(intensity_map, lower_bound, upper_bound, vmin_sumer=8e-3, vmax_sumer=1e-0, title='auto', contours_highlights_both='highlights'):

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
            mlines.Line2D([],[],color='cyan', label=f'{np.round(lower_bound,4)}'),
            mlines.Line2D([],[],color='blue', label=f'{np.round(lower_bound,4)}')
        ]

    # Labels and title
    ax.set_title(f'SOHO/SUMER intensity map of Ne VIII 770.428 \u212B', fontsize=18)
    ax.set_xlabel('Wavelength dimension (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial dimension (pixels)', color='black', fontsize=16)
    ax.legend()

    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)

##################################################
# 


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



##################################################
# create intensity map from spectral images interpolated

def create_spectroheliogram_from_interpolated_data(spectralimage_interpolated_list, w_px_range, w_px_range_bckg, slope_list, intercept_list):
    c1, c2 = w_px_range
    cb1, cb2 = w_px_range_bckg
    
    spectroheliogram_line_NT, spectroheliogram_line_nobg_NT  = [],[]
    for spectrum_array in spectralimage_interpolated_list:
        
        integrated_intensity_line_1img, integrated_intensity_line_nobg_1img = [],[]
        for row in range(spectrum_array.shape[0]): 
            
            # Calculate the average of the profile of each row
            bckg_average = np.mean(spectrum_array[row, cb1:cb2+1])
            integrated_intensity_line_1row1img = slope_list[row] * np.sum(spectrum_array[row, c1:c2+1]) # Line with background (background not subtracted)
            integrated_intensity_line_nobg_1row1img = slope_list[row] * (np.sum(spectrum_array[row, c1:c2+1] - bckg_average) ) # Line without (background subtracted)
            
            # Save in lists
            integrated_intensity_line_1img.append(integrated_intensity_line_1row1img) # Line with background (background not subtracted)
            integrated_intensity_line_nobg_1img.append(integrated_intensity_line_nobg_1row1img) # Line without background (background subtracted)
        
        # save the above lists (which represent the columns) in lists to crate the 2D-array of the spectroheliogram
        spectroheliogram_line_NT.append(integrated_intensity_line_1img) # Line with background (background not subtracted)
        spectroheliogram_line_nobg_NT.append(integrated_intensity_line_nobg_1img) # Line without background (background subtracted)
        
    # Convert to array and transpose
    spectroheliogram_line = np.array(spectroheliogram_line_NT).T # Line with background (background not subtracted)
    spectroheliogram_line_nobg = np.array(spectroheliogram_line_nobg_NT).T # Line without background (background subtracted)
    
    #return [spectroheliogram_line, spectroheliogram_line_nobg]
    return spectroheliogram_line_nobg
    

##################################################
# crop array, convolution, scale

def convolution_FWHM_in_pixels(wavelength, fwhm_wavelength):
    """
    FWHM of the convolution Gaussian profile. This function calculates it in pixels of the "wavelength" array. 
    """
    w = wavelength
    pixel_scale = (w[-1] - w[0]) / (len(w)-1)
    fwhm_px = fwhm_wavelength / pixel_scale
    return fwhm_px

def crop_range(list_to_crop, range_values):
    """
    This function crops a sequence in the closest values to "range_values" and the indices of this closest value. 
    """
    # Convert sequence to Numpy array (if it isn't)
    import numpy as np
    if type(list_to_crop)==list or type(list_to_crop)==tuple: list_to_crop = np.array(list_to_crop)
    
    # Indices of the closest value to range_values[0] and range_values[1]
    idx_start = (np.abs(list_to_crop - range_values[0])).argmin() 
    idx_end = (np.abs(list_to_crop - range_values[1])).argmin()
    idx_range = [idx_start, idx_end]
    
    # Crop
    list_cropped = list_to_crop[idx_start:idx_end+1]
    
    return [list_cropped, idx_range]


def prepare_HRTS_for_subtraction(lam_hrts, rad_hrts, fwhm_conv, scale_factor):
    """
    fwhm_conv should have the same units as lam_hrts.
    """
    
    # 1) Convolve HRTS spectrum with a gaussian profile of FWHM fwhm_conv (e.g. SUMER instrumental profile)
    inst_fwhm_HRTSpx = convolution_FWHM_in_pixels(wavelength=lam_hrts, fwhm_wavelength=fwhm_conv)
    rad_hrts_conv = gaussian_filter1d(rad_hrts, sigma=inst_fwhm_HRTSpx/(2*np.sqrt(2*np.log(2)))) #without background

    # 2) Scale HRTS spectrum multiplying by a factor
    rad_hrts_conv_scaled = scale_factor * rad_hrts_conv
    
    # 3) Create the interpolation function of the entire HRTS spectrum (after convolvolution and rescaling)
    rad_hrts_conv_scaled = np.ma.filled(rad_hrts_conv_scaled, np.nan) # Convert masked values to NaNs
    interp_func_hrts = interp1d(lam_hrts, rad_hrts_conv_scaled, kind='linear', bounds_error=False, fill_value=np.nan) 

    return [interp_func_hrts, lam_hrts, rad_hrts_conv_scaled]



##################################################
# plot 2D arrays

def plot_Dopplermap(dopplermap_2Darray, x_axis='pixels', y_axis='pixels', title='auto', mark_pixel__row_col='no', color_pixel='black', set_vmin_vmax='no'):
    
    if x_axis=='pixels': x_axis = np.arange(dopplermap_2Darray.shape[1]) 
    else: x_axis = x_axis
    if y_axis=='pixels': y_axis = np.arange(dopplermap_2Darray.shape[0])
    else: y_axis = y_axis
    
    # create symetric limits of the colors to ensure that 0 values are white, negatives in blue, and positives in red
    if set_vmin_vmax=='no':
        v_max_lambda = np.nanmax(np.abs(dopplermap_2Darray)) # Find max absolute value (different from NaNs) for symmetric limits
        v_min_lambda = -v_max_lambda  # Ensure zero is white
    else: v_min_lambda, v_max_lambda = set_vmin_vmax
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    img = ax.pcolormesh(x_axis, y_axis, dopplermap_2Darray, cmap='seismic', vmin=v_min_lambda, vmax=v_max_lambda)
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
    ax.set_xlabel('helioprojective longitude (arcsec). Rot. compensated', color='black', fontsize=16)
    ax.set_ylabel('helioprojective latitude (arcsec)', color='black', fontsize=16)
    #ax.set_aspect('equal')
    if title=='auto': ax.set_title('Doppler shift of the centroid of the line', fontsize=18) 
    else: ax.set_title(title, fontsize=18) 
    # Mark the pixel selected
    if mark_pixel__row_col!='no': 
        row, col = mark_pixel__row_col
        rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
        ax.add_patch(rect)
    ax.invert_yaxis()
    plt.show(block=False)



def create_contours_range_pcolormesh_1contour(ax, image_arr, bound, x_coords, y_coords, color='yellow'):
    
    X, Y = np.meshgrid(x_coords, y_coords)  # Explicit coordinate grid
    # Plot contours with proper alignment
    contour_ = ax.contour(X, Y, image_arr, levels=[bound], colors=color, linewidths=1.5)
    ax.plot([], [], color=color, linestyle='-', linewidth=1.5, label=f'{bound}')
    return ax

    
def plot_Dopplermap_with_contours_and_pixel(dopplermap_2Darray, spectroheliogram, bound, row_col__indices, show_contours='yes', color_contours='black', show_pixel='yes', color_pixel='brown', title='auto', x_label='auto', y_label='auto', colorbar_label='auto', z_scale='log', vmin_vmax='auto', show_legend='yes'):
    
    x_axis = np.arange(dopplermap_2Darray.shape[1]) 
    y_axis = np.arange(dopplermap_2Darray.shape[0])
    
    # create symetric limits of the colors to ensure that 0 values are white, negatives in blue, and positives in red
    if vmin_vmax=='auto':
        v_max_lambda = np.nanmax(np.abs(dopplermap_2Darray)) # Find max absolute value (different from NaNs) for symmetric limits
        v_min_lambda = -v_max_lambda  # Ensure zero is white
    else: v_min_lambda, v_max_lambda = vmin_vmax
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    img = ax.pcolormesh(x_axis, y_axis, dopplermap_2Darray, cmap='seismic', vmin=v_min_lambda, vmax=v_max_lambda)
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
    ax.set_xlabel('helioprojective longitude (pixels)', color='black', fontsize=16)
    ax.set_ylabel('helioprojective latitude (pixels)', color='black', fontsize=16)
    ax.set_aspect('auto')
    if title=='auto': ax.set_title('Doppler shift of the centroid of the line', fontsize=18) 
    else: ax.set_title(title, fontsize=18) 
    # Mark the pixel selected and contours
    if show_pixel=='yes':
        row, col = row_col__indices
        rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
        ax.add_patch(rect)
    if show_contours=='yes': ax = create_contours_range_pcolormesh_1contour(ax=ax, image_arr=spectroheliogram, bound=bound, x_coords=x_axis, y_coords=y_axis, color=color_contours)
    ax.invert_yaxis()
    if show_legend=='yes': ax.legend()
    plt.show(block=False)
    

def create_contours_range_pcolormesh_1contour_nopatches(ax, mask_, color='yellow'):
    ax.contour(
    mask_.astype(int),   # convert boolean → 0/1
    levels=[0.5],           # contour boundary
    colors=color, linewidths=1.5)
    
    return ax

def plot_Dopplermap_with_contours_and_pixel_nopatches(dopplermap_2Darray, mask_, row_col__indices, show_contours='yes', color_contours='black', show_pixel='yes', color_pixel='brown', title='auto', x_label='auto', y_label='auto', colorbar_label='auto', z_scale='log', vmin_vmax='auto', show_legend='yes'):
    
    x_axis = np.arange(dopplermap_2Darray.shape[1]) 
    y_axis = np.arange(dopplermap_2Darray.shape[0])
    
    # create symetric limits of the colors to ensure that 0 values are white, negatives in blue, and positives in red
    if vmin_vmax=='auto':
        v_max_lambda = np.nanmax(np.abs(dopplermap_2Darray)) # Find max absolute value (different from NaNs) for symmetric limits
        v_min_lambda = -v_max_lambda  # Ensure zero is white
    else: v_min_lambda, v_max_lambda = vmin_vmax
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    img = ax.pcolormesh(x_axis, y_axis, dopplermap_2Darray, cmap='seismic', vmin=v_min_lambda, vmax=v_max_lambda)
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
    ax.set_xlabel('helioprojective longitude (pixels)', color='black', fontsize=16)
    ax.set_ylabel('helioprojective latitude (pixels)', color='black', fontsize=16)
    ax.set_aspect('auto')
    if title=='auto': ax.set_title('Doppler shift of the centroid of the line', fontsize=18) 
    else: ax.set_title(title, fontsize=18) 
    # Mark the pixel selected and contours
    if show_pixel=='yes':
        row, col = row_col__indices
        rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
        ax.add_patch(rect)
    if show_contours=='yes': ax = create_contours_range_pcolormesh_1contour_nopatches(ax=ax, mask_=mask_, color=color_contours)
    ax.invert_yaxis()
    if show_legend=='yes': ax.legend()
    plt.show(block=False)

    
def plot_2Darray(array_2D, x_axis='pixels', y_axis='pixels', title='auto', x_label='auto', y_label='auto', z_label='auto', cmap='Greys_r', z_scale='log', vmin_vmax='auto'):
    
    if x_axis=='pixels': x_axis = np.arange(array_2D.shape[1]) 
    else: x_axis = x_axis
    if y_axis=='pixels': y_axis = np.arange(array_2D.shape[0])
    else: y_axis = y_axis
                 
    if title=='auto': title = '' 
    else: title = title
    if x_label=='auto': x_label = 'helioprojective longitude (pixels)'
    else: x_label = x_label
    if y_label=='auto': y_label = 'helioprojective latitude (pixels)'
    else: y_label = y_label
    if z_label=='auto': z_label = ''
    else: z_label = z_label
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    if z_scale=='log': 
        if vmin_vmax=='auto': img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, norm=LogNorm())
        else: img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, norm=LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1]))
    else: 
        if vmin_vmax=='auto': img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap)
        else: img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1])
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(z_label, fontsize=16)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=18) 
    ax.set_aspect('auto') #'auto', 'equal'
    ax.invert_yaxis()
    plt.show(block=False)


def plot_2Darray_with_contours_and_pixel(array_2D, spectroheliogram, bound, row_col__indices, show_contours='yes', color_contours='black', show_pixel='yes', color_pixel='brown', title='auto', x_label='auto', y_label='auto', colorbar_label='auto', cmap='Greys_r', z_scale='log', vmin_vmax='auto', show_legend='yes'):
    
    x_axis = np.arange(array_2D.shape[1]) 
    y_axis = np.arange(array_2D.shape[0])
    
    if title=='auto': title = '' 
    else: title = title
    if x_label=='auto': x_label = 'helioprojective longitude (pixels)'
    else: x_label = x_label
    if y_label=='auto': y_label = 'helioprojective latitude (pixels)'
    else: y_label = y_label
    if colorbar_label=='auto': colorbar_label = ''
    else: colorbar_label = colorbar_label
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    
    if z_scale=='log': 
        if vmin_vmax=='auto': img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, norm=LogNorm())
        else: img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, norm=LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1]))
    else: 
        if vmin_vmax=='auto': img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap)
        else: img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(colorbar_label, fontsize=16)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=18) 
    ax.set_aspect('auto') #'auto', 'equal'
    ax.invert_yaxis()
    # Mark the pixel selected and contours
    if show_pixel=='yes':
        row, col = row_col__indices
        rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
        ax.add_patch(rect)
    if show_contours=='yes': ax = create_contours_range_pcolormesh_1contour(ax=ax, image_arr=spectroheliogram, bound=bound, x_coords=x_axis, y_coords=y_axis, color=color_contours)
    ax.set_aspect('auto') #'auto', 'equal'
    if show_legend=='yes': ax.legend()
    plt.show(block=False)



def plot_2Darray_with_contours_and_pixel_nopatches(array_2D, mask_, row_col__indices, show_contours='yes', color_contours='black', show_pixel='yes', color_pixel='brown', title='auto', x_label='auto', y_label='auto', colorbar_label='auto', cmap='Greys_r', z_scale='log', vmin_vmax='auto', show_legend='yes'):
    
    x_axis = np.arange(array_2D.shape[1]) 
    y_axis = np.arange(array_2D.shape[0])
    
    if title=='auto': title = '' 
    else: title = title
    if x_label=='auto': x_label = 'helioprojective longitude (pixels)'
    else: x_label = x_label
    if y_label=='auto': y_label = 'helioprojective latitude (pixels)'
    else: y_label = y_label
    if colorbar_label=='auto': colorbar_label = ''
    else: colorbar_label = colorbar_label
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    
    if z_scale=='log': 
        if vmin_vmax=='auto': img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, norm=LogNorm())
        else: img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, norm=LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1]))
    else: 
        if vmin_vmax=='auto': img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap)
        else: img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(colorbar_label, fontsize=16)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=18) 
    ax.set_aspect('auto') #'auto', 'equal'
    ax.invert_yaxis()
    # Mark the pixel selected and contours
    if show_pixel=='yes':
        row, col = row_col__indices
        rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
        ax.add_patch(rect)
    if show_contours=='yes': ax = create_contours_range_pcolormesh_1contour_nopatches(ax=ax, mask_=mask_, color=color_contours)
    ax.set_aspect('auto') #'auto', 'equal'
    if show_legend=='yes': ax.legend()
    plt.show(block=False)


##################################################
# plot 2 contours

def create_contours_range_pcolormesh_2contour(ax, image_arr, bound, x_coords, y_coords, color='yellow', label='line name'):
    
    X, Y = np.meshgrid(x_coords, y_coords)  # Explicit coordinate grid
    # Plot contours with proper alignment
    contour_ = ax.contour(X, Y, image_arr, levels=[bound], colors=color, linewidths=1.5)
    ax.plot([], [], color=color, linestyle='-', linewidth=1.5, label=f'{label} {np.round(bound,4)}')
    return ax



def plot_2Darray_with_contours_and_pixel_2contour(array_2D, spectroheliogram1, bound1, label1, spectroheliogram2, bound2, label2, row_col__indices, show_contours1='yes', show_contours2='yes', color_contours1='black', color_contours2='green', show_pixel='yes', color_pixel='brown', title='auto', x_label='auto', y_label='auto', colorbar_label='auto', cmap='Greys_r', z_scale='log', vmin_vmax='auto', show_legend='yes'):
    
    x_axis = np.arange(array_2D.shape[1]) 
    y_axis = np.arange(array_2D.shape[0])
    
    if title=='auto': title = '' 
    else: title = title
    if x_label=='auto': x_label = 'helioprojective longitude (pixels)'
    else: x_label = x_label
    if y_label=='auto': y_label = 'helioprojective latitude (pixels)'
    else: y_label = y_label
    if colorbar_label=='auto': colorbar_label = ''
    else: colorbar_label = colorbar_label
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    
    if z_scale=='log': 
        if vmin_vmax=='auto': img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, norm=LogNorm())
        else: img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, norm=LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1]))
    else: 
        if vmin_vmax=='auto': img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap)
        else: img=ax.pcolormesh(x_axis, y_axis, array_2D, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(colorbar_label, fontsize=16)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=18) 
    ax.set_aspect('auto') #'auto', 'equal'
    ax.invert_yaxis()
    # Mark the pixel selected and contours
    if show_pixel=='yes':
        row, col = row_col__indices
        rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
        ax.add_patch(rect)
    if show_contours1=='yes': ax = create_contours_range_pcolormesh_2contour(ax=ax, image_arr=spectroheliogram1, bound=bound1, x_coords=x_axis, y_coords=y_axis, color=color_contours1, label=label1)
    if show_contours2=='yes': ax = create_contours_range_pcolormesh_2contour(ax=ax, image_arr=spectroheliogram2, bound=bound2, x_coords=x_axis, y_coords=y_axis, color=color_contours2, label=label2)
    ax.set_aspect('auto') #'auto', 'equal'
    if show_legend=='yes': ax.legend()
    plt.show(block=False)


def plot_Dopplermap_with_contours_and_pixel_2contour(dopplermap_2Darray, spectroheliogram1, bound1, label1, spectroheliogram2, bound2, label2, row_col__indices, show_contours1='yes', show_contours2='yes', color_contours1='black', color_contours2='green', show_pixel='yes', color_pixel='brown', title='auto', x_label='auto', y_label='auto', colorbar_label='auto', z_scale='log', vmin_vmax='auto', show_legend='yes'):
    
    x_axis = np.arange(dopplermap_2Darray.shape[1]) 
    y_axis = np.arange(dopplermap_2Darray.shape[0])
    
    # create symetric limits of the colors to ensure that 0 values are white, negatives in blue, and positives in red
    if vmin_vmax=='auto':
        v_max_lambda = np.nanmax(np.abs(dopplermap_2Darray)) # Find max absolute value (different from NaNs) for symmetric limits
        v_min_lambda = -v_max_lambda  # Ensure zero is white
    else: v_min_lambda, v_max_lambda = vmin_vmax
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    img = ax.pcolormesh(x_axis, y_axis, dopplermap_2Darray, cmap='seismic', vmin=v_min_lambda, vmax=v_max_lambda)
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
    ax.set_xlabel('helioprojective longitude (pixels)', color='black', fontsize=16)
    ax.set_ylabel('helioprojective latitude (pixels)', color='black', fontsize=16)
    ax.set_aspect('auto')
    if title=='auto': ax.set_title('Doppler shift of the centroid of the line', fontsize=18) 
    else: ax.set_title(title, fontsize=18) 
    # Mark the pixel selected and contours
    if show_pixel=='yes':
        row, col = row_col__indices
        rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
        ax.add_patch(rect)
    if show_contours1=='yes': ax = create_contours_range_pcolormesh_2contour(ax=ax, image_arr=spectroheliogram1, bound=bound1, x_coords=x_axis, y_coords=y_axis, color=color_contours1, label=label1)
    if show_contours2=='yes': ax = create_contours_range_pcolormesh_2contour(ax=ax, image_arr=spectroheliogram2, bound=bound2, x_coords=x_axis, y_coords=y_axis, color=color_contours2, label=label2)
    ax.invert_yaxis()
    if show_legend=='yes': ax.legend()
    plt.show(block=False)
    


##################################################
# single gaussian fit

def estimate_guess_parameters_for_a_simple_gaussian_fit(x, y):
    """
    Function that estimates the parameters [background, peak, mean, FWHM] of a possible gaussian that a set of data follows. 
    These estimated parameters will be used later as guess parameters (initial parameters) for a simple gaussian fit. 
    """
    # 1) Guess the mean, peak, and background
    idx_max = np.argmax(y)
    guess_mean = x[idx_max]
    guess_peak = y[idx_max]
    guess_bckg = np.min(y)
    
    # 2) Guess the FWHM
    half_max = guess_bckg + (guess_peak - guess_bckg) / 2 #Calculate half maximum 
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        return None  # Not enough data above half max
    guess_fwhm = x[indices[-1]] - x[indices[0]]

    return [guess_bckg, guess_peak, guess_mean, guess_fwhm]

def gaussian_with_background(x, background, peak, mean, fwhm):
    """
    Formula of a Gaussian, including background level. 
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return background + peak * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def gaussian_with_background_uncertainty(x, x_unc, peak, peak_unc, mean, mean_unc, fwhm, fwhm_unc, background_unc):
    """
    Calculates uncertainty of a single Gaussian + background. It uses error propagation over the Gaussian formula.
    """
    I, dI = peak, peak_unc
    m, dm = mean, mean_unc
    w, dw = fwhm, fwhm_unc
    dx = x_unc
    db = background_unc

    ln2 = np.log(2)
    arg = -4 * ln2 * ((x - m) ** 2) / (w ** 2)
    TE = np.exp(arg)
    
    df_dx = I * TE * (8 * ln2 * (x - m) / w**2)
    df_dI = TE
    df_dm = I * TE * (8 * ln2 * (x - m) / w**2)
    df_dw = I * TE * (8 * ln2 * (x - m)**2 / w**3)
    
    total_uncertainty = np.sqrt(
        (df_dx * dx)**2 +
        (df_dI * dI)**2 +
        (df_dm * dm)**2 +
        (df_dw * dw)**2 +
        db**2
    )
    
    return total_uncertainty


def fit_gaussian_with_background(x_data, y_data, y_unc_data, guess_bckg, guess_peak, guess_mean, guess_fwhm): 
    """
    Function that fits a simple gaussian to a set of data. It uses scipy.optimize.curve_fit(). 
    """
    
    guess_parameters = [guess_bckg, guess_peak, guess_mean, guess_fwhm] #background, amplitude, mean, FWHM
    
    try:
        # Do the fit (sigma is uncertainty in y_data)
        popt, pcov = curve_fit(
            gaussian_with_background,
            x_data,
            y_data,
            sigma=y_unc_data,
            p0=guess_parameters,
            absolute_sigma=True  # Use this to scale errors properly
        )

        perr = np.sqrt(np.diag(pcov))  # standard deviation errors on parameters
        
        
        # --- compute reduced chi-square ---
        residuals = y_data - gaussian_with_background(x_data, *popt)
        chi_square = np.sum((residuals / y_unc_data)**2)
        dof = len(y_data) - len(popt)   # degrees of freedom = N_data − N_parameters
        reduced_chi_square = chi_square / dof
        

        fit_results = {}
        param_names = ['background', 'peak', 'mean', 'FWHM']
        for name, val, err in zip(param_names, popt, perr):
            fit_results[name] = [val, err]
        fit_results['chi2red'] = reduced_chi_square

        return fit_results
    
    except RuntimeError:
        # Fit failed — return None
        return None


def fit_automatic_gaussian_with_background(x_data, y_data, y_unc_data):
    """
    Function that fit a Gaussian function over a set of data using scipy.optimize.curve_fit(), but, previously has calculated the initial parameters. 
    So this is a more automatic function to do a Gaussian fit. 
    """
    egp = estimate_guess_parameters_for_a_simple_gaussian_fit(x=x_data, y=y_data)
    if egp!=None:
        guess_bckg, guess_peak, guess_mean, guess_fwhm = egp
        fit_results = fit_gaussian_with_background(x_data=x_data, y_data=y_data, y_unc_data=y_unc_data, guess_bckg=guess_bckg, guess_peak=guess_peak, guess_mean=guess_mean, guess_fwhm=guess_fwhm)
        return fit_results
    else: 
        return None

##################################################
# 

def plot_one_fit_of_the_Dopplergram(spectralimage_index, row_index, wavelength_range_preliminary, rest_wavelength, spectral_image_list, unc_spectral_image_list, centroid_gaussian_or_parabola='gaussian'):
    
    sp_j = spectralimage_index
    row_i = row_index
    
    # Columns where to cut
    wavelength_range_pixcenter, pixel_range_pixcenter, pixel_range_float = range__wavelength_to_closest_pixels(wavelength_range=wavelength_range_preliminary, slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
    col1, col2 = pixel_range_pixcenter

    # Take the data of the current spectral image and row
    x_wavelength_1row = pixels_to_wavelength(pixel=np.arange(col1, col2+1), slope_cal=slope_list[row_i], intercept_cal=intercept_list[row_i])
    x_data = vkms_doppler(lamb=x_wavelength_1row, lamb_0=rest_wavelength)
    y_data = spectral_image_list[sp_j][row_i, col1:col2+1]
    y_unc_data = unc_spectral_image_list[sp_j][row_i, col1:col2+1]
            
    if centroid_gaussian_or_parabola == 'gaussian':
        # Fit a single gaussian (and background) to the data
        fit_results = fit_automatic_gaussian_with_background(x_data=x_data, y_data=y_data, y_unc_data=y_unc_data)
        g_background, dg_background = fit_results['background']
        g_peak, dg_peak = fit_results['peak']
        g_mean, dg_mean = fit_results['mean']
        g_fwhm, dg_fwhm = fit_results['FWHM']
        
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 500)
        y_fit = gaussian_with_background(x=x_fit, background=g_background, peak=g_peak, mean=g_mean, fwhm=g_fwhm)
        x_centroid = g_mean
        
    elif centroid_gaussian_or_parabola == 'parabola':
        # Fit a parabola to the highest 3 data points
        idx_max = np.argmax(y_data) #Index of the maximum of the data
        x_nearmax = x_data[[idx_max-1, idx_max, idx_max+1]]
        y_nearmax = y_data[[idx_max-1, idx_max, idx_max+1]]
        x_vertex, y_vertex, a, b, c = find_parabolic_coefficients_and_vertex(x=x_nearmax, y=y_nearmax)
        x_centroid = x_vertex
        x_fit = np.linspace(np.min(x_nearmax), np.max(x_nearmax), 200)
        y_fit = a*x_fit**2 + b*x_fit + c
        
    else: print('Variable centroid_gaussian_or_parabola should be the string "gaussian" or "parabola"')
    
    
    if centroid_gaussian_or_parabola == 'gaussian':
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2.5,1]})
        ax[0].errorbar(x=x_data, y=y_data, yerr=y_unc_data, color='black', linewidth=0, elinewidth=0.5, marker='.', markersize=5, label='data')
        ax[0].plot(x_fit, y_fit, 'r-', label='Fit')
        ax[0].set_title(f'Single gaussian (automatic) fit', fontsize=18) 
        ax[0].set_xlabel('Doppler velocity (km/s)', color='black', fontsize=16)
        ax[0].set_ylabel(f'Intensity [W/sr/m^2/Angstroem]', color='black', fontsize=16)
        ax[0].axvline(x=g_mean, linestyle='--', linewidth=1.5, color='blue', label=f'mean: {np.round(g_mean,3)}'r'$\pm$'f'{np.round(dg_mean,3)}')
        ax[0].axvspan(g_mean-dg_mean/2, g_mean+dg_mean/2, color='blue', linestyle=':', alpha=0.15)#, label='Blue range\n'f'{range_B[0]} - {range_B[1]} {xunits}')
        ax[0].axhline(y=g_background, linestyle=':', linewidth=1.5, color='green', label=f'background: {np.round(g_background,3)}'r'$\pm$'f'{np.round(dg_background,3)}')
        ax[0].axhspan(g_background-dg_background/2, g_background+dg_background/2, color='green', linestyle=':', alpha=0.15)#, label='Blue range\n'f'{range_B[0]} - {range_B[1]} {xunits}')
        ax[0].legend()
        
        # Lower panel: residuals
        y_fit_length_data = gaussian_with_background(x=x_data, background=g_background, peak=g_peak, mean=g_mean, fwhm=g_fwhm)
        y_unc_fit_length_data = gaussian_with_background_uncertainty(x=x_data, x_unc=np.zeros(len(x_data)), background_unc=dg_background, peak=g_peak, peak_unc=dg_peak, mean=g_mean, mean_unc=dg_mean, fwhm=g_fwhm, fwhm_unc=dg_fwhm)
        y_residuals = y_data - y_fit_length_data
        y_unc_residuals = np.sqrt(y_unc_data**2 + y_unc_fit_length_data**2)
        ax[1].errorbar(x=x_data, y=y_residuals, yerr=y_unc_residuals, color='black', linewidth=1, marker='o', label='residuals')
        #ax[1].set_xlabel('Wavelength direction [pixels]')
        ax[1].set_xlabel('Doppler velocity (km/s)')
        ax[1].set_ylabel(r'Residuals ($y_{\rm data} - y_{\rm fit}$)')
        ax[1].axhline(y=0, color='grey', linestyle='--', linewidth=1, label='y=0')
        ylimres_ = max(abs(min(y_residuals)), abs(max(y_residuals)))
        ylimres = 1.1*ylimres_
        ax[1].set_ylim([-ylimres, ylimres])
        ax[1].legend()

        plt.subplots_adjust(left=0.05, right=0.8, bottom=0.05, top=0.95, wspace=0, hspace=0)
        plt.show(block=False)
    
    elif centroid_gaussian_or_parabola == 'parabola':
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
        ax.errorbar(x=x_data, y=y_data, yerr=y_unc_data, color='black', linewidth=0, elinewidth=0.5, marker='.', markersize=5, label='data')
        ax.plot(x_fit, y_fit, 'r-', label='Fit')
        ax.set_title(f'Parabolic (automatic) fit', fontsize=18) 
        ax.set_xlabel('Doppler velocity (km/s)', color='black', fontsize=16)
        ax.set_ylabel(f'Intensity [W/sr/m^2/Angstroem]', color='black', fontsize=16)
        ax.axvline(x=x_centroid, linestyle='--', linewidth=1.5, color='blue', label=f'mean: {np.round(x_centroid,3)}')
        ax.legend()
        
        plt.subplots_adjust(left=0.05, right=0.8, bottom=0.05, top=0.95, wspace=0, hspace=0)
        plt.show(block=False)
    
    return [x_centroid, x_fit, y_fit, x_data, y_data, y_unc_data]


##################################################
# double gaussian fit

def double_gaussian_with_background(x, background, peak1, mean1, fwhm1, peak2, mean2, fwhm2):
    """
    Formula of a double Gaussian, including background level. 
    """
    sigma1 = fwhm1 / (2 * np.sqrt(2 * np.log(2)))
    sigma2 = fwhm2 / (2 * np.sqrt(2 * np.log(2)))
    gaussian_1_ = peak1 * np.exp(-((x - mean1) ** 2) / (2 * sigma1 ** 2))
    gaussian_2_ = peak2 * np.exp(-((x - mean2) ** 2) / (2 * sigma2 ** 2))
    return background + gaussian_1_ + gaussian_2_


def fit_double_gaussian_with_background( x_data, y_data, y_unc_data, guess_background, guess_peak1, guess_mean1, guess_fwhm1, guess_peak2, guess_mean2, guess_fwhm2):
    """
    Fit a double Gaussian with background using scipy.optimize.curve_fit.
    """
    
    # Initial parameter guesses
    guess_parameters = [
        guess_background,
        guess_peak1, guess_mean1, guess_fwhm1,
        guess_peak2, guess_mean2, guess_fwhm2
    ]
    
    try:
        popt, pcov = curve_fit( double_gaussian_with_background, x_data, y_data, sigma=y_unc_data, p0=guess_parameters, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))

        # Compute reduced chi-square
        residuals = y_data - double_gaussian_with_background(x_data, *popt)
        chi_square = np.sum((residuals / y_unc_data)**2)
        dof = len(y_data) - len(popt)
        reduced_chi_square = chi_square / dof

        # Pack results
        fit_results = {}
        param_names = [
            'background',
            'peak1', 'mean1', 'fwhm1',
            'peak2', 'mean2', 'fwhm2'
        ]

        for name, val, err in zip(param_names, popt, perr):
            fit_results[name] = [val, err]

        fit_results['chi2red'] = reduced_chi_square
        
        return fit_results

    except RuntimeError:
        return None


def fit_automatic_double_gaussian_with_background(x_data, y_data, y_unc_data):
    """
    Function that fit a double Gaussian function over a set of data using scipy.optimize.curve_fit(), but, previously has calculated the initial parameters. 
    So this is a more automatic function to do a Gaussian fit. 
    """
    egp = estimate_guess_parameters_for_a_simple_gaussian_fit(x=x_data, y=y_data)
    if egp!=None:
        guess_bckg, guess_peak, guess_mean, guess_fwhm = egp
        guess_mean_1, guess_mean_2 = 1540.76, 1540.93
        guess_peak_1, guess_peak_2 = 0.8*guess_peak, 0.8*guess_peak # 0.8 for example
        guess_fwhm_1, guess_fwhm_2 = 0.5*guess_fwhm, 0.5*guess_fwhm
        
        fit_results = fit_double_gaussian_with_background(x_data=x_data, y_data=y_data, y_unc_data=y_unc_data, guess_background=guess_bckg, guess_peak1=guess_peak_1, guess_mean1=guess_mean_1, guess_fwhm1=guess_fwhm_1, guess_peak2=guess_peak_2, guess_mean2=guess_mean_2, guess_fwhm2=guess_fwhm_2)
        return fit_results
    else: 
        return None



##################################################
# create 2D-array of scaling factors (SUMER / HRTS)

def get_factor_SUMER_HRTS(lam_sumer, spectralimage_interp_list_, unc_spectralimage_interp_list_, lam_hrts, rad_hrts, fwhm_conv, wavelength_range_preliminary, show__row_col='no', y_scale='linear', show_legend='yes'):
    
    N_img = len(spectralimage_interp_list_)
    N_rows = spectralimage_interp_list_[0].shape[0]
    #N_cols_full = spectralimage_interp_list_[0].shape[1]
    
    # 1) Crop SUMER wavelength array in the range selected in the input
    lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_preliminary)
    col1, col2 = idx_sumer_crop
    
    # 2) Convolve HRTS spectrum with SUMER instrumental profile (gaussian profile of FWHM fwhm_conv)
    inst_fwhm_HRTSpx = convolution_FWHM_in_pixels(wavelength=lam_hrts, fwhm_wavelength=fwhm_conv)
    rad_hrts_conv = gaussian_filter1d(rad_hrts, sigma=inst_fwhm_HRTSpx/(2*np.sqrt(2*np.log(2)))) #without background

    # 3) Create the interpolation function of the entire HRTS spectrum (after convolvolution)
    rad_hrts_conv = np.ma.filled(rad_hrts_conv, np.nan) # Convert masked values to NaNs
    interp_func_hrts = interp1d(lam_hrts, rad_hrts_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
    
    # 4) Get data of HRTS corresponding to SUMER wavelengths (of the cropped range)
    rad_hrts_SUMERgrid = interp_func_hrts(lam_sumer_crop)
    
    # 5) Crop HRTS
    lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrts, range_values=wavelength_range_preliminary)
    rad_hrts_conv_crop = rad_hrts_conv[idx_hrts_crop[0]:idx_hrts_crop[1]+1]
    
    scaling_factor_NT, chi2red_NT = [],[]
    for img_ in tqdm(range(N_img)):
        scaling_factor_1sp_allrows, chi2red_1sp_allrows = [], []
        for row_ in range(N_rows):
            
            # 6) Take the data of the current spectral image and row (of SUMER)
            lam_sumer = lam_sumer
            rad_sumer = spectralimage_interp_list_[img_][row_, :]
            erad_sumer = unc_spectralimage_interp_list_[img_][row_, :]

            # 7) Crop SUMER in the wavelength range
            lam_sumer_crop = lam_sumer[col1:col2+1]
            rad_sumer_crop = rad_sumer[col1:col2+1]
            erad_sumer_crop = erad_sumer[col1:col2+1]

            # 8) Fit straight line to the radiance of HRTS vs SUMER (y = m*x + 0 (intercept = 0))
            y_hrts = rad_hrts_SUMERgrid
            y_sumer = rad_sumer_crop
            yerr_sumer = erad_sumer_crop
            weights = 1 / yerr_sumer**2
            scaling_factor_ = np.sum(weights * y_hrts * y_sumer) / np.sum(weights * y_hrts**2) #this is the slope and it is the scaling factor

            # 9) Compute chi^2
            chi2 = np.sum(((y_sumer - scaling_factor_ * y_hrts)**2) / (yerr_sumer**2))
            dof = len(y_sumer) - 1 # Degrees of freedom (N - number_of_parameters = N - 1)
            chi2red_ = chi2 / dof #reduced chi^2
            
            
            ####################################
            # 10) Show profile
            if show__row_col!='no' and show__row_col[0]==row_ and show__row_col[1]==img_:
                color_sumer = 'red'
                color_hrts = 'blue'

                # Generate fitted line for plotting
                xfit = np.linspace(min(rad_hrts_SUMERgrid), max(rad_hrts_SUMERgrid), 1000)
                yfit = scaling_factor_ * xfit

                # Show profiles and region cropped
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
                ax.errorbar(x=lam_sumer, y=rad_sumer, yerr=erad_sumer, color=color_sumer, linewidth=0, marker='.', markersize=10, label='SUMER original')
                ax.errorbar(x=lam_hrts, y=rad_hrts_conv, color=color_hrts, linewidth=0.5, label=f'HRTS convolved (FWHM: {fwhm_conv} ''\u212B)')
                ax.set_yscale(y_scale)
                ax.axvspan(wavelength_range_preliminary[0], wavelength_range_preliminary[1], color='orange', alpha=0.15, label='Region to analyze')
                ax.set_title(f'SUMER and HRTS spectra. Wavelength range used to calculate the scaling factor. Indices: Spectral image {img_}, latitude {row_}', fontsize=18) 
                ax.set_xlabel('Wavelength direction (\u212B)', color='black', fontsize=16)
                ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', color='black', fontsize=16)
                if show_legend=='yes': ax.legend(fontsize=10)
                plt.show(block=False)
                
                # ) Show radiances of SUMER and HRTS
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
                ax.errorbar(x=rad_hrts_SUMERgrid, y=rad_sumer_crop, yerr=erad_sumer_crop, color='black', linewidth=0, elinewidth=1., marker='.', markersize=10, label='Data')
                ax.plot(xfit, yfit, color='green', label=f'Linear fit. Slope (scaling factor): {np.round(scaling_factor_,4)}')
                ax.set_title(f'Radiances of SUMER vs HRTS, and fitting.', fontsize=18) 
                ax.set_xlabel(r'Radiance HRTS [W/sr/m$^2$/''\u212B]', color=color_hrts, fontsize=16)
                ax.set_ylabel(r'Radiance SUMER [W/sr/m$^2$/''\u212B]', color=color_sumer, fontsize=16)
                # Change tick label colors
                ax.tick_params(axis='x', colors=color_hrts)
                ax.tick_params(axis='y', colors=color_sumer)
                # ---- Make axes square with same limits ----
                min_val = min(rad_sumer_crop.min(), rad_hrts_SUMERgrid.min())
                max_val = max(rad_sumer_crop.max(), rad_hrts_SUMERgrid.max())
                delta_extremes = 0.05 * (max_val - min_val)
                ax.set_xlim(min_val-delta_extremes, max_val+delta_extremes)
                ax.set_ylim(min_val-delta_extremes, max_val+delta_extremes)
                print('###############')
                print(min_val)
                print(max_val)
                print(min_val-delta_extremes, max_val+delta_extremes)
                ax.set_aspect('equal', adjustable='box')
                if show_legend=='yes': ax.legend(fontsize=10)
                plt.show(block=False)

                # ) Show profiles cropped
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
                ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop, yerr=erad_sumer_crop, color=color_sumer, linewidth=0.5, label='SUMER original, region to analyze')
                ax.errorbar(x=lam_hrts, y=rad_hrts_conv, color=color_hrts, linewidth=0.5, label=f'HRTS convolved (FWHM: {fwhm_conv} ''\u212B)')
                ax.errorbar(x=lam_sumer_crop, y=rad_hrts_SUMERgrid, color=color_hrts, linewidth=0, marker='.', markersize=7, label='Interpolated HRTS addapted to SUMER grid')
                ax.set_yscale(y_scale)
                ax.axvspan(wavelength_range_preliminary[0], wavelength_range_preliminary[1], color='orange', alpha=0.15, label='Region to analyze')
                ax.set_title(f'SUMER and HRTS spectra in the wavelength range used to calculate the scaling factor. Indices: Spectral image {img_}, latitude {row_}', fontsize=18) 
                ax.set_xlabel('Wavelength direction (\u212B)', color='black', fontsize=16)
                ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', color='black', fontsize=16)
                if show_legend=='yes': ax.legend(fontsize=10)
                plt.show(block=False)
                
                # ) Show profiles cropped with HRTS scaled
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
                ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop, yerr=erad_sumer_crop, color=color_sumer, linewidth=0, marker='.', markersize=10, label='SUMER original')
                ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop, yerr=erad_sumer_crop, color=color_sumer, linewidth=0.5, label='SUMER original')
                ax.errorbar(x=lam_hrts_crop, y=scaling_factor_*rad_hrts_conv_crop, color=color_hrts, linewidth=0.5, label=f'HRTS convolved')
                ax.errorbar(x=lam_sumer_crop, y=scaling_factor_*rad_hrts_SUMERgrid, color=color_hrts, linewidth=0, marker='.', markersize=7, label='Interpolated SUMER wavelength data')
                ax.set_yscale(y_scale)
                #ax.set_xlim([min(lam_sumer_crop), max(lam_sumer_crop)])
                ax.set_title(f'HRTS scaled', fontsize=18) 
                ax.set_xlabel('Wavelength direction (\u212B)', color='black', fontsize=16)
                ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', color='black', fontsize=16)
                if show_legend=='yes': ax.legend(fontsize=10)
                plt.show(block=False)
            
            ####################################
            
            # 10) Save scaling factor and reduced chi^2 each in a list
            scaling_factor_1sp_allrows.append(scaling_factor_)
            chi2red_1sp_allrows.append(chi2red_)

        # 11) Save the above lists (which represent the different spectral images) in lists to crate the 2D-array
        scaling_factor_NT.append(scaling_factor_1sp_allrows)
        chi2red_NT.append(chi2red_1sp_allrows)

    # 12) Convert to array and transpose
    scaling_factor_map = np.array(scaling_factor_NT).T 
    chi2red_map = np.array(chi2red_NT).T 

    return [scaling_factor_map, chi2red_map]




##################################################
# create Dopplermap

def create_SUMER_dopplermap_single_gaussianfit(lam_sumer, spectralimage_interp_list_, unc_spectralimage_interp_list_, wavelength_range_preliminary, rest_wavelength, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', show__row_col='no', y_scale='linear', show_legend='yes'):
    
    # 1) Crop SUMER wavelength array in the range selected in the input
    lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_preliminary)
    col1, col2 = idx_sumer_crop    
    
    N_img = len(spectralimage_interp_list_)
    N_rows = spectralimage_interp_list_[0].shape[0]
    #N_cols_full = spectralimage_interp_list_[0].shape[1]
    #N_cols = col2-col1+1
    dopplershift_NT, unc_dopplershift_NT, chi2red_NT = [],[],[]
    for img_ in tqdm(range(N_img)):
        dopplershift_1sp_allrows, unc_dopplershift_1sp_allrows, chi2red_1sp_allrows = [], [], []
        for row_ in range(N_rows):
                        
            # 2) Take the data of the current spectral image and row
            lam_sumer = lam_sumer #or lam_sumer[col1:col2+1]
            rad_sumer = spectralimage_interp_list_[img_][row_,:]
            erad_sumer = unc_spectralimage_interp_list_[img_][row_,:]
            
            # 3) Crop SUMER data
            lam_sumer_crop = lam_sumer_crop #or lam_sumer[col1:col2+1]
            rad_sumer_crop = rad_sumer[col1:col2+1]
            erad_sumer_crop = erad_sumer[col1:col2+1]
            
            if subtract_HRTS!='no':
                # 4) Prepare HRTS data for the subtraction
                scale_factor_hrts = scalefactor_hrts_2Darr[row_, img_]
                interp_func_hrts, lam_hrts, rad_hrts_conv_scaled = prepare_HRTS_for_subtraction(lam_hrts=lam_hrts, rad_hrts=rad_hrts, fwhm_conv=fwhm_conv, scale_factor=scale_factor_hrts)

                # 5) Get data of HRTS corresponding to SUMER's wavelength grid (of the cropped range)
                rad_hrts_SUMERgrid = interp_func_hrts(lam_sumer_crop)
                
                # 6) Crop HRTS
                lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrts, range_values=wavelength_range_preliminary)
                rad_hrts_conv_scaled_crop = rad_hrts_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]

                # 7) Subtract HRTS to SUMER
                rad_sumer_crop_tofit = rad_sumer_crop - rad_hrts_SUMERgrid
                erad_sumer_crop_tofit = erad_sumer_crop #TODO: not considering HRTS erorbars (wich we don't know)
            
            else: 
                rad_sumer_crop_tofit = rad_sumer_crop
                erad_sumer_crop_tofit = erad_sumer_crop
            
            # 8) Fit a single gaussian (and background) to the data. 
            res = fit_automatic_gaussian_with_background(x_data=lam_sumer_crop, y_data=rad_sumer_crop_tofit, y_unc_data=erad_sumer_crop_tofit)
            if res!=None: 
                x_centroid = res['mean'][0]
                v_centroid = vkms_doppler(lamb=x_centroid, lamb_0=rest_wavelength)
                #dv_centroid = vkms_doppler_unc(lamb=res['mean'][1], lamb_unc=dx_centroid, lamb_0=rest_wavelength, lamb_0_unc=0.0)
                chi2red_ = res['chi2red']
            else:
                v_centroid = np.nan
                chi2red_ = np.nan
            
            # 9) Show profile
            if show__row_col!='no' and show__row_col[0]==row_ and show__row_col[1]==img_:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
                ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop, yerr=erad_sumer_crop, color='black', linewidth=0.8, label='SUMER original')
                ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop_tofit, yerr=erad_sumer_crop_tofit, color='blue', linewidth=0, elinewidth=1., marker='.', label='Data to fit')
                if subtract_HRTS!='no':
                    ax.errorbar(x=lam_hrts_crop, y=rad_hrts_conv_scaled_crop, color='brown', linewidth=0.8, label=f'HRTS convolved and scaled (factor {scale_factor_hrts})')
                    ax.errorbar(x=lam_sumer_crop, y=rad_hrts_SUMERgrid, color='black', linewidth=0, marker='.', markersize=7, label='Interpolated SUMER wavelength data')
                x_gauss_fit = np.linspace(lam_sumer_crop[0], lam_sumer_crop[-1], 3000)
                y_gauss_fit = gaussian_with_background(x=x_gauss_fit, background=res['background'][0], peak=res['peak'][0], mean=res['mean'][0], fwhm=res['FWHM'][0])
                ax.errorbar(x=x_gauss_fit, y=y_gauss_fit, color='red', linewidth=1.5, label='Fitted Gaussian')
                ax.axvline(x=rest_wavelength, linestyle='--', color='green', linewidth=1.5, label=f'Rest wavelength {rest_wavelength}'' \u212B')
                ax.set_yscale(y_scale)
                ax.set_title(f'SOHO/SUMER profile analysis. Spectral image {img_}, lattitude row {row_}', fontsize=18) 
                ax.set_xlabel('Wavelength direction (\u212B)', color='black', fontsize=16)
                ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', color='black', fontsize=16)
                # Add secondary x-axis with the Doppler velocity
                def x_old_to_new(lam__): #Wavelength (delta) to doppler velocity. 
                    c = 299792.4580 #[km/s] speed of light
                    return c*(lam__-rest_wavelength)/rest_wavelength
                def x_new_to_old(v_): #Doppler velocity to wavelength.
                    c = 299792.4580 #[km/s] speed of light
                    return rest_wavelength*((v_/c)+1)
                secax = ax.secondary_xaxis('top', functions=(lambda lam__: x_old_to_new(lam__), lambda v_: x_new_to_old(v_)))
                secax.set_xlabel("Doppler velocity (km/s)", fontsize=14)
                dopplershift__ = x_old_to_new(lam__=x_centroid)
                if res!=None: y_centroid = res['background'][0] + res['peak'][0]
                ax.scatter(x_centroid, y_centroid, color='red', s=30, marker='^', label=f'Centroid of the Gaussian: {dopplershift__} km/s')
                if show_legend=='yes': ax.legend(fontsize=10)
                plt.show(block=False)
            
            # 10) Save centroid of the Gaussian in a list
            dopplershift_1sp_allrows.append(v_centroid)
            #unc_dopplershift_1sp_allrows.append(dv_centroid)
            chi2red_1sp_allrows.append(chi2red_)
        
        # 11) Save the above lists (which represent the different spectral images) in lists to crate the 2D-array
        dopplershift_NT.append(dopplershift_1sp_allrows)
        #unc_dopplershift_NT.append(unc_dopplershift_1sp_allrows)
        chi2red_NT.append(chi2red_1sp_allrows)
    
    # 12) Convert to array and transpose
    dopplershift_map = np.array(dopplershift_NT).T 
    #unc_dopplershift_map = np.array(unc_dopplershift_NT).T # Uncertainty
    chi2red_map = np.array(chi2red_NT).T
    
    return [dopplershift_map, chi2red_map]


def create_SUMER_dopplermap_single_gaussianfit__1pixel_profile(lam_sumer, spectralimage_interp_list_, unc_spectralimage_interp_list_, wavelength_range_preliminary, rest_wavelength, index__row_col, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', y_scale='linear', show_legend='yes'):
    
    # 1) Crop SUMER wavelength array in the range selected in the input
    lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_preliminary)
    col1, col2 = idx_sumer_crop 
    
    row_, img_ = index__row_col
                        
    # 2) Take the data of the current spectral image and row
    lam_sumer = lam_sumer #or lam_sumer[col1:col2+1]
    rad_sumer = spectralimage_interp_list_[img_][row_,:]
    erad_sumer = unc_spectralimage_interp_list_[img_][row_,:]

    # 3) Crop SUMER data
    lam_sumer_crop = lam_sumer_crop #or lam_sumer[col1:col2+1]
    rad_sumer_crop = rad_sumer[col1:col2+1]
    erad_sumer_crop = erad_sumer[col1:col2+1]

    if subtract_HRTS!='no':
        # 4) Prepare HRTS data for the subtraction
        scale_factor_hrts = scalefactor_hrts_2Darr[row_, img_]
        interp_func_hrts, lam_hrts, rad_hrts_conv_scaled = prepare_HRTS_for_subtraction(lam_hrts=lam_hrts, rad_hrts=rad_hrts, fwhm_conv=fwhm_conv, scale_factor=scale_factor_hrts)

        # 5) Get data of HRTS corresponding to SUMER's wavelength grid (of the cropped range)
        rad_hrts_SUMERgrid = interp_func_hrts(lam_sumer_crop)

        # 6) Crop HRTS
        lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrts, range_values=wavelength_range_preliminary)
        rad_hrts_conv_scaled_crop = rad_hrts_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]

        # 7) Subtract HRTS to SUMER
        rad_sumer_crop_tofit = rad_sumer_crop - rad_hrts_SUMERgrid
        erad_sumer_crop_tofit = erad_sumer_crop #TODO: not considering HRTS erorbars (wich we don't know)

    else: 
        rad_sumer_crop_tofit = rad_sumer_crop
        erad_sumer_crop_tofit = erad_sumer_crop

    # 8) Fit a single gaussian (and background) to the data. 
    res = fit_automatic_gaussian_with_background(x_data=lam_sumer_crop, y_data=rad_sumer_crop_tofit, y_unc_data=erad_sumer_crop_tofit)
    if res!=None: 
        x_centroid = res['mean'][0]
        v_centroid = vkms_doppler(lamb=x_centroid, lamb_0=rest_wavelength)
        #dv_centroid = vkms_doppler_unc(lamb=res['mean'][1], lamb_unc=dx_centroid, lamb_0=rest_wavelength, lamb_0_unc=0.0)
        chi2red_ = res['chi2red']
        bg_ = res['background'][0]
        peak_ = res['peak'][0]
        fwhm_ = res['FWHM'][0]
    else:
        v_centroid = np.nan
        chi2red_ = np.nan
        bg_ = np.nan
        peak_ = np.nan
        fwhm_ = np.nan

    # 9) Show profile
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop, yerr=erad_sumer_crop, color='black', linewidth=0.8, label='SUMER original')
    ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop_tofit, yerr=erad_sumer_crop_tofit, color='blue', linewidth=0, elinewidth=1., marker='.', label='Data to fit')
    if subtract_HRTS!='no':
        ax.errorbar(x=lam_hrts_crop, y=rad_hrts_conv_scaled_crop, color='brown', linewidth=0.8, label=f'HRTS convolved and scaled (factor {scale_factor_hrts})')
        ax.errorbar(x=lam_sumer_crop, y=rad_hrts_SUMERgrid, color='black', linewidth=0, marker='.', markersize=7, label='Interpolated SUMER wavelength data')
    x_gauss_fit = np.linspace(lam_sumer_crop[0], lam_sumer_crop[-1], 3000)
    y_gauss_fit = gaussian_with_background(x=x_gauss_fit, background=res['background'][0], peak=res['peak'][0], mean=res['mean'][0], fwhm=res['FWHM'][0])
    ax.errorbar(x=x_gauss_fit, y=y_gauss_fit, color='red', linewidth=1.5, label='Fitted Gaussian')
    ax.axvline(x=rest_wavelength, linestyle='--', color='green', linewidth=1.5, label=f'Rest wavelength {rest_wavelength}'' \u212B')
    ax.set_yscale(y_scale)
    ax.set_title(f'SOHO/SUMER profile analysis. Spectral image {img_}, lattitude row {row_}', fontsize=18) 
    ax.set_xlabel('Wavelength direction (\u212B)', color='black', fontsize=16)
    ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', color='black', fontsize=16)
    # Add secondary x-axis with the Doppler velocity
    def x_old_to_new(lam__): #Wavelength (delta) to doppler velocity. 
        c = 299792.4580 #[km/s] speed of light
        return c*(lam__-rest_wavelength)/rest_wavelength
    def x_new_to_old(v_): #Doppler velocity to wavelength.
        c = 299792.4580 #[km/s] speed of light
        return rest_wavelength*((v_/c)+1)
    secax = ax.secondary_xaxis('top', functions=(lambda lam__: x_old_to_new(lam__), lambda v_: x_new_to_old(v_)))
    secax.set_xlabel("Doppler velocity (km/s)", fontsize=14)
    dopplershift__ = x_old_to_new(lam__=x_centroid)
    if res!=None: y_centroid = res['background'][0] + res['peak'][0]
    ax.scatter(x_centroid, y_centroid, color='red', s=30, marker='^', label=f'Centroid of the Gaussian: {dopplershift__} km/s')
    if show_legend=='yes': ax.legend(fontsize=10)
    plt.show(block=False)
        
    if subtract_HRTS!='no': return {'x_centroid': x_centroid, 'v_centroid': v_centroid, 'background': bg_, 'peak': peak_, 'fwhm': fwhm_, 'chi2red': chi2red_, 'lam_sumer_crop': lam_sumer_crop, 'rad_sumer_crop': rad_sumer_crop, 'erad_sumer_crop': erad_sumer_crop, 'rad_sumer_crop_tofit': rad_sumer_crop_tofit, 'erad_sumer_crop_tofit': erad_sumer_crop_tofit, 'lam_hrts_crop': lam_hrts_crop, 'rad_hrts_conv_scaled_crop': rad_hrts_conv_scaled_crop}
    else: return {'x_centroid': x_centroid, 'v_centroid': v_centroid, 'background': bg_, 'peak': peak_, 'fwhm': fwhm_, 'chi2red': chi2red_, 'lam_sumer_crop': lam_sumer_crop, 'rad_sumer_crop': rad_sumer_crop, 'erad_sumer_crop': erad_sumer_crop}


##################################################
#

def create_SUMER_dopplermap__1pixel_profile_add_profiles(lam_profile_list, rad_profile_list, erad_profile_list, label_profile_list, lam_sumer, spectralimage_interp_list_, unc_spectralimage_interp_list_, wavelength_range_preliminary, rest_wavelength, index__row_col, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', y_scale='linear', show_legend='yes'):
    
    # 1) Crop SUMER wavelength array in the range selected in the input
    lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_preliminary)
    col1, col2 = idx_sumer_crop 
    
    row_, img_ = index__row_col
                        
    # 2) Take the data of the current spectral image and row
    lam_sumer = lam_sumer #or lam_sumer[col1:col2+1]
    rad_sumer = spectralimage_interp_list_[img_][row_,:]
    erad_sumer = unc_spectralimage_interp_list_[img_][row_,:]

    # 3) Crop SUMER data
    lam_sumer_crop = lam_sumer_crop #or lam_sumer[col1:col2+1]
    rad_sumer_crop = rad_sumer[col1:col2+1]
    erad_sumer_crop = erad_sumer[col1:col2+1]

    if subtract_HRTS!='no':
        # 4) Prepare HRTS data for the subtraction
        scale_factor_hrts = scalefactor_hrts_2Darr[row_, img_]
        interp_func_hrts, lam_hrts, rad_hrts_conv_scaled = prepare_HRTS_for_subtraction(lam_hrts=lam_hrts, rad_hrts=rad_hrts, fwhm_conv=fwhm_conv, scale_factor=scale_factor_hrts)

        # 5) Get data of HRTS corresponding to SUMER's wavelength grid (of the cropped range)
        rad_hrts_SUMERgrid = interp_func_hrts(lam_sumer_crop)

        # 6) Crop HRTS
        lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrts, range_values=wavelength_range_preliminary)
        rad_hrts_conv_scaled_crop = rad_hrts_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]

        # 7) Subtract HRTS to SUMER
        rad_sumer_crop_tofit = rad_sumer_crop - rad_hrts_SUMERgrid
        erad_sumer_crop_tofit = erad_sumer_crop #TODO: not considering HRTS erorbars (wich we don't know)

    else: 
        rad_sumer_crop_tofit = rad_sumer_crop
        erad_sumer_crop_tofit = erad_sumer_crop

    # 8) Fit a single gaussian (and background) to the data. 
    """
    res = fit_automatic_gaussian_with_background(x_data=lam_sumer_crop, y_data=rad_sumer_crop_tofit, y_unc_data=erad_sumer_crop_tofit)
    if res!=None: 
        x_centroid = res['mean'][0]
        v_centroid = vkms_doppler(lamb=x_centroid, lamb_0=rest_wavelength)
        #dv_centroid = vkms_doppler_unc(lamb=res['mean'][1], lamb_unc=dx_centroid, lamb_0=rest_wavelength, lamb_0_unc=0.0)
        chi2red_ = res['chi2red']
        bg_ = res['background'][0]
        peak_ = res['peak'][0]
        fwhm_ = res['FWHM'][0]
    else:
        v_centroid = np.nan
        chi2red_ = np.nan
        bg_ = np.nan
        peak_ = np.nan
        fwhm_ = np.nan
    """
    # 9) Show profile
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop, yerr=erad_sumer_crop, color='black', linewidth=0.8, label='SUMER original')
    ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop_tofit, yerr=erad_sumer_crop_tofit, color='blue', linewidth=0, elinewidth=1., marker='.', label='Data to fit')
    color_profile_list = ['magenta', 'purple', 'orange', 'pink', 'grey']
    for k in range(len(lam_profile_list)):
        ax.errorbar(x=lam_profile_list[k], y=rad_profile_list[k], yerr=erad_profile_list[k], label=label_profile_list[k], color=color_profile_list[k], linewidth=0.7)
    if subtract_HRTS!='no':
        ax.errorbar(x=lam_hrts_crop, y=rad_hrts_conv_scaled_crop, color='brown', linewidth=0.8, label=f'HRTS convolved and scaled (factor {np.round(scale_factor_hrts,3)})')
    """
        ax.errorbar(x=lam_sumer_crop, y=rad_hrts_SUMERgrid, color='black', linewidth=0, marker='.', markersize=7, label='Interpolated SUMER wavelength data')
    x_gauss_fit = np.linspace(lam_sumer_crop[0], lam_sumer_crop[-1], 3000)
    y_gauss_fit = gaussian_with_background(x=x_gauss_fit, background=res['background'][0], peak=res['peak'][0], mean=res['mean'][0], fwhm=res['FWHM'][0])
    ax.errorbar(x=x_gauss_fit, y=y_gauss_fit, color='red', linewidth=1.5, label='Fitted Gaussian')
    """
    ax.axvline(x=rest_wavelength, linestyle='--', color='green', linewidth=1.5, label=f'Rest wavelength {rest_wavelength}'' \u212B')
    ax.set_yscale(y_scale)
    ax.set_title(f'SOHO/SUMER profile analysis. Spectral image {img_}, lattitude row {row_}', fontsize=18) 
    ax.set_xlabel('Wavelength direction (\u212B)', color='black', fontsize=16)
    ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', color='black', fontsize=16)
    # Add secondary x-axis with the Doppler velocity
    def x_old_to_new(lam__): #Wavelength (delta) to doppler velocity. 
        c = 299792.4580 #[km/s] speed of light
        return c*(lam__-rest_wavelength)/rest_wavelength
    def x_new_to_old(v_): #Doppler velocity to wavelength.
        c = 299792.4580 #[km/s] speed of light
        return rest_wavelength*((v_/c)+1)
    secax = ax.secondary_xaxis('top', functions=(lambda lam__: x_old_to_new(lam__), lambda v_: x_new_to_old(v_)))
    secax.set_xlabel("Doppler velocity (km/s)", fontsize=14)
    """
    dopplershift__ = x_old_to_new(lam__=x_centroid)
    if res!=None: y_centroid = res['background'][0] + res['peak'][0]
    ax.scatter(x_centroid, y_centroid, color='red', s=30, marker='^', label=f'Centroid of the Gaussian: {dopplershift__} km/s')
    """
    if show_legend=='yes': ax.legend(fontsize=10)
    plt.show(block=False)

    """
    if subtract_HRTS!='no': return {'x_centroid': x_centroid, 'v_centroid': v_centroid, 'background': bg_, 'peak': peak_, 'fwhm': fwhm_, 'chi2red': chi2red_, 'lam_sumer_crop': lam_sumer_crop, 'rad_sumer_crop': rad_sumer_crop, 'erad_sumer_crop': erad_sumer_crop, 'rad_sumer_crop_tofit': rad_sumer_crop_tofit, 'erad_sumer_crop_tofit': erad_sumer_crop_tofit, 'lam_hrts_crop': lam_hrts_crop, 'rad_hrts_conv_scaled_crop': rad_hrts_conv_scaled_crop}
    else: return {'x_centroid': x_centroid, 'v_centroid': v_centroid, 'background': bg_, 'peak': peak_, 'fwhm': fwhm_, 'chi2red': chi2red_, 'lam_sumer_crop': lam_sumer_crop, 'rad_sumer_crop': rad_sumer_crop, 'erad_sumer_crop': erad_sumer_crop}
    """
    #if subtract_HRTS!='no': return {'lam_sumer_crop': lam_sumer_crop, 'rad_sumer_crop': rad_sumer_crop, 'erad_sumer_crop': erad_sumer_crop, 'rad_sumer_crop_tofit': rad_sumer_crop_tofit, 'erad_sumer_crop_tofit': erad_sumer_crop_tofit, 'lam_hrts_crop': lam_hrts_crop, 'rad_hrts_conv_scaled_crop': rad_hrts_conv_scaled_crop}
    #else: return {'lam_sumer_crop': lam_sumer_crop, 'rad_sumer_crop': rad_sumer_crop, 'erad_sumer_crop': erad_sumer_crop}



##################################################
# create Dopplermap of Doble Gaussians

def create_SUMER_dopplermap_double_gaussianfit(lam_sumer, spectralimage_interp_list_, unc_spectralimage_interp_list_, wavelength_range_preliminary, rest_wavelength, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', show__row_col='no', y_scale='linear', show_legend='yes'):
    
    # 1) Crop SUMER wavelength array in the range selected in the input
    lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_preliminary)
    col1, col2 = idx_sumer_crop    
    
    N_img = len(spectralimage_interp_list_)
    N_rows = spectralimage_interp_list_[0].shape[0]
    #N_cols_full = spectralimage_interp_list_[0].shape[1]
    #N_cols = col2-col1+1
    background_NT, dopplervelocity1_NT, dopplervelocity2_NT, centroid1_NT, peak1_NT, fwhm1_NT, centroid2_NT, peak2_NT, fwhm2_NT, chi2red_NT = [],[],[],[],[],[],[],[],[],[]
    for img_ in tqdm(range(N_img)):
        background_1sp_allrows, dopplervelocity1_1sp_allrows, dopplervelocity2_1sp_allrows, centroid1_1sp_allrows, peak1_1sp_allrows, fwhm1_1sp_allrows, centroid2_1sp_allrows, peak2_1sp_allrows, fwhm2_1sp_allrows, chi2red_1sp_allrows = [],[],[],[],[],[],[],[],[],[]
        for row_ in range(N_rows):
                        
            # 2) Take the data of the current spectral image and row
            lam_sumer = lam_sumer #or lam_sumer[col1:col2+1]
            rad_sumer = spectralimage_interp_list_[img_][row_,:]
            erad_sumer = unc_spectralimage_interp_list_[img_][row_,:]
            
            # 3) Crop SUMER data
            lam_sumer_crop = lam_sumer_crop #or lam_sumer[col1:col2+1]
            rad_sumer_crop = rad_sumer[col1:col2+1]
            erad_sumer_crop = erad_sumer[col1:col2+1]
            
            if subtract_HRTS!='no':
                # 4) Prepare HRTS data for the subtraction
                scale_factor_hrts = scalefactor_hrts_2Darr[row_, img_]
                interp_func_hrts, lam_hrts, rad_hrts_conv_scaled = prepare_HRTS_for_subtraction(lam_hrts=lam_hrts, rad_hrts=rad_hrts, fwhm_conv=fwhm_conv, scale_factor=scale_factor_hrts)
                
                # 5) Get data of HRTS corresponding to SUMER's wavelength grid (of the cropped range)
                rad_hrts_SUMERgrid = interp_func_hrts(lam_sumer_crop)
                
                # 6) Crop HRTS
                lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrts, range_values=wavelength_range_preliminary)
                rad_hrts_conv_scaled_crop = rad_hrts_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]

                # 7) Subtract HRTS to SUMER
                rad_sumer_crop_tofit = rad_sumer_crop - rad_hrts_SUMERgrid
                erad_sumer_crop_tofit = erad_sumer_crop #TODO: not considering HRTS erorbars (wich we don't know)
            else: 
                rad_sumer_crop_tofit = rad_sumer_crop
                erad_sumer_crop_tofit = erad_sumer_crop
            
            # 8) Fit a double gaussian (and background) to the data. 
            res = fit_automatic_double_gaussian_with_background(x_data=lam_sumer_crop, y_data=rad_sumer_crop_tofit, y_unc_data=erad_sumer_crop_tofit)
            if res!=None: 
                bg = res['background'][0]
                m1 = res['mean1'][0]
                f1 = res['fwhm1'][0]
                p1 = res['peak1'][0]
                m2 = res['mean2'][0]
                f2 = res['fwhm2'][0]
                p2 = res['peak2'][0]
                chi2red_ = res['chi2red']
                dopplervelocity1 = vkms_doppler(lamb=m1, lamb_0=rest_wavelength)
                dopplervelocity2 = vkms_doppler(lamb=m2, lamb_0=rest_wavelength)
                #"m" = "mean", "1,2"="gaussian 1 (left, blue), gaussian 2 (right, red)", "res"="fitting results"
            else:
                bg = np.nan
                m1 = np.nan
                f1 = np.nan
                p1 = np.nan
                m2 = np.nan
                f2 = np.nan
                p2 = np.nan
                chi2red_ = np.nan
            
            # 9) Show profile
            if show__row_col!='no' and show__row_col[0]==row_ and show__row_col[1]==img_:
                
                # Model (curve)
                x_fit = np.linspace(lam_sumer_crop[0], lam_sumer_crop[-1], 1000)
                y_fit = double_gaussian_with_background(x=x_fit, background=bg, peak1=p1, mean1=m1, fwhm1=f1, peak2=p2, mean2=m2, fwhm2=f2)
                y_fit_1 = gaussian_with_background(x=x_fit, background=bg, peak=p1, mean=m1, fwhm=f1)
                y_fit_2 = gaussian_with_background(x=x_fit, background=bg, peak=p2, mean=m2, fwhm=f2)
                
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))                
                
                ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop, yerr=erad_sumer_crop, color='black', linewidth=0.8, label='SUMER original')
                ax.errorbar(x=x_fit, y=y_fit, color='orange', linewidth=1.5, label='Double gaussian fit')
                ax.errorbar(x=x_fit, y=y_fit_1, color='blue', linestyle=':', linewidth=1.5)#, label='Fit Gaussian 1 ("dirty")')
                ax.errorbar(x=x_fit, y=y_fit_2, color='red', linestyle=':', linewidth=1.5)#, label='Fit Gaussian 2 ("dirty")')
                ax.scatter(m1, p1+bg, color='blue', s=30, marker='^')#, label=f'Centroid of the Gaussian 1')
                ax.scatter(m2, p2+bg, color='red', s=30, marker='^')#, label=f'Centroid of the Gaussian 2')
                
                if subtract_HRTS!='no':
                    ax.errorbar(x=lam_hrts_crop, y=rad_hrts_conv_scaled_crop, color='grey', linewidth=0.8, label=f'HRTS convolved and scaled (factor {scale_factor_hrts})')
                    ax.errorbar(x=lam_sumer_crop, y=rad_hrts_SUMERgrid, color='black', linewidth=0, marker='.', markersize=7, label='HRTS interpolated to SUMER grid')
                    ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop_tofit, yerr=erad_sumer_crop_tofit, color='brown', linewidth=0.5, marker='.', markersize=7, label='SUMER after subtraction of HRTS')
                    
                ax.axvline(x=rest_wavelength, linestyle='--', color='green', linewidth=1.5, label=f'Rest wavelength {rest_wavelength}'' \u212B')
                ax.set_yscale(y_scale)
                ax.set_title(f'SOHO/SUMER profile analysis. Spectral image {img_}, lattitude row {row_}', fontsize=18) 
                ax.set_xlabel('Wavelength direction (\u212B)', color='black', fontsize=16)
                ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', color='black', fontsize=16)
                # Add secondary x-axis with the Doppler velocity
                def x_old_to_new(lam__): #Wavelength (delta) to doppler velocity. 
                    c = 299792.4580 #[km/s] speed of light
                    return c*(lam__-rest_wavelength)/rest_wavelength
                def x_new_to_old(v_): #Doppler velocity to wavelength.
                    c = 299792.4580 #[km/s] speed of light
                    return rest_wavelength*((v_/c)+1)
                secax = ax.secondary_xaxis('top', functions=(lambda lam__: x_old_to_new(lam__), lambda v_: x_new_to_old(v_)))
                secax.set_xlabel("Doppler velocity (km/s)", fontsize=14)
                #dopplershift__ = x_old_to_new(lam__=x_centroid)
                if show_legend=='yes': ax.legend(fontsize=10)
                plt.show(block=False)
            
            # 10) Save centroid of the Gaussian in a list
            background_1sp_allrows.append(bg)
            centroid1_1sp_allrows.append(m1)
            peak1_1sp_allrows.append(p1)
            fwhm1_1sp_allrows.append(f1)
            centroid2_1sp_allrows.append(m2)
            peak2_1sp_allrows.append(p2)
            fwhm2_1sp_allrows.append(f2)
            chi2red_1sp_allrows.append(chi2red_)
            dopplervelocity1_1sp_allrows.append(dopplervelocity1)
            dopplervelocity2_1sp_allrows.append(dopplervelocity2)
        
        # 11) Save the above lists (which represent the different spectral images) in lists to crate the 2D-array
        background_NT.append(background_1sp_allrows)
        centroid1_NT.append(centroid1_1sp_allrows)
        peak1_NT.append(peak1_1sp_allrows)
        fwhm1_NT.append(fwhm1_1sp_allrows)
        centroid2_NT.append(centroid2_1sp_allrows)
        peak2_NT.append(peak2_1sp_allrows)
        fwhm2_NT.append(fwhm2_1sp_allrows)
        chi2red_NT.append(chi2red_1sp_allrows)
        dopplervelocity1_NT.append(dopplervelocity1_1sp_allrows)
        dopplervelocity2_NT.append(dopplervelocity2_1sp_allrows)
    
    # 12) Convert to array and transpose
    background_map = np.array(background_NT).T
    centroid1_map = np.array(centroid1_NT).T
    peak1_map = np.array(peak1_NT).T
    fwhm1_map = np.array(fwhm1_NT).T
    centroid2_map = np.array(centroid2_NT).T
    peak2_map = np.array(peak2_NT).T
    fwhm2_map = np.array(fwhm2_NT).T
    chi2red_map = np.array(chi2red_NT).T
    dopplervelocity1_map = np.array(dopplervelocity1_NT).T
    dopplervelocity2_map = np.array(dopplervelocity2_NT).T
    
    #return [background_map, centroid1_map, peak1_map, fwhm1_map, centroid2_map, peak2_map, fwhm2_map, chi2red_map]
    return {'background_map':background_map, 'dopplervelocity1_map': dopplervelocity1_map, 'dopplervelocity2_map': dopplervelocity2_map, 'centroid1_map':centroid1_map, 'peak1_map':peak1_map, 'fwhm1_map':fwhm1_map, 'centroid2_map':centroid2_map, 'peak2_map':peak2_map, 'fwhm2_map':fwhm2_map, 'chi2red_map':chi2red_map}


def create_SUMER_dopplermap_double_gaussianfit__1pixel_profile(lam_sumer, spectralimage_interp_list_, unc_spectralimage_interp_list_, wavelength_range_preliminary, rest_wavelength, index__row_col, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', y_scale='linear', show_legend='yes'):
    
    # 1) Crop SUMER wavelength array in the range selected in the input
    lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_preliminary)
    col1, col2 = idx_sumer_crop
    
    row_, img_ = index__row_col
    
    # 2) Take the data of the current spectral image and row
    lam_sumer = lam_sumer #or lam_sumer[col1:col2+1]
    rad_sumer = spectralimage_interp_list_[img_][row_,:]
    erad_sumer = unc_spectralimage_interp_list_[img_][row_,:]

    # 3) Crop SUMER data
    lam_sumer_crop = lam_sumer_crop #or lam_sumer[col1:col2+1]
    rad_sumer_crop = rad_sumer[col1:col2+1]
    erad_sumer_crop = erad_sumer[col1:col2+1]

    if subtract_HRTS!='no':
        # 4) Prepare HRTS data for the subtraction
        scale_factor_hrts = scalefactor_hrts_2Darr[row_, img_]
        interp_func_hrts, lam_hrts, rad_hrts_conv_scaled = prepare_HRTS_for_subtraction(lam_hrts=lam_hrts, rad_hrts=rad_hrts, fwhm_conv=fwhm_conv, scale_factor=scale_factor_hrts)

        # 5) Get data of HRTS corresponding to SUMER's wavelength grid (of the cropped range)
        rad_hrts_SUMERgrid = interp_func_hrts(lam_sumer_crop)

        # 6) Crop HRTS
        lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrts, range_values=wavelength_range_preliminary)
        rad_hrts_conv_scaled_crop = rad_hrts_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]

        # 7) Subtract HRTS to SUMER
        rad_sumer_crop_tofit = rad_sumer_crop - rad_hrts_SUMERgrid
        erad_sumer_crop_tofit = erad_sumer_crop #TODO: not considering HRTS erorbars (wich we don't know)
    else: 
        rad_sumer_crop_tofit = rad_sumer_crop
        erad_sumer_crop_tofit = erad_sumer_crop

    # 8) Fit a double gaussian (and background) to the data. 
    res = fit_automatic_double_gaussian_with_background(x_data=lam_sumer_crop, y_data=rad_sumer_crop_tofit, y_unc_data=erad_sumer_crop_tofit)
    if res!=None: 
        bg = res['background'][0]
        m1 = res['mean1'][0]
        f1 = res['fwhm1'][0]
        p1 = res['peak1'][0]
        m2 = res['mean2'][0]
        f2 = res['fwhm2'][0]
        p2 = res['peak2'][0]
        chi2red_ = res['chi2red']
        dopplervelocity1 = vkms_doppler(lamb=m1, lamb_0=rest_wavelength)
        dopplervelocity2 = vkms_doppler(lamb=m2, lamb_0=rest_wavelength)
        #"m" = "mean", "1,2"="gaussian 1 (left, blue), gaussian 2 (right, red)", "res"="fitting results"
    else:
        bg = np.nan
        m1 = np.nan
        f1 = np.nan
        p1 = np.nan
        m2 = np.nan
        f2 = np.nan
        p2 = np.nan
        chi2red_ = np.nan
        dopplervelocity1 = np.nan
        dopplervelocity2 = np.nan

    # 9) Show profile

    # Model (curve)
    x_fit = np.linspace(lam_sumer_crop[0], lam_sumer_crop[-1], 1000)
    y_fit = double_gaussian_with_background(x=x_fit, background=bg, peak1=p1, mean1=m1, fwhm1=f1, peak2=p2, mean2=m2, fwhm2=f2)
    y_fit_1 = gaussian_with_background(x=x_fit, background=bg, peak=p1, mean=m1, fwhm=f1)
    y_fit_2 = gaussian_with_background(x=x_fit, background=bg, peak=p2, mean=m2, fwhm=f2)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))                

    ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop, yerr=erad_sumer_crop, color='black', linewidth=0.8, label='SUMER original')
    ax.errorbar(x=x_fit, y=y_fit, color='orange', linewidth=1.5, label='Double gaussian fit')
    ax.errorbar(x=x_fit, y=y_fit_1, color='blue', linestyle=':', linewidth=1.5)#, label='Fit Gaussian 1 ("dirty")')
    ax.errorbar(x=x_fit, y=y_fit_2, color='red', linestyle=':', linewidth=1.5)#, label='Fit Gaussian 2 ("dirty")')
    ax.scatter(m1, p1+bg, color='blue', s=30, marker='^')#, label=f'Centroid of the Gaussian 1')
    ax.scatter(m2, p2+bg, color='red', s=30, marker='^')#, label=f'Centroid of the Gaussian 2')

    if subtract_HRTS!='no':
        ax.errorbar(x=lam_hrts_crop, y=rad_hrts_conv_scaled_crop, color='grey', linewidth=0.8, label=f'HRTS convolved and scaled (factor {scale_factor_hrts})')
        ax.errorbar(x=lam_sumer_crop, y=rad_hrts_SUMERgrid, color='black', linewidth=0, marker='.', markersize=7, label='HRTS interpolated to SUMER grid')
        ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop_tofit, yerr=erad_sumer_crop_tofit, color='brown', linewidth=0.5, marker='.', markersize=7, label='SUMER after subtraction of HRTS')

    ax.axvline(x=rest_wavelength, linestyle='--', color='green', linewidth=1.5, label=f'Rest wavelength {rest_wavelength}'' \u212B')
    ax.set_yscale(y_scale)
    ax.set_title(f'SOHO/SUMER profile analysis. Spectral image {img_}, lattitude row {row_}', fontsize=18) 
    ax.set_xlabel('Wavelength direction (\u212B)', color='black', fontsize=16)
    ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', color='black', fontsize=16)
    # Add secondary x-axis with the Doppler velocity
    def x_old_to_new(lam__): #Wavelength (delta) to doppler velocity. 
        c = 299792.4580 #[km/s] speed of light
        return c*(lam__-rest_wavelength)/rest_wavelength
    def x_new_to_old(v_): #Doppler velocity to wavelength.
        c = 299792.4580 #[km/s] speed of light
        return rest_wavelength*((v_/c)+1)
    secax = ax.secondary_xaxis('top', functions=(lambda lam__: x_old_to_new(lam__), lambda v_: x_new_to_old(v_)))
    secax.set_xlabel("Doppler velocity (km/s)", fontsize=14)
    #dopplershift__ = x_old_to_new(lam__=x_centroid)
    if show_legend=='yes': ax.legend(fontsize=10)
    plt.show(block=False)
    
    if subtract_HRTS!='no': return {'background': bg, 'v1': dopplervelocity1, 'v2': dopplervelocity2, 'mean1': m1, 'fwhm1': f1, 'peak1': p1, 'mean2': m2, 'fwhm2': f2, 'peak2': p2, 'chi2red': chi2red_, 'lam_sumer_crop': lam_sumer_crop, 'rad_sumer_crop': rad_sumer_crop, 'erad_sumer_crop': erad_sumer_crop, 'rad_sumer_crop_tofit': rad_sumer_crop_tofit, 'erad_sumer_crop_tofit': erad_sumer_crop_tofit, 'lam_hrts_crop': lam_hrts_crop, 'rad_hrts_conv_scaled_crop': rad_hrts_conv_scaled_crop}
    else: return {'background': bg, 'v1': dopplervelocity1, 'v2': dopplervelocity2, 'mean1': m1, 'fwhm1': f1, 'peak1': p1, 'mean2': m2, 'fwhm2': f2, 'peak2': p2, 'chi2red': chi2red_, 'lam_sumer_crop': lam_sumer_crop, 'rad_sumer_crop': rad_sumer_crop, 'erad_sumer_crop': erad_sumer_crop}



##################################################
# 

def create_SUMER_dopplermap_no_gaussianfit__1pixel_profile(lam_sumer, spectralimage_interp_list_, unc_spectralimage_interp_list_, wavelength_range_preliminary, rest_wavelength, index__row_col, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', y_scale='linear', show_legend='yes'):
    
    # 1) Crop SUMER wavelength array in the range selected in the input
    lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_preliminary)
    col1, col2 = idx_sumer_crop 
    
    row_, img_ = index__row_col
                        
    # 2) Take the data of the current spectral image and row
    lam_sumer = lam_sumer #or lam_sumer[col1:col2+1]
    rad_sumer = spectralimage_interp_list_[img_][row_,:]
    erad_sumer = unc_spectralimage_interp_list_[img_][row_,:]

    # 3) Crop SUMER data
    lam_sumer_crop = lam_sumer_crop #or lam_sumer[col1:col2+1]
    rad_sumer_crop = rad_sumer[col1:col2+1]
    erad_sumer_crop = erad_sumer[col1:col2+1]

    if subtract_HRTS!='no':
        # 4) Prepare HRTS data for the subtraction
        scale_factor_hrts = scalefactor_hrts_2Darr[row_, img_]
        interp_func_hrts, lam_hrts, rad_hrts_conv_scaled = prepare_HRTS_for_subtraction(lam_hrts=lam_hrts, rad_hrts=rad_hrts, fwhm_conv=fwhm_conv, scale_factor=scale_factor_hrts)

        # 5) Get data of HRTS corresponding to SUMER's wavelength grid (of the cropped range)
        rad_hrts_SUMERgrid = interp_func_hrts(lam_sumer_crop)

        # 6) Crop HRTS
        lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrts, range_values=wavelength_range_preliminary)
        rad_hrts_conv_scaled_crop = rad_hrts_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]

        # 7) Subtract HRTS to SUMER
        rad_sumer_crop_tofit = rad_sumer_crop - rad_hrts_SUMERgrid
        erad_sumer_crop_tofit = erad_sumer_crop #TODO: not considering HRTS erorbars (wich we don't know)

    else: 
        rad_sumer_crop_tofit = rad_sumer_crop
        erad_sumer_crop_tofit = erad_sumer_crop
    
    # 9) Show profile
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop, yerr=erad_sumer_crop, color='black', linewidth=0.8, label='SUMER. HRTS not subtracted.')
    if subtract_HRTS!='no':
        ax.errorbar(x=lam_hrts_crop, y=rad_hrts_conv_scaled_crop, color='brown', linewidth=0.8, label=f'HRTS convolved and scaled (factor {scale_factor_hrts})')
        ax.errorbar(x=lam_sumer_crop, y=rad_hrts_SUMERgrid, color='black', linewidth=0, marker='.', markersize=7, label='Interpolated SUMER wavelength data')
        ax.errorbar(x=lam_sumer_crop, y=rad_sumer_crop_tofit, yerr=erad_sumer_crop_tofit, color='blue', linewidth=0.8, elinewidth=1., marker='.', label='SUMER. HRTS subtracted.')
    ax.axvline(x=rest_wavelength, linestyle='--', color='green', linewidth=1.5, label=f'Rest wavelength {rest_wavelength}'' \u212B')
    ax.set_yscale(y_scale)
    ax.set_title(f'SOHO/SUMER profile analysis. Spectral image {img_}, lattitude row {row_}', fontsize=18) 
    ax.set_xlabel('Wavelength direction (\u212B)', color='black', fontsize=16)
    ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', color='black', fontsize=16)
    # Add secondary x-axis with the Doppler velocity
    def x_old_to_new(lam__): #Wavelength (delta) to doppler velocity. 
        c = 299792.4580 #[km/s] speed of light
        return c*(lam__-rest_wavelength)/rest_wavelength
    def x_new_to_old(v_): #Doppler velocity to wavelength.
        c = 299792.4580 #[km/s] speed of light
        return rest_wavelength*((v_/c)+1)
    secax = ax.secondary_xaxis('top', functions=(lambda lam__: x_old_to_new(lam__), lambda v_: x_new_to_old(v_)))
    secax.set_xlabel("Doppler velocity (km/s)", fontsize=14)
    if show_legend=='yes': ax.legend(fontsize=10)
    plt.show(block=False)
        
    if subtract_HRTS!='no': return {'lam_sumer_crop': lam_sumer_crop, 'rad_sumer_crop': rad_sumer_crop, 'erad_sumer_crop': erad_sumer_crop, 'rad_sumer_crop_tofit': rad_sumer_crop_tofit, 'erad_sumer_crop_tofit': erad_sumer_crop_tofit, 'lam_hrts_crop': lam_hrts_crop, 'rad_hrts_conv_scaled_crop': rad_hrts_conv_scaled_crop}
    else: return {'lam_sumer_crop': lam_sumer_crop, 'rad_sumer_crop': rad_sumer_crop, 'erad_sumer_crop': erad_sumer_crop}



##################################################
##################################################
##################################################
# Create BR asymmetry map using the rest wavelength


def BR_asymmetry_one_spectrum_with_rest_wavelength(x_data, y_data, y_unc_data, BR_distance_centroid, BR_width, show_profile='no', show_legend='yes'):

    BR_centroid = 0.

    # width of the pixels (supposing same width for all of them)
    pixel_width = x_data[1]-x_data[0] 
    
    # Convert masked values to NaNs:
    y_data = np.ma.filled(y_data, np.nan)
    
    # Create the interpolation function (linear unterpolation)
    interp_func = interp1d(x_data, y_data, kind='linear')
    
    # Edges of the pixels
    pixel_low = x_data - pixel_width/2
    pixel_high = x_data + pixel_width/2
    
    # Calculate the ranges for blue and red parts.
    ## Blue part
    range_blue0 = BR_centroid - BR_distance_centroid - BR_width/2
    range_blue1 = BR_centroid - BR_distance_centroid + BR_width/2
    range_B = [range_blue0, range_blue1]
    ## Red part
    range_red0 = BR_centroid + BR_distance_centroid - BR_width/2
    range_red1 = BR_centroid + BR_distance_centroid + BR_width/2
    range_R = [range_red0, range_red1]
    
    # Indices inside the range
    ## Blue
    ilB = np.argmax(pixel_low > range_B[0]) # find the first index where the data point is greater than range_[0]
    ihB = np.argmax(pixel_high > range_B[1])-1 # find the first index where the data point is smaller than range_[1]
    ## Red
    ilR = np.argmax(pixel_low > range_R[0]) # find the first index where the data point is greater than range_[0]
    ihR = np.argmax(pixel_high > range_R[1])-1 # find the first index where the data point is smaller than range_[1]
    
    # Data points (entire pixels) that fit inside each range 
    ## Blue
    indices_inrange_B = np.arange(ilB,ihB+1)
    x_inrange_B = x_data[indices_inrange_B]
    y_inrange_B = y_data[indices_inrange_B]
    y_unc_data_inrange_B = y_unc_data[indices_inrange_B]
    ## Red
    indices_inrange_R = np.arange(ilR,ihR+1)
    x_inrange_R = x_data[indices_inrange_R]
    y_inrange_R = y_data[indices_inrange_R]
    y_unc_data_inrange_R = y_unc_data[indices_inrange_R]
    
    # Adding the partial pixels of the edge of the range
    ## Blue
    x_interpolated_left_B = np.mean([range_B[0], x_data[ilB]-pixel_width/2])
    x_interpolated_right_B = np.mean([range_B[1], x_data[ihB]+pixel_width/2])
    x_analysis_B = np.concatenate([[x_interpolated_left_B], x_inrange_B, [x_interpolated_right_B]])
    y_analysis_B = interp_func(x_analysis_B)
    ## Red
    x_interpolated_left_R = np.mean([range_R[0], x_data[ilR]-pixel_width/2])
    x_interpolated_right_R = np.mean([range_R[1], x_data[ihR]+pixel_width/2])
    x_analysis_R = np.concatenate([[x_interpolated_left_R], x_inrange_R, [x_interpolated_right_R]])
    y_analysis_R = interp_func(x_analysis_R)
    
    # Delta pixels (of the range)
    ## Blue
    deltapixel_inrange_B = pixel_width * np.ones(len(x_inrange_B))
    deltapixel_left_B = abs((x_data[ilB]-pixel_width/2) - range_B[0])
    deltapixel_right_B = abs(range_B[1] - (x_data[ihB]+pixel_width/2))
    deltapixel_analysis_B = np.concatenate([[deltapixel_left_B], deltapixel_inrange_B, [deltapixel_right_B]])
    ## Red
    deltapixel_inrange_R = pixel_width * np.ones(len(x_inrange_R))
    deltapixel_left_R = abs((x_data[ilR]-pixel_width/2) - range_R[0])
    deltapixel_right_R = abs(range_R[1] - (x_data[ihR]+pixel_width/2))
    deltapixel_analysis_R = np.concatenate([[deltapixel_left_R], deltapixel_inrange_R, [deltapixel_right_R]])
    
    # Mask the NaN values (otherwise the result is a nan). If the input has masked values, they are converted to NaN values, so here we need to mask them.
    ## Blue
    mask_B = np.isnan(y_analysis_B)  # Create a boolean mask where ia has NaNs
    y_analysis_masked_B = y_analysis_B[~mask_B]  # Select elements of ia where mask is False (i.e., not NaN)
    deltapixel_analysis_masked_B = deltapixel_analysis_B[~mask_B]  # Similarly filter the weights
    ia_B, w_B = y_analysis_masked_B, deltapixel_analysis_masked_B
    y_weighted_mean_B = np.sum(w_B*ia_B) / np.sum(w_B)
    ## Red
    mask_R = np.isnan(y_analysis_R)  # Create a boolean mask where ia has NaNs
    y_analysis_masked_R = y_analysis_R[~mask_R]  # Select elements of ia where mask is False (i.e., not NaN)
    deltapixel_analysis_masked_R = deltapixel_analysis_R[~mask_R]  # Similarly filter the weights
    ia_R, w_R = y_analysis_masked_R, deltapixel_analysis_masked_R
    y_weighted_mean_R = np.sum(w_R*ia_R) / np.sum(w_R)
    
    # Calculate B-R asymmetry
    BR_asymmetry = y_weighted_mean_R - y_weighted_mean_B
    BR_asymmetry_normalized = (y_weighted_mean_R - y_weighted_mean_B) / (y_weighted_mean_R + y_weighted_mean_B)
    
    
    # 9) Show profile
    if show_profile=='yes':
        # Generate new x values for smooth curve
        interp_xcurve = np.linspace(min(x_data), max(x_data), 500) # More points for smooth line
        interp_ycurve = interp_func(interp_xcurve)  # Interpolated y values

        # Wavelengths corresponding to the edges of the pixels
        pixel_edges = np.concatenate([pixel_low, [pixel_high[-1]]])

        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,6))
        ax.errorbar(x=x_data, y=y_data, yerr=y_unc_data, linewidth=0, elinewidth=1, marker='.', color='black')#, label='Intensities of every pixel (center)') # Plot the original data points
        for pe in pixel_edges:
            ax.axvline(pe, color='black', linewidth=0.5)
        ax.plot([],[], color='black', linewidth=0.5, label='Edges of the pixels')
        ax.plot(interp_xcurve, interp_ycurve, linestyle='--', linewidth=0.5, color='blue')#, label='Linear interpolation') # Plot the interpolated line
        ax.axvspan(range_B[0], range_B[1], color='blue', linestyle=':', alpha=0.15)#, label='Blue range\n'f'{range_B[0]} - {range_B[1]} {xunits}')
        ax.axvspan(range_R[0], range_R[1], color='red', linestyle=':', alpha=0.15)#, label='Red range\n'f'{range_R[0]} - {range_R[1]} {xunits}')

        # Adding the partial pixels of the edge of the range
        ax.scatter(x_analysis_B, y_analysis_B, color='blue', marker='s', s=20)#, label='"Artificial" pixels at the edges of the range (blue)') # Plot the original data points
        ax.scatter(x_analysis_R, y_analysis_R, color='red', marker='s', s=20)#, label='"Artificial" pixels at the edges of the range (red)') # Plot the original data points
        
        # Delta pixel
        ax.errorbar(x=x_analysis_B, y=y_analysis_B, xerr=deltapixel_analysis_masked_B/2, color='blue', linewidth=0, elinewidth=1.2)
        ax.errorbar(x=x_analysis_R, y=y_analysis_R, xerr=deltapixel_analysis_masked_R/2, color='red', linewidth=0, elinewidth=1.2)

        # Horizontal line as weighted mean
        ax.plot(range_B, [y_weighted_mean_B, y_weighted_mean_B], color='blue', linestyle='--')#, label=f'Weighted mean of the intensity (blue): {y_weighted_mean_B}')
        ax.plot(range_R, [y_weighted_mean_R, y_weighted_mean_R], color='red', linestyle='--')#, label=f'Weighted mean of the intensity (red): {y_weighted_mean_R}')

        # Vertical lines: centroid and rest wavelength
        ax.axvline(x=0, color='green', linestyle=':', label=f'Rest wavelength')
        ax.axvline(x=BR_centroid, color='purple', linestyle=':', label=f'Centroid: {np.round(BR_centroid,3)} km/s')
        
        ax.set_xlabel('Doppler velocity (km/s)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'B-R asymmetry (with interpolation): {np.round(BR_asymmetry,5)}')
        if show_legend=='yes': ax.legend(fontsize=10)
        plt.show(block=False)
        #ax.set_xlim([-110,-50])
        ax.legend()
    
        
    return [BR_asymmetry, BR_asymmetry_normalized]
    #return [interp_func, BR_asymmetry, interp_func, pixel_width, pixel_low, pixel_high, range_B, x_inrange_B, y_inrange_B, y_unc_data_inrange_B, x_interpolated_left_B, x_interpolated_right_B, x_analysis_B, y_analysis_B, deltapixel_inrange_B, deltapixel_left_B, deltapixel_right_B, deltapixel_analysis_B, y_weighted_mean_B, range_R, x_inrange_R, y_inrange_R, y_unc_data_inrange_R, x_interpolated_left_R, x_interpolated_right_R, x_analysis_R, y_analysis_R, deltapixel_inrange_R, deltapixel_left_R, deltapixel_right_R, deltapixel_analysis_R, y_weighted_mean_R]

##################################################
# create BR asymmetry map

def create_BRasymmetrymap_with_rest_wavelength(lam_sumer, spectralimage_interp_list_, unc_spectralimage_interp_list_, wavelength_range_preliminary, rest_wavelength, BR_distance_centroid, BR_width, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', show__row_col='no', y_scale='linear', show_legend='yes'):
    
    # 1) Crop SUMER wavelength array in the range selected in the input
    lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_preliminary)
    col1, col2 = idx_sumer_crop    
    v_sumer_crop = vkms_doppler(lamb=lam_sumer_crop, lamb_0=rest_wavelength)
    
    N_img = len(spectralimage_interp_list_)
    N_rows = spectralimage_interp_list_[0].shape[0]
    #N_cols_full = spectralimage_interp_list_[0].shape[1]
    #N_cols = col2-col1+1
    BR_asymmetry_NT, BR_asymmetry_normalized_NT = [],[]
    for img_ in tqdm(range(N_img)):
        BR_asymmetry_1sp_allrows, BR_asymmetry_normalized_1sp_allrows = [],[]
        for row_ in range(N_rows):
                        
            # 2) Take the data of the current spectral image and row
            lam_sumer = lam_sumer #or lam_sumer[col1:col2+1]
            rad_sumer = spectralimage_interp_list_[img_][row_,:]
            erad_sumer = unc_spectralimage_interp_list_[img_][row_,:]
            
            # 3) Crop SUMER data
            lam_sumer_crop = lam_sumer_crop #or lam_sumer[col1:col2+1]
            rad_sumer_crop = rad_sumer[col1:col2+1]
            erad_sumer_crop = erad_sumer[col1:col2+1]
            
            if subtract_HRTS!='no':
                # 4) Prepare HRTS data for the subtraction
                scale_factor_hrts = scalefactor_hrts_2Darr[row_, img_]
                interp_func_hrts, lam_hrts, rad_hrts_conv_scaled = prepare_HRTS_for_subtraction(lam_hrts=lam_hrts, rad_hrts=rad_hrts, fwhm_conv=fwhm_conv, scale_factor=scale_factor_hrts)

                # 5) Get data of HRTS corresponding to SUMER's wavelength grid (of the cropped range)
                rad_hrts_SUMERgrid = interp_func_hrts(lam_sumer_crop)
                
                # 6) Crop HRTS
                lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrts, range_values=wavelength_range_preliminary)
                rad_hrts_conv_scaled_crop = rad_hrts_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]

                # 7) Subtract HRTS to SUMER
                rad_sumer_crop_tofit = rad_sumer_crop - rad_hrts_SUMERgrid
                erad_sumer_crop_tofit = erad_sumer_crop #TODO: not considering HRTS erorbars (wich we don't know)
            
            else: 
                rad_sumer_crop_tofit = rad_sumer_crop
                erad_sumer_crop_tofit = erad_sumer_crop
                
            v_data = v_sumer_crop
            y_data = rad_sumer_crop_tofit
            y_unc_data = erad_sumer_crop_tofit
            
            if show__row_col!='no' and show__row_col[0]==row_ and show__row_col[1]==img_: show_profile_ij='yes'
            else: show_profile_ij='no'

            # Calculate B-R asymmetry of a single spectrum, and save it in a list
            threshold_left = 0. - BR_distance_centroid - BR_width/2
            threshold_right = 0. + BR_distance_centroid + BR_width/2
            if threshold_left < min(v_data) or threshold_right > max(v_data): 
                BR_asymmetry_normalized_1sp_allrows.append(np.nan)
                BR_asymmetry_1sp_allrows.append(np.nan)
            else:
                BR_asymmetry_ij, BR_asymmetry_normalized_ij = BR_asymmetry_one_spectrum_with_rest_wavelength(x_data=v_data, y_data=y_data, y_unc_data=y_unc_data, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width, show_profile=show_profile_ij, show_legend=show_legend)
                BR_asymmetry_normalized_1sp_allrows.append(BR_asymmetry_normalized_ij)
                BR_asymmetry_1sp_allrows.append(BR_asymmetry_ij)
            
        # save the above lists (which represent the different spectral images) in lists to crate the 2D-array
        BR_asymmetry_normalized_NT.append(BR_asymmetry_normalized_1sp_allrows)
        BR_asymmetry_NT.append(BR_asymmetry_1sp_allrows)

    # Convert to array and transpose
    BR_asymmetry_normalized_map = np.array(BR_asymmetry_normalized_NT).T 
    BR_asymmetry_map = np.array(BR_asymmetry_NT).T 
    
    return [BR_asymmetry_map, BR_asymmetry_normalized_map]



##################################################
##################################################
##################################################
# Create BR asymmetry map using the centroid of a single gaussian fit


def BR_asymmetry_one_spectrum_with_centroid_of_gaussian(x_data, y_data, y_unc_data, BR_distance_centroid, BR_width, BR_centroid, show_profile='no', show_legend='yes'):

    # width of the pixels (supposing same width for all of them)
    pixel_width = x_data[1]-x_data[0] 
    
    # Convert masked values to NaNs:
    y_data = np.ma.filled(y_data, np.nan)
    
    # Create the interpolation function (linear unterpolation)
    interp_func = interp1d(x_data, y_data, kind='linear')
    
    # Edges of the pixels
    pixel_low = x_data - pixel_width/2
    pixel_high = x_data + pixel_width/2
    
    # Calculate the ranges for blue and red parts.
    ## Blue part
    range_blue0 = BR_centroid - BR_distance_centroid - BR_width/2
    range_blue1 = BR_centroid - BR_distance_centroid + BR_width/2
    range_B = [range_blue0, range_blue1]
    ## Red part
    range_red0 = BR_centroid + BR_distance_centroid - BR_width/2
    range_red1 = BR_centroid + BR_distance_centroid + BR_width/2
    range_R = [range_red0, range_red1]
    
    # Indices inside the range
    ## Blue
    ilB = np.argmax(pixel_low > range_B[0]) # find the first index where the data point is greater than range_[0]
    ihB = np.argmax(pixel_high > range_B[1])-1 # find the first index where the data point is smaller than range_[1]
    ## Red
    ilR = np.argmax(pixel_low > range_R[0]) # find the first index where the data point is greater than range_[0]
    ihR = np.argmax(pixel_high > range_R[1])-1 # find the first index where the data point is smaller than range_[1]
    
    # Data points (entire pixels) that fit inside each range 
    ## Blue
    indices_inrange_B = np.arange(ilB,ihB+1)
    x_inrange_B = x_data[indices_inrange_B]
    y_inrange_B = y_data[indices_inrange_B]
    y_unc_data_inrange_B = y_unc_data[indices_inrange_B]
    ## Red
    indices_inrange_R = np.arange(ilR,ihR+1)
    x_inrange_R = x_data[indices_inrange_R]
    y_inrange_R = y_data[indices_inrange_R]
    y_unc_data_inrange_R = y_unc_data[indices_inrange_R]
    
    # Adding the partial pixels of the edge of the range
    ## Blue
    x_interpolated_left_B = np.mean([range_B[0], x_data[ilB]-pixel_width/2])
    x_interpolated_right_B = np.mean([range_B[1], x_data[ihB]+pixel_width/2])
    x_analysis_B = np.concatenate([[x_interpolated_left_B], x_inrange_B, [x_interpolated_right_B]])
    y_analysis_B = interp_func(x_analysis_B)
    ## Red
    x_interpolated_left_R = np.mean([range_R[0], x_data[ilR]-pixel_width/2])
    x_interpolated_right_R = np.mean([range_R[1], x_data[ihR]+pixel_width/2])
    x_analysis_R = np.concatenate([[x_interpolated_left_R], x_inrange_R, [x_interpolated_right_R]])
    y_analysis_R = interp_func(x_analysis_R)
    
    # Delta pixels (of the range)
    ## Blue
    deltapixel_inrange_B = pixel_width * np.ones(len(x_inrange_B))
    deltapixel_left_B = abs((x_data[ilB]-pixel_width/2) - range_B[0])
    deltapixel_right_B = abs(range_B[1] - (x_data[ihB]+pixel_width/2))
    deltapixel_analysis_B = np.concatenate([[deltapixel_left_B], deltapixel_inrange_B, [deltapixel_right_B]])
    ## Red
    deltapixel_inrange_R = pixel_width * np.ones(len(x_inrange_R))
    deltapixel_left_R = abs((x_data[ilR]-pixel_width/2) - range_R[0])
    deltapixel_right_R = abs(range_R[1] - (x_data[ihR]+pixel_width/2))
    deltapixel_analysis_R = np.concatenate([[deltapixel_left_R], deltapixel_inrange_R, [deltapixel_right_R]])
    
    # Mask the NaN values (otherwise the result is a nan). If the input has masked values, they are converted to NaN values, so here we need to mask them.
    ## Blue
    mask_B = np.isnan(y_analysis_B)  # Create a boolean mask where ia has NaNs
    y_analysis_masked_B = y_analysis_B[~mask_B]  # Select elements of ia where mask is False (i.e., not NaN)
    deltapixel_analysis_masked_B = deltapixel_analysis_B[~mask_B]  # Similarly filter the weights
    ia_B, w_B = y_analysis_masked_B, deltapixel_analysis_masked_B
    y_weighted_mean_B = np.sum(w_B*ia_B) / np.sum(w_B)
    ## Red
    mask_R = np.isnan(y_analysis_R)  # Create a boolean mask where ia has NaNs
    y_analysis_masked_R = y_analysis_R[~mask_R]  # Select elements of ia where mask is False (i.e., not NaN)
    deltapixel_analysis_masked_R = deltapixel_analysis_R[~mask_R]  # Similarly filter the weights
    ia_R, w_R = y_analysis_masked_R, deltapixel_analysis_masked_R
    y_weighted_mean_R = np.sum(w_R*ia_R) / np.sum(w_R)
    
    # Calculate B-R asymmetry
    BR_asymmetry = y_weighted_mean_R - y_weighted_mean_B
    BR_asymmetry_normalized = (y_weighted_mean_R - y_weighted_mean_B) / (y_weighted_mean_R + y_weighted_mean_B)
    
    
    # 9) Show profile
    if show_profile=='yes':
        # Generate new x values for smooth curve
        interp_xcurve = np.linspace(min(x_data), max(x_data), 500) # More points for smooth line
        interp_ycurve = interp_func(interp_xcurve)  # Interpolated y values

        # Wavelengths corresponding to the edges of the pixels
        pixel_edges = np.concatenate([pixel_low, [pixel_high[-1]]])

        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,6))
        ax.errorbar(x=x_data, y=y_data, yerr=y_unc_data, linewidth=0, elinewidth=1, marker='.', color='black')#, label='Intensities of every pixel (center)') # Plot the original data points
        for pe in pixel_edges:
            ax.axvline(pe, color='black', linewidth=0.5)
        ax.plot([],[], color='black', linewidth=0.5, label='Edges of the pixels')
        ax.plot(interp_xcurve, interp_ycurve, linestyle='--', linewidth=0.5, color='blue')#, label='Linear interpolation') # Plot the interpolated line
        ax.axvspan(range_B[0], range_B[1], color='blue', linestyle=':', alpha=0.15)#, label='Blue range\n'f'{range_B[0]} - {range_B[1]} {xunits}')
        ax.axvspan(range_R[0], range_R[1], color='red', linestyle=':', alpha=0.15)#, label='Red range\n'f'{range_R[0]} - {range_R[1]} {xunits}')

        # Adding the partial pixels of the edge of the range
        ax.scatter(x_analysis_B, y_analysis_B, color='blue', marker='s', s=20)#, label='"Artificial" pixels at the edges of the range (blue)') # Plot the original data points
        ax.scatter(x_analysis_R, y_analysis_R, color='red', marker='s', s=20)#, label='"Artificial" pixels at the edges of the range (red)') # Plot the original data points
        
        # Delta pixel
        ax.errorbar(x=x_analysis_B, y=y_analysis_B, xerr=deltapixel_analysis_masked_B/2, color='blue', linewidth=0, elinewidth=1.2)
        ax.errorbar(x=x_analysis_R, y=y_analysis_R, xerr=deltapixel_analysis_masked_R/2, color='red', linewidth=0, elinewidth=1.2)

        # Horizontal line as weighted mean
        ax.plot(range_B, [y_weighted_mean_B, y_weighted_mean_B], color='blue', linestyle='--')#, label=f'Weighted mean of the intensity (blue): {y_weighted_mean_B}')
        ax.plot(range_R, [y_weighted_mean_R, y_weighted_mean_R], color='red', linestyle='--')#, label=f'Weighted mean of the intensity (red): {y_weighted_mean_R}')

        # Vertical lines: centroid and rest wavelength
        ax.axvline(x=0, color='green', linestyle=':', label=f'Rest wavelength')
        ax.axvline(x=BR_centroid, color='purple', linestyle=':', label=f'Centroid: {np.round(BR_centroid,3)} km/s')
        
        ax.set_xlabel('Doppler velocity (km/s)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'B-R asymmetry (with interpolation): {np.round(BR_asymmetry,5)}')
        if show_legend=='yes': ax.legend(fontsize=10)
        plt.show(block=False)
        #ax.set_xlim([-110,-50])
        ax.legend()
    
        
    return [BR_asymmetry, BR_asymmetry_normalized] 
    #return [interp_func, BR_asymmetry, interp_func, pixel_width, pixel_low, pixel_high, range_B, x_inrange_B, y_inrange_B, y_unc_data_inrange_B, x_interpolated_left_B, x_interpolated_right_B, x_analysis_B, y_analysis_B, deltapixel_inrange_B, deltapixel_left_B, deltapixel_right_B, deltapixel_analysis_B, y_weighted_mean_B, range_R, x_inrange_R, y_inrange_R, y_unc_data_inrange_R, x_interpolated_left_R, x_interpolated_right_R, x_analysis_R, y_analysis_R, deltapixel_inrange_R, deltapixel_left_R, deltapixel_right_R, deltapixel_analysis_R, y_weighted_mean_R]

##################################################
# create BR asymmetry map

def create_BRasymmetrymap_with_centroid_of_gaussian(lam_sumer, spectralimage_interp_list_, unc_spectralimage_interp_list_, wavelength_range_preliminary, rest_wavelength, BR_distance_centroid, BR_width, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', show__row_col='no', y_scale='linear', show_legend='yes'):
    
    # 1) Crop SUMER wavelength array in the range selected in the input
    lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_preliminary)
    col1, col2 = idx_sumer_crop    
    v_sumer_crop = vkms_doppler(lamb=lam_sumer_crop, lamb_0=rest_wavelength)
    
    N_img = len(spectralimage_interp_list_)
    N_rows = spectralimage_interp_list_[0].shape[0]
    #N_cols_full = spectralimage_interp_list_[0].shape[1]
    #N_cols = col2-col1+1
    BR_asymmetry_NT, BR_asymmetry_normalized_NT = [],[]
    for img_ in tqdm(range(N_img)):
        BR_asymmetry_1sp_allrows, BR_asymmetry_normalized_1sp_allrows = [],[]
        for row_ in range(N_rows):
                        
            # 2) Take the data of the current spectral image and row
            lam_sumer = lam_sumer #or lam_sumer[col1:col2+1]
            rad_sumer = spectralimage_interp_list_[img_][row_,:]
            erad_sumer = unc_spectralimage_interp_list_[img_][row_,:]
            
            # 3) Crop SUMER data
            lam_sumer_crop = lam_sumer_crop #or lam_sumer[col1:col2+1]
            rad_sumer_crop = rad_sumer[col1:col2+1]
            erad_sumer_crop = erad_sumer[col1:col2+1]
            
            if subtract_HRTS!='no':
                # 4) Prepare HRTS data for the subtraction
                scale_factor_hrts = scalefactor_hrts_2Darr[row_, img_]
                interp_func_hrts, lam_hrts, rad_hrts_conv_scaled = prepare_HRTS_for_subtraction(lam_hrts=lam_hrts, rad_hrts=rad_hrts, fwhm_conv=fwhm_conv, scale_factor=scale_factor_hrts)

                # 5) Get data of HRTS corresponding to SUMER's wavelength grid (of the cropped range)
                rad_hrts_SUMERgrid = interp_func_hrts(lam_sumer_crop)
                
                # 6) Crop HRTS
                lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrts, range_values=wavelength_range_preliminary)
                rad_hrts_conv_scaled_crop = rad_hrts_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]

                # 7) Subtract HRTS to SUMER
                rad_sumer_crop_tofit = rad_sumer_crop - rad_hrts_SUMERgrid
                erad_sumer_crop_tofit = erad_sumer_crop #TODO: not considering HRTS erorbars (wich we don't know)
            
            else: 
                rad_sumer_crop_tofit = rad_sumer_crop
                erad_sumer_crop_tofit = erad_sumer_crop
                
            # Take the data of the current spectral image and row
            v_data = v_sumer_crop
            y_data = rad_sumer_crop_tofit
            y_unc_data = erad_sumer_crop_tofit
            
            # Fit a single gaussian (and background) to the data
            fit_results = fit_automatic_gaussian_with_background(x_data=v_data, y_data=y_data, y_unc_data=y_unc_data)

            # Calculate the Doppler shift of the centroid of the line, and the uncertainty. And save in lists.
            if fit_results!=None: 
                BR_centroid_i_j = fit_results['mean'][0]
            else:
                BR_centroid_i_j = np.nan
            
            # Calculate B-R asymmetry of a single spectrum, and save it in a list
            threshold_left = BR_centroid_i_j - BR_distance_centroid - BR_width/2
            threshold_right = BR_centroid_i_j + BR_distance_centroid + BR_width/2
            if threshold_left < min(v_data) or threshold_right > max(v_data): 
                BR_asymmetry_1sp_allrows.append(np.nan)
                BR_asymmetry_normalized_1sp_allrows.append(np.nan)
            else:
                BR_asymmetry_i_j, BR_asymmetry_normalized_i_j = BR_asymmetry_one_spectrum_with_centroid_of_gaussian(x_data=v_data, y_data=y_data, y_unc_data=y_unc_data, BR_centroid=BR_centroid_i_j, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width)
                BR_asymmetry_1sp_allrows.append(BR_asymmetry_i_j)
                BR_asymmetry_normalized_1sp_allrows.append(BR_asymmetry_normalized_i_j)
            
            
            if show__row_col!='no' and show__row_col[0]==row_ and show__row_col[1]==img_: show_profile_ij='yes'
            else: show_profile_ij='no'

            
        # save the above lists (which represent the different spectral images) in lists to crate the 2D-array
        BR_asymmetry_NT.append(BR_asymmetry_1sp_allrows)
        BR_asymmetry_normalized_NT.append(BR_asymmetry_normalized_1sp_allrows)

    # Convert to array and transpose
    BR_asymmetry_normalized_map = np.array(BR_asymmetry_normalized_NT).T 
    BR_asymmetry_map = np.array(BR_asymmetry_NT).T 
    
    return [BR_asymmetry_map, BR_asymmetry_normalized_map]





