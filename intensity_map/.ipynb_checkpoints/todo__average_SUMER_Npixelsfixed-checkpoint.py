#  Inputs
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'
fwhm_conv = 1.95*0.04215 #Angstrom
N_pixels_bin = 8000

# Wavelength ranges to crop spectra
wavelength_range_to_average = [1531.1147, 1551.7688]
wavelength_range_to_analyze_NeVIII = [1540.2, 1541.4]

## Ranges of wavelength
wavelength_range_scalefactor_left = [1537.7, 1539.5] #nm
wavelength_range_scalefactor_right = [1542., 1544.] #nm

# save average profile as .npy?
save_average_profile_map = 'no' 

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
from auxiliar_functions.data_path import path_data_soho 
from auxiliar_functions.SOHO_aux_functions import *
from auxiliar_functions.calibration_parameters__output import *
from auxiliar_functions.spectroheliogram_functions import *
from auxiliar_functions.solar_rotation_variables import *
from auxiliar_functions.aux_functions import *
from auxiliar_functions.general_variables import *
from auxiliar_functions.NeVIII_rest_wavelength import *

######################################################
######################################################
######################################################

# Load the intensity map and uncertainties
intensitymap_loaded_dic = np.load('../auxiliar_functions/intensity_map_'+line_label+'_interpolated.npz')
intensity_map = intensitymap_loaded_dic['intensity_map'] #2D-array
intensity_map_unc = intensitymap_loaded_dic['intensity_map_unc'] #2D-array
intensity_map_croplat = intensitymap_loaded_dic['intensity_map_croplat'] #2D-array
intensity_map_unc_croplat = intensitymap_loaded_dic['intensity_map_unc_croplat'] #2D-array
line_center_label = intensitymap_loaded_dic['line_center_label'] 
vmin_sumer, vmax_sumer = intensitymap_loaded_dic['vmin_vmax'] 

######################################################

# Rest wavelength used
rest_wavelength_label = 'Peter_and_Judge_1999' #'SUMER_atlas', 'Peter_1998', 'Dammasch_1999', 'Peter_and_Judge_1999', 'Kelly_database'
lam_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][0] #Angstrom
lam_unc_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][1] #Angstrom
print('Rest wavelength Ne VIII (2nd order):', lam_0, r'$\pm$', lam_unc_0, '\u212B')

# uncertainty of the rest wavelength in km/s
v_unc_0 = vkms_doppler_unc(lamb=lam_0, lamb_unc=lam_unc_0, lamb_0=lam_0, lamb_0_unc=lam_unc_0) 

######################################################

# Import SUMER data interpolated (wavelength calibrated)
data_interpolated_loaded = np.load('../auxiliar_functions/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)
spectral_image_interpolated_list = data_interpolated_loaded['spectral_image_interpolated_list']
spectral_image_unc_interpolated_list = data_interpolated_loaded['spectral_image_unc_interpolated_list']
lam_sumer = data_interpolated_loaded['reference_wavelength']          # scalar (0‑d array; use a_loaded.item() for Python float)
lam_sumer_unc = data_interpolated_loaded['unc_reference_wavelength'] #uncertainty of lam_sumer
row_reference = int(data_interpolated_loaded['row_reference'])        # becomes a NumPy array or object array, so I conver it to integer again

######################################################
# HRTS

from hrts_spectra.data__qqr_a_xdr import lambda__qqr_a_xdr, radiance__qqr_a_xdr, unc_radiance__qqr_a_xdr
lam_hrtsa, rad_hrtsa = 10.*lambda__qqr_a_xdr, 0.1*radiance__qqr_a_xdr #multiply by 10. and 0.1 to convert nm to Angstrom
from hrts_spectra.data__qqr_b_xdr import lambda__qqr_b_xdr, radiance__qqr_b_xdr, unc_radiance__qqr_b_xdr
lam_hrtsb, rad_hrtsb = 10.*lambda__qqr_b_xdr, 0.1*radiance__qqr_b_xdr
from hrts_spectra.data__qqr_l_xdr import lambda__qqr_l_xdr, radiance__qqr_l_xdr, unc_radiance__qqr_l_xdr
lam_hrtsl, rad_hrtsl = 10.*lambda__qqr_l_xdr, 0.1*radiance__qqr_l_xdr
lam_hrts = lam_hrtsa
erad_hrtsa = erad_hrtsb = erad_hrtsl = np.zeros(len(rad_hrtsa)) #TODO: we don't know the uncertainties of HRTS

def indices_closer_to_range(arr_1d, range_):
    rg0, rg1 = range_

    # Find indices of closest values
    idx0 = (np.abs(arr_1d - rg0)).argmin()
    idx1 = (np.abs(arr_1d - rg1)).argmin()
    return [idx0, idx1]
    
def convolution_FWHM_in_pixels(wavelength, fwhm_wavelength):
    """
    FWHM of the convolution Gaussian profile. This function calculates it in pixels of the "wavelength" array. 
    """
    w = wavelength
    pixel_scale = (w[-1] - w[0]) / (len(w)-1)
    fwhm_px = fwhm_wavelength / pixel_scale
    return fwhm_px

from scipy.optimize import curve_fit

def linear_func(x_, m_):
    return m_ * x_   # zero intercept

def error_propagation_product(a_, a_err, b_, b_err):
    """
    Error propagartion of c_=a_*b_
    """
    T1 = a_*b_err
    T2 = b_*a_err
    return np.sqrt(T1**2 + T2**2)

######################################################
######################################################
######################################################

# rows and columns inside the intensity bin
def pixels_higher_intensity_than_a_value(arr_2d, bound_, N_pixels):
    mask_ = arr_2d > bound_                    # keep only pixels brighter than the value "bound_"
    values_ = arr_2d[mask_] #values (intensity) of the pixels brighter than the value "bound_"
    coords_ = np.argwhere(mask_) #coordinates (row, col) of the pixels brighter than the value "bound_"
    idx_sorted = np.argsort(values_)   # indices that sort values in ascending order 
    coords_sorted = coords_[idx_sorted]      # coordinates ordered by brightness
    rowscols_inside_range__ = coords_sorted[:N_pixels] # Take first N_pixels (brightest N_pixels pixels)
    return rowscols_inside_range__


# Extent in pixels 
extent_sumer_px_contours = [0., intensity_map_croplat.shape[1]-1, intensity_map_croplat.shape[0]-1, 0.]
extent_sumer_px_image = [-0.5, intensity_map_croplat.shape[1]-1+0.5, intensity_map_croplat.shape[0]-1+0.5, -0.5]



########################### uncorrected
lam_peak_uncorrected_list = []
elam_peak_uncorrected_list = []
v_peak_uncorrected_list = []
ev_peak_uncorrected_list = []
########################### corrected QR A
lam_peak_corrected_qra_list = []
elam_peak_corrected_qra_list = []
v_peak_corrected_qra_list = []
ev_peak_corrected_qra_list = []
rad_peak_corrected_qra_list = []
########################### corrected QR B
lam_peak_corrected_qrb_list = []
elam_peak_corrected_qrb_list = []
v_peak_corrected_qrb_list = []
ev_peak_corrected_qrb_list = []
rad_peak_corrected_qrb_list = []
########################### corrected QR L
lam_peak_corrected_qrl_list = []
elam_peak_corrected_qrl_list = []
v_peak_corrected_qrl_list = []
ev_peak_corrected_qrl_list = []
rad_peak_corrected_qrl_list = []
###########################
bound_mean_list, bound_unc_list = [],[]

bound_i = np.min(intensity_map_croplat)-0.01
while bound_i<np.max(intensity_map_croplat):
    lower_bound_i = bound_i
    print(bound_i, ',', np.max(intensity_map_croplat))
    rowscols_inside_range_i = pixels_higher_intensity_than_a_value(arr_2d=intensity_map_croplat, bound_=bound_i, N_pixels=N_pixels_bin)
    print(len(rowscols_inside_range_i))
    row_high, col_high = rowscols_inside_range_i[-1] #address of the pixel with maximum intensity of this bin
    bound_i = intensity_map_croplat[row_high, col_high] #maximum intensity of this bin
    upper_bound_i = bound_i
    bound_mean_list.append((upper_bound_i+lower_bound_i)/2.)
    bound_unc_list.append((upper_bound_i-lower_bound_i)/2.)
    
    
    
    # convert the list of pairs [row, column] into 2 lists of rows and columns (for the scatterplot)
    y_row_list_plot, x_col_list_plot = convert_list_of_pairs_to_2_lists(list_of_pairs=rowscols_inside_range_i)
    
    
    # Average spectra of the pixels selected
    lam_sumer_av, elam_sumer_av, rad_sumer_av, erad_sumer_av = average_profiles_from_pixels_selected_from_interpolated_data(wavelength_range_=wavelength_range_to_average, data_interpolated_loaded_=data_interpolated_loaded, rows_cols_of_spectroheliogram=rowscols_inside_range_i)

    lam_sumer, elam_sumer, rad_sumer, erad_sumer = lam_sumer_av, elam_sumer_av, rad_sumer_av, erad_sumer_av

    # crop near Ne VIII
    lam_sumer_cropNeVIII, idx_sumer_crop_ = crop_range(list_to_crop=lam_sumer_av, range_values=wavelength_range_to_analyze_NeVIII)
    elam_sumer_cropNeVIII = elam_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    rad_sumer_cropNeVIII_uncorrected = rad_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    erad_sumer_cropNeVIII_uncorrected = erad_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    
    
    ######################################################
    ######################################################
    ######################################################
    # 1) Convolve HRTS spectrum with SUMER instrumental profile (gaussian profile of FWHM fwhm_conv)
    
    from scipy.ndimage import convolve1d
    
    inst_fwhm_HRTSpx = convolution_FWHM_in_pixels(wavelength=lam_hrts, fwhm_wavelength=fwhm_conv) #FWHM in pixels
    inst_sigma_HRTSpx = inst_fwhm_HRTSpx/(2*np.sqrt(2*np.log(2))) # convert FWHM to sigma (pixels)
    
    # build kernel
    half_width_ = int(5*inst_sigma_HRTSpx) #Using ±5σ captures essentially all the Gaussian
    xk = np.arange(-half_width_, half_width_+1)
    kernel_ = np.exp(-0.5*(xk/inst_sigma_HRTSpx)**2)
    kernel_ /= kernel_.sum() #Properly normalized kernel
    
    # convolve flux
    rad_hrtsa_conv = convolve1d(rad_hrtsa, kernel_, mode='constant', cval=0)
    
    # propagate variance properly
    varrad_hrtsa_conv = convolve1d(erad_hrtsa**2, kernel_**2, mode='constant', cval=0)
    erad_hrtsa_conv = np.sqrt(varrad_hrtsa_conv)
    
    
    ######################################################
    # 2) Create the interpolation function of the entire HRTS spectrum (after convolvolution)
    
    from scipy.interpolate import interp1d
    
    # Radiances
    rad_hrtsa_conv = np.ma.filled(rad_hrtsa_conv, np.nan) # Convert masked values to NaNs
    interp_func_hrtsa = interp1d(lam_hrtsa, rad_hrtsa_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
    rad_hrtsa_conv_SUMERgrid = interp_func_hrtsa(lam_sumer)
    
    # Uncertainties of the radiance
    erad_hrtsa_conv = np.ma.filled(erad_hrtsa_conv, np.nan) # Convert masked values to NaNs
    interp_func_hrtsa_err = interp1d(lam_hrtsa, erad_hrtsa_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
    erad_hrtsa_conv_SUMERgrid = interp_func_hrtsa_err(lam_sumer)
    
    ######################################################
    # 3) Scaling factor SUMER/HRTS. Crop regions at left and right of Ne VIII line to calculate the scaling factor of HRTS
    
    ################
    # 3.1) Select range(s) of wavelength
    
    # Left
    idx_left_sumer_0, idx_left_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_scalefactor_left)
    lam_sumer_cropleft = lam_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
    rad_sumer_cropleft = rad_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
    erad_sumer_cropleft = erad_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
    rad_hrtsa_conv_SUMERgrid_cropleft = rad_hrtsa_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
    erad_hrtsa_conv_SUMERgrid_cropleft = erad_hrtsa_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
    
    # Right
    idx_right_sumer_0, idx_right_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_scalefactor_right)
    lam_sumer_cropright = lam_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
    rad_sumer_cropright = rad_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
    erad_sumer_cropright = erad_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
    rad_hrtsa_conv_SUMERgrid_cropright = rad_hrtsa_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
    erad_hrtsa_conv_SUMERgrid_cropright = erad_hrtsa_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
    
    
    # Concatenate left and right
    lam_sumer_cropscale  = np.concatenate([lam_sumer_cropleft,  lam_sumer_cropright])
    rad_sumer_cropscale  = np.concatenate([rad_sumer_cropleft,  rad_sumer_cropright])
    erad_sumer_cropscale = np.concatenate([erad_sumer_cropleft, erad_sumer_cropright])
    rad_hrtsa_conv_SUMERgrid_cropscale  = np.concatenate([rad_hrtsa_conv_SUMERgrid_cropleft, rad_hrtsa_conv_SUMERgrid_cropright])
    erad_hrtsa_conv_SUMERgrid_cropscale  = np.concatenate([erad_hrtsa_conv_SUMERgrid_cropleft, erad_hrtsa_conv_SUMERgrid_cropright])
    
    ######################################################
    # 
    
    ################
    # 3.2) Calculate the factor with a linear fit (and zero intercept). Fit straight line to the radiance of HRTS vs SUMER (y = m*x + 0 (intercept = 0))
    
    y_hrtsa = rad_hrtsa_conv_SUMERgrid_cropscale
    yerr_hrtsa = erad_hrtsa_conv_SUMERgrid_cropscale
    y_sumer = rad_sumer_cropscale
    yerr_sumer = erad_sumer_cropscale
    popt_sf, pcov_sf = curve_fit(linear_func, y_hrtsa, y_sumer, sigma=yerr_sumer, absolute_sigma=True)
    
    scaling_factor_ = popt_sf[0]
    scaling_factor_err = np.sqrt(pcov_sf[0, 0])
    
    # Compute reduced chi^2
    y_model = linear_func(y_hrtsa, *popt_sf) # Model prediction 
    chi2_sf = np.sum(((y_sumer - y_model) / yerr_sumer) ** 2) # Chi-square
    dof_sf = len(y_sumer) - len(popt_sf) # Degrees of freedom
    chi2_red_sf = chi2_sf / dof_sf # Reduced chi-square
    
    print("Scaling factor SUMER/HRTS:", scaling_factor_)
    print("Scaling factor error     :", scaling_factor_err)
    print("Reduced chi-square       :", chi2_red_sf)
    
    
    # Generate fitted line for plotting
    xfit_sf = np.linspace(min(y_hrtsa), max(y_hrtsa), 1000)
    yfit_sf = scaling_factor_ * xfit_sf #or linear_func(x_=xfit_sf, m_=scaling_factor_), they're the same
    
    
    # Scale HRTS and crop in range of analysis
    ## Radiance
    rad_hrtsa_conv_scaled = scaling_factor_ * rad_hrtsa_conv
    rad_hrtsa_conv_scaled_SUMERgrid = scaling_factor_ * rad_hrtsa_conv_SUMERgrid
    ## Uncertainties
    erad_hrtsa_conv_scaled = error_propagation_product(a_=scaling_factor_, a_err=scaling_factor_err, b_=rad_hrtsa_conv, b_err=erad_hrtsa_conv)
    erad_hrtsa_conv_scaled_SUMERgrid = error_propagation_product(a_=scaling_factor_, a_err=scaling_factor_err, b_=rad_hrtsa_conv_SUMERgrid, b_err=erad_hrtsa_conv_SUMERgrid)
    
    ######################################################
    # 4) Crop in the range of analysis
    
    
    # Crop SUMER and HRTS (interpolated to SUMER grid)
    rad_hrtsa_conv_SUMERgrid_cropNeVIII = rad_hrtsa_conv_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    erad_hrtsa_conv_SUMERgrid_cropNeVIII = erad_hrtsa_conv_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    rad_hrtsa_conv_scaled_SUMERgrid_cropNeVIII = rad_hrtsa_conv_scaled_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    erad_hrtsa_conv_scaled_SUMERgrid_cropNeVIII = erad_hrtsa_conv_scaled_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    
    # Crop HRTS original (nor interpolated to SUMER grid)
    idx_NeVIII_hrtsa_0, idx_NeVIII_hrtsa_1 = indices_closer_to_range(arr_1d=lam_hrtsa, range_=wavelength_range_to_analyze_NeVIII)
    lam_hrtsa_cropNeVIII = lam_hrtsa[idx_NeVIII_hrtsa_0:idx_NeVIII_hrtsa_1+1]
    rad_hrtsa_conv_scaled_cropNeVIII = rad_hrtsa_conv_scaled[idx_NeVIII_hrtsa_0:idx_NeVIII_hrtsa_1+1]
    erad_hrtsa_conv_scaled_cropNeVIII = erad_hrtsa_conv_scaled[idx_NeVIII_hrtsa_0:idx_NeVIII_hrtsa_1+1]
    
    ######################################################
    # 5) Subtract HRTS
    
    rad_sumer_cropNeVIII_corrected_qra = rad_sumer_cropNeVIII_uncorrected - rad_hrtsa_conv_scaled_SUMERgrid_cropNeVIII
    erad_sumer_cropNeVIII_corrected_qra = np.sqrt(erad_sumer_cropNeVIII_uncorrected**2 + erad_hrtsa_conv_scaled_SUMERgrid_cropNeVIII**2) #TODO: we don't have uncertainties in the HRTS spectrum


    ######################################################
    # 6) calculate peak
    mpi_corrected_qra = find_maximum_by_parabolic_interpolation_adapted(wavelength=lam_sumer_cropNeVIII, radiance=rad_sumer_cropNeVIII_corrected_qra, radiance_unc=erad_sumer_cropNeVIII_corrected_qra, show_figure='yes')
    mpi_corrected_qra["v_vertex"] = vkms_doppler(lamb=mpi_corrected_qra["x_vertex"], lamb_0=lam_0) #convert wavelength to speed
    mpi_corrected_qra["v_unc_vertex"] = vkms_doppler_unc(lamb=mpi_corrected_qra["x_vertex"], lamb_unc=mpi_corrected_qra["x_unc_vertex"], lamb_0=lam_0, lamb_0_unc=lam_unc_0) 

    ######################################################
    ######################################################
    ######################################################

    ######################################################
    ######################################################
    ######################################################
    # 1) Convolve HRTS spectrum with SUMER instrumental profile (gaussian profile of FWHM fwhm_conv)
    
    from scipy.ndimage import convolve1d
    
    inst_fwhm_HRTSpx = convolution_FWHM_in_pixels(wavelength=lam_hrts, fwhm_wavelength=fwhm_conv) #FWHM in pixels
    inst_sigma_HRTSpx = inst_fwhm_HRTSpx/(2*np.sqrt(2*np.log(2))) # convert FWHM to sigma (pixels)
    
    # build kernel
    half_width_ = int(5*inst_sigma_HRTSpx) #Using ±5σ captures essentially all the Gaussian
    xk = np.arange(-half_width_, half_width_+1)
    kernel_ = np.exp(-0.5*(xk/inst_sigma_HRTSpx)**2)
    kernel_ /= kernel_.sum() #Properly normalized kernel
    
    # convolve flux
    rad_hrtsb_conv = convolve1d(rad_hrtsb, kernel_, mode='constant', cval=0)
    
    # propagate variance properly
    varrad_hrtsb_conv = convolve1d(erad_hrtsb**2, kernel_**2, mode='constant', cval=0)
    erad_hrtsb_conv = np.sqrt(varrad_hrtsb_conv)
    
    
    ######################################################
    # 2) Create the interpolation function of the entire HRTS spectrum (after convolvolution)
    
    from scipy.interpolate import interp1d
    
    # Radiances
    rad_hrtsb_conv = np.ma.filled(rad_hrtsb_conv, np.nan) # Convert masked values to NaNs
    interp_func_hrtsb = interp1d(lam_hrtsb, rad_hrtsb_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
    rad_hrtsb_conv_SUMERgrid = interp_func_hrtsb(lam_sumer)
    
    # Uncertainties of the radiance
    erad_hrtsb_conv = np.ma.filled(erad_hrtsb_conv, np.nan) # Convert masked values to NaNs
    interp_func_hrtsb_err = interp1d(lam_hrtsb, erad_hrtsb_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
    erad_hrtsb_conv_SUMERgrid = interp_func_hrtsb_err(lam_sumer)
    
    ######################################################
    # 3) Scaling factor SUMER/HRTS. Crop regions at left and right of Ne VIII line to calculate the scaling factor of HRTS
    
    ################
    # 3.1) Select range(s) of wavelength
    
    # Left
    idx_left_sumer_0, idx_left_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_scalefactor_left)
    lam_sumer_cropleft = lam_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
    rad_sumer_cropleft = rad_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
    erad_sumer_cropleft = erad_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
    rad_hrtsb_conv_SUMERgrid_cropleft = rad_hrtsb_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
    erad_hrtsb_conv_SUMERgrid_cropleft = erad_hrtsb_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
    
    # Right
    idx_right_sumer_0, idx_right_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_scalefactor_right)
    lam_sumer_cropright = lam_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
    rad_sumer_cropright = rad_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
    erad_sumer_cropright = erad_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
    rad_hrtsb_conv_SUMERgrid_cropright = rad_hrtsb_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
    erad_hrtsb_conv_SUMERgrid_cropright = erad_hrtsb_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
    
    
    # Concatenate left and right
    lam_sumer_cropscale  = np.concatenate([lam_sumer_cropleft,  lam_sumer_cropright])
    rad_sumer_cropscale  = np.concatenate([rad_sumer_cropleft,  rad_sumer_cropright])
    erad_sumer_cropscale = np.concatenate([erad_sumer_cropleft, erad_sumer_cropright])
    rad_hrtsb_conv_SUMERgrid_cropscale  = np.concatenate([rad_hrtsb_conv_SUMERgrid_cropleft, rad_hrtsb_conv_SUMERgrid_cropright])
    erad_hrtsb_conv_SUMERgrid_cropscale  = np.concatenate([erad_hrtsb_conv_SUMERgrid_cropleft, erad_hrtsb_conv_SUMERgrid_cropright])
    
    ######################################################
    # 
    
    ################
    # 3.2) Calculate the factor with a linear fit (and zero intercept). Fit straight line to the radiance of HRTS vs SUMER (y = m*x + 0 (intercept = 0))
    
    y_hrtsb = rad_hrtsb_conv_SUMERgrid_cropscale
    yerr_hrtsb = erad_hrtsb_conv_SUMERgrid_cropscale
    y_sumer = rad_sumer_cropscale
    yerr_sumer = erad_sumer_cropscale
    popt_sf, pcov_sf = curve_fit(linear_func, y_hrtsb, y_sumer, sigma=yerr_sumer, absolute_sigma=True)
    
    scaling_factor_ = popt_sf[0]
    scaling_factor_err = np.sqrt(pcov_sf[0, 0])
    
    # Compute reduced chi^2
    y_model = linear_func(y_hrtsb, *popt_sf) # Model prediction 
    chi2_sf = np.sum(((y_sumer - y_model) / yerr_sumer) ** 2) # Chi-square
    dof_sf = len(y_sumer) - len(popt_sf) # Degrees of freedom
    chi2_red_sf = chi2_sf / dof_sf # Reduced chi-square
    
    print("Scaling factor SUMER/HRTS:", scaling_factor_)
    print("Scaling factor error     :", scaling_factor_err)
    print("Reduced chi-square       :", chi2_red_sf)
    
    
    # Generate fitted line for plotting
    xfit_sf = np.linspace(min(y_hrtsb), max(y_hrtsb), 1000)
    yfit_sf = scaling_factor_ * xfit_sf #or linear_func(x_=xfit_sf, m_=scaling_factor_), they're the same
    
    
    # Scale HRTS and crop in range of analysis
    ## Radiance
    rad_hrtsb_conv_scaled = scaling_factor_ * rad_hrtsb_conv
    rad_hrtsb_conv_scaled_SUMERgrid = scaling_factor_ * rad_hrtsb_conv_SUMERgrid
    ## Uncertainties
    erad_hrtsb_conv_scaled = error_propagation_product(a_=scaling_factor_, a_err=scaling_factor_err, b_=rad_hrtsb_conv, b_err=erad_hrtsb_conv)
    erad_hrtsb_conv_scaled_SUMERgrid = error_propagation_product(a_=scaling_factor_, a_err=scaling_factor_err, b_=rad_hrtsb_conv_SUMERgrid, b_err=erad_hrtsb_conv_SUMERgrid)
    
    ######################################################
    # 4) Crop in the range of analysis
    
    
    # Crop SUMER and HRTS (interpolated to SUMER grid)
    rad_hrtsb_conv_SUMERgrid_cropNeVIII = rad_hrtsb_conv_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    erad_hrtsb_conv_SUMERgrid_cropNeVIII = erad_hrtsb_conv_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    rad_hrtsb_conv_scaled_SUMERgrid_cropNeVIII = rad_hrtsb_conv_scaled_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    erad_hrtsb_conv_scaled_SUMERgrid_cropNeVIII = erad_hrtsb_conv_scaled_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    
    # Crop HRTS original (nor interpolated to SUMER grid)
    idx_NeVIII_hrtsb_0, idx_NeVIII_hrtsb_1 = indices_closer_to_range(arr_1d=lam_hrtsb, range_=wavelength_range_to_analyze_NeVIII)
    lam_hrtsb_cropNeVIII = lam_hrtsb[idx_NeVIII_hrtsb_0:idx_NeVIII_hrtsb_1+1]
    rad_hrtsb_conv_scaled_cropNeVIII = rad_hrtsb_conv_scaled[idx_NeVIII_hrtsb_0:idx_NeVIII_hrtsb_1+1]
    erad_hrtsb_conv_scaled_cropNeVIII = erad_hrtsb_conv_scaled[idx_NeVIII_hrtsb_0:idx_NeVIII_hrtsb_1+1]
    
    ######################################################
    # 5) Subtract HRTS
    
    rad_sumer_cropNeVIII_corrected_qrb = rad_sumer_cropNeVIII_uncorrected - rad_hrtsb_conv_scaled_SUMERgrid_cropNeVIII
    erad_sumer_cropNeVIII_corrected_qrb = np.sqrt(erad_sumer_cropNeVIII_uncorrected**2 + erad_hrtsb_conv_scaled_SUMERgrid_cropNeVIII**2) #TODO: we don't have uncertainties in the HRTS spectrum


    ######################################################
    # 6) calculate peak
    mpi_corrected_qrb = find_maximum_by_parabolic_interpolation_adapted(wavelength=lam_sumer_cropNeVIII, radiance=rad_sumer_cropNeVIII_corrected_qrb, radiance_unc=erad_sumer_cropNeVIII_corrected_qrb, show_figure='yes')
    mpi_corrected_qrb["v_vertex"] = vkms_doppler(lamb=mpi_corrected_qrb["x_vertex"], lamb_0=lam_0) #convert wavelength to speed
    mpi_corrected_qrb["v_unc_vertex"] = vkms_doppler_unc(lamb=mpi_corrected_qrb["x_vertex"], lamb_unc=mpi_corrected_qrb["x_unc_vertex"], lamb_0=lam_0, lamb_0_unc=lam_unc_0) 

    ######################################################
    ######################################################
    ######################################################

    ######################################################
    ######################################################
    ######################################################
    # 1) Convolve HRTS spectrum with SUMER instrumental profile (gaussian profile of FWHM fwhm_conv)
    
    from scipy.ndimage import convolve1d
    
    inst_fwhm_HRTSpx = convolution_FWHM_in_pixels(wavelength=lam_hrts, fwhm_wavelength=fwhm_conv) #FWHM in pixels
    inst_sigma_HRTSpx = inst_fwhm_HRTSpx/(2*np.sqrt(2*np.log(2))) # convert FWHM to sigma (pixels)
    
    # build kernel
    half_width_ = int(5*inst_sigma_HRTSpx) #Using ±5σ captures essentially all the Gaussian
    xk = np.arange(-half_width_, half_width_+1)
    kernel_ = np.exp(-0.5*(xk/inst_sigma_HRTSpx)**2)
    kernel_ /= kernel_.sum() #Properly normalized kernel
    
    # convolve flux
    rad_hrtsl_conv = convolve1d(rad_hrtsl, kernel_, mode='constant', cval=0)
    
    # propagate variance properly
    varrad_hrtsl_conv = convolve1d(erad_hrtsl**2, kernel_**2, mode='constant', cval=0)
    erad_hrtsl_conv = np.sqrt(varrad_hrtsl_conv)
    
    
    ######################################################
    # 2) Create the interpolation function of the entire HRTS spectrum (after convolvolution)
    
    from scipy.interpolate import interp1d
    
    # Radiances
    rad_hrtsl_conv = np.ma.filled(rad_hrtsl_conv, np.nan) # Convert masked values to NaNs
    interp_func_hrtsl = interp1d(lam_hrtsl, rad_hrtsl_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
    rad_hrtsl_conv_SUMERgrid = interp_func_hrtsl(lam_sumer)
    
    # Uncertainties of the radiance
    erad_hrtsl_conv = np.ma.filled(erad_hrtsl_conv, np.nan) # Convert masked values to NaNs
    interp_func_hrtsl_err = interp1d(lam_hrtsl, erad_hrtsl_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
    erad_hrtsl_conv_SUMERgrid = interp_func_hrtsl_err(lam_sumer)
    
    ######################################################
    # 3) Scaling factor SUMER/HRTS. Crop regions at left and right of Ne VIII line to calculate the scaling factor of HRTS
    
    ################
    # 3.1) Select range(s) of wavelength
    
    # Left
    idx_left_sumer_0, idx_left_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_scalefactor_left)
    lam_sumer_cropleft = lam_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
    rad_sumer_cropleft = rad_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
    erad_sumer_cropleft = erad_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
    rad_hrtsl_conv_SUMERgrid_cropleft = rad_hrtsl_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
    erad_hrtsl_conv_SUMERgrid_cropleft = erad_hrtsl_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
    
    # Right
    idx_right_sumer_0, idx_right_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_scalefactor_right)
    lam_sumer_cropright = lam_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
    rad_sumer_cropright = rad_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
    erad_sumer_cropright = erad_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
    rad_hrtsl_conv_SUMERgrid_cropright = rad_hrtsl_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
    erad_hrtsl_conv_SUMERgrid_cropright = erad_hrtsl_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
    
    
    # Concatenate left and right
    lam_sumer_cropscale  = np.concatenate([lam_sumer_cropleft,  lam_sumer_cropright])
    rad_sumer_cropscale  = np.concatenate([rad_sumer_cropleft,  rad_sumer_cropright])
    erad_sumer_cropscale = np.concatenate([erad_sumer_cropleft, erad_sumer_cropright])
    rad_hrtsl_conv_SUMERgrid_cropscale  = np.concatenate([rad_hrtsl_conv_SUMERgrid_cropleft, rad_hrtsl_conv_SUMERgrid_cropright])
    erad_hrtsl_conv_SUMERgrid_cropscale  = np.concatenate([erad_hrtsl_conv_SUMERgrid_cropleft, erad_hrtsl_conv_SUMERgrid_cropright])
    
    ######################################################
    # 
    
    ################
    # 3.2) Calculate the factor with a linear fit (and zero intercept). Fit straight line to the radiance of HRTS vs SUMER (y = m*x + 0 (intercept = 0))
    
    y_hrtsl = rad_hrtsl_conv_SUMERgrid_cropscale
    yerr_hrtsl = erad_hrtsl_conv_SUMERgrid_cropscale
    y_sumer = rad_sumer_cropscale
    yerr_sumer = erad_sumer_cropscale
    popt_sf, pcov_sf = curve_fit(linear_func, y_hrtsl, y_sumer, sigma=yerr_sumer, absolute_sigma=True)
    
    scaling_factor_ = popt_sf[0]
    scaling_factor_err = np.sqrt(pcov_sf[0, 0])
    
    # Compute reduced chi^2
    y_model = linear_func(y_hrtsl, *popt_sf) # Model prediction 
    chi2_sf = np.sum(((y_sumer - y_model) / yerr_sumer) ** 2) # Chi-square
    dof_sf = len(y_sumer) - len(popt_sf) # Degrees of freedom
    chi2_red_sf = chi2_sf / dof_sf # Reduced chi-square
    
    print("Scaling factor SUMER/HRTS:", scaling_factor_)
    print("Scaling factor error     :", scaling_factor_err)
    print("Reduced chi-square       :", chi2_red_sf)
    
    
    # Generate fitted line for plotting
    xfit_sf = np.linspace(min(y_hrtsl), max(y_hrtsl), 1000)
    yfit_sf = scaling_factor_ * xfit_sf #or linear_func(x_=xfit_sf, m_=scaling_factor_), they're the same
    
    
    # Scale HRTS and crop in range of analysis
    ## Radiance
    rad_hrtsl_conv_scaled = scaling_factor_ * rad_hrtsl_conv
    rad_hrtsl_conv_scaled_SUMERgrid = scaling_factor_ * rad_hrtsl_conv_SUMERgrid
    ## Uncertainties
    erad_hrtsl_conv_scaled = error_propagation_product(a_=scaling_factor_, a_err=scaling_factor_err, b_=rad_hrtsl_conv, b_err=erad_hrtsl_conv)
    erad_hrtsl_conv_scaled_SUMERgrid = error_propagation_product(a_=scaling_factor_, a_err=scaling_factor_err, b_=rad_hrtsl_conv_SUMERgrid, b_err=erad_hrtsl_conv_SUMERgrid)
    
    ######################################################
    # 4) Crop in the range of analysis
    
    
    # Crop SUMER and HRTS (interpolated to SUMER grid)
    rad_hrtsl_conv_SUMERgrid_cropNeVIII = rad_hrtsl_conv_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    erad_hrtsl_conv_SUMERgrid_cropNeVIII = erad_hrtsl_conv_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    rad_hrtsl_conv_scaled_SUMERgrid_cropNeVIII = rad_hrtsl_conv_scaled_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    erad_hrtsl_conv_scaled_SUMERgrid_cropNeVIII = erad_hrtsl_conv_scaled_SUMERgrid[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    
    # Crop HRTS original (nor interpolated to SUMER grid)
    idx_NeVIII_hrtsl_0, idx_NeVIII_hrtsl_1 = indices_closer_to_range(arr_1d=lam_hrtsl, range_=wavelength_range_to_analyze_NeVIII)
    lam_hrtsl_cropNeVIII = lam_hrtsl[idx_NeVIII_hrtsl_0:idx_NeVIII_hrtsl_1+1]
    rad_hrtsl_conv_scaled_cropNeVIII = rad_hrtsl_conv_scaled[idx_NeVIII_hrtsl_0:idx_NeVIII_hrtsl_1+1]
    erad_hrtsl_conv_scaled_cropNeVIII = erad_hrtsl_conv_scaled[idx_NeVIII_hrtsl_0:idx_NeVIII_hrtsl_1+1]
    
    ######################################################
    # 5) Subtract HRTS
    
    rad_sumer_cropNeVIII_corrected_qrl = rad_sumer_cropNeVIII_uncorrected - rad_hrtsl_conv_scaled_SUMERgrid_cropNeVIII
    erad_sumer_cropNeVIII_corrected_qrl = np.sqrt(erad_sumer_cropNeVIII_uncorrected**2 + erad_hrtsl_conv_scaled_SUMERgrid_cropNeVIII**2) #TODO: we don't have uncertainties in the HRTS spectrum


    ######################################################
    # 6) calculate peak
    mpi_corrected_qrl = find_maximum_by_parabolic_interpolation_adapted(wavelength=lam_sumer_cropNeVIII, radiance=rad_sumer_cropNeVIII_corrected_qrl, radiance_unc=erad_sumer_cropNeVIII_corrected_qrl, show_figure='yes')
    mpi_corrected_qrl["v_vertex"] = vkms_doppler(lamb=mpi_corrected_qrl["x_vertex"], lamb_0=lam_0) #convert wavelength to speed
    mpi_corrected_qrl["v_unc_vertex"] = vkms_doppler_unc(lamb=mpi_corrected_qrl["x_vertex"], lamb_unc=mpi_corrected_qrl["x_unc_vertex"], lamb_0=lam_0, lamb_0_unc=lam_unc_0) 

    ######################################################
    ######################################################
    ######################################################
    
    # calculate maxima
    ## uncorrected data
    mpi_uncorrected = find_maximum_by_parabolic_interpolation_adapted(wavelength=lam_sumer_cropNeVIII, radiance=rad_sumer_cropNeVIII_uncorrected, radiance_unc=erad_sumer_cropNeVIII_uncorrected, show_figure='yes')
    mpi_uncorrected["v_vertex"] = vkms_doppler(lamb=mpi_uncorrected["x_vertex"], lamb_0=lam_0) #convert wavelength to speed
    mpi_uncorrected["v_unc_vertex"] = vkms_doppler_unc(lamb=mpi_uncorrected["x_vertex"], lamb_unc=mpi_uncorrected["x_unc_vertex"], lamb_0=lam_0, lamb_0_unc=lam_unc_0) 
    


    ########################### uncorrected')
    lam_peak_uncorrected_list.append(mpi_uncorrected["x_vertex"])
    elam_peak_uncorrected_list.append(mpi_uncorrected["x_unc_vertex"])
    v_peak_uncorrected_list.append(mpi_uncorrected["v_vertex"])
    ev_peak_uncorrected_list.append(mpi_uncorrected["v_unc_vertex"])
    ########################### corrected QR-A')
    lam_peak_corrected_qra_list.append(mpi_corrected_qra["x_vertex"])
    elam_peak_corrected_qra_list.append(mpi_corrected_qra["x_unc_vertex"])
    v_peak_corrected_qra_list.append(mpi_corrected_qra["v_vertex"])
    ev_peak_corrected_qra_list.append(mpi_corrected_qra["v_unc_vertex"])
    ########################### corrected QR-B')
    lam_peak_corrected_qrb_list.append(mpi_corrected_qrb["x_vertex"])
    elam_peak_corrected_qrb_list.append(mpi_corrected_qrb["x_unc_vertex"])
    v_peak_corrected_qrb_list.append(mpi_corrected_qrb["v_vertex"])
    ev_peak_corrected_qrb_list.append(mpi_corrected_qrb["v_unc_vertex"])
    ########################### corrected QR-B')
    lam_peak_corrected_qrl_list.append(mpi_corrected_qrl["x_vertex"])
    elam_peak_corrected_qrl_list.append(mpi_corrected_qrl["x_unc_vertex"])
    v_peak_corrected_qrl_list.append(mpi_corrected_qrl["v_vertex"])
    ev_peak_corrected_qrl_list.append(mpi_corrected_qrl["v_unc_vertex"])
    
    

    ######################################################
    ######################################################
    ######################################################


    ######################################################
    # Show image of the intensity map with the contours and the pixels inside the contours

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5.7))
    img = ax.imshow(intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer), extent=extent_sumer_px_image)
    cbar = fig.colorbar(img, ax=ax, pad=0.03)
    ax.set_title(f'SOHO/SUMER intensity map {line_center_label}, solar rotation NOT compensated')
    ax.set_xlabel('Helioprojective longitude (arcsec), rotation compensated')
    ax.set_ylabel('Helioprojective latitude (arcsec)')
    ax.axis('auto') # Ensures equal scaling of axis x and y
    ax.scatter(x=x_col_list_plot, y=y_row_list_plot, s=1, color='yellow')
    contour_upper = ax.contour(intensity_map_croplat[::-1], levels=[bound_i], colors='blue', linewidths=1, extent=extent_sumer_px_contours)
    legend_elements = [mlines.Line2D([],[],color='blue', label=f'{bound_i}')]
    plt.show(block=False)


    ######################################################
    ######################################################
    ######################################################
"""

    
    ######################################################
    # Show averaged spectra

    # Full wavelength range
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
    ax.errorbar(x=lam_sumer_av, y=rad_sumer_av, yerr=erad_sumer_av, color='blue', linewidth=1., label='SUMER data')
    ax.set_title(f'SOHO/SUMER, profile averaged', fontsize=18) 
    ax.set_xlabel('Wavelength (\u212B)', color='black', fontsize=16)
    ax.set_ylabel(f'Av. spectral radiance [W/sr/m^2/Angstroem]', color='black', fontsize=16)
    ax.axvline(lam_0, color='green', linewidth=1., label=f'Rest wavelength ({lam_0})'' \u212B')
    ax.axvspan(lam_0-lam_unc_0, lam_0+lam_unc_0, color='green', alpha=0.2)
    ax.legend()
    plt.show(block=False)


    # Wavelength range cropped around Ne VIII
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
    ax.errorbar(x=lam_sumer_cropNeVIII, y=rad_sumer_cropNeVIII_uncorrected, yerr=erad_sumer_cropNeVIII_uncorrected, color='blue', linewidth=1., label='SUMER data')
    ax.set_title(f'SOHO/SUMER, profile averaged', fontsize=18) 
    ax.set_xlabel('Wavelength (\u212B)', color='black', fontsize=16)
    ax.set_ylabel(f'Av. spectral radiance [W/sr/m^2/Angstroem]', color='black', fontsize=16)
    ax.axvline(lam_0, color='green', linewidth=1., label=f'Rest wavelength ({lam_0})'' \u212B')
    ax.axvspan(lam_0-lam_unc_0, lam_0+lam_unc_0, color='green', alpha=0.2)
    ax.legend()
    plt.show(block=False)


    ######################################################
    ######################################################
    ######################################################


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.errorbar(x=rad_integrated_sumer_mean_list, xerr=erad_integrated_sumer_mean_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='.', label='SUMER uncorrected')
    ax.errorbar(x=rad_integrated_sumer_mean_list, xerr=erad_integrated_sumer_mean_list, y=v_peak_corrected_list, yerr=ev_peak_corrected_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected')
    ax.set_title(sun_region_list[0], fontsize=18) 
    ax.set_xlabel(r'Spectral radiance integrated (W/sr/m$^2$)', color='black', fontsize=16)
    ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
    ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
    ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
    ax.legend()
    plt.show(block=False)
"""


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Spectral radiance peak (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.legend()
plt.show(block=False)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Spectral radiance peak (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_xscale('log')
ax.legend()
plt.show(block=False)


"""

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=rad_percentage_mean_sumer, xerr=erad_percentage_mean_sumer, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='.', label='SUMER uncorrected')
ax.errorbar(x=rad_percentage_mean_sumer, xerr=erad_percentage_mean_sumer, y=v_peak_corrected_list, yerr=ev_peak_corrected_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Percentage of the mean (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.legend()
plt.show(block=False)





fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
for i in range(len(v_sumer_cropNeVIII_list)):
    x_peak = v_peak_uncorrected_list[i]
    dx_peak = ev_peak_uncorrected_list[i]
    y_peak = rad_peak_uncorrected_list[i]
    dy_peak = erad_peak_uncorrected_list[i]
    x_profile = v_sumer_cropNeVIII_list[i]
    dx_profile = ev_sumer_cropNeVIII_list[i]
    y_profile = rad_sumer_cropNeVIII_uncorrected_list[i]
    dy_profile = erad_sumer_cropNeVIII_uncorrected_list[i]
    ax.errorbar(x=x_profile, xerr=dx_profile, y=y_profile, yerr=dy_profile, color=color_list[i], linewidth=1.0)#, label='SUMER corrected')
    ax.errorbar(x=x_peak, xerr=dx_peak, y=y_peak, yerr=dy_peak, color=color_list[i], linewidth=0., elinewidth=1.0, marker='^')
ax.errorbar(x=x_peak, xerr=dx_peak, y=y_peak, yerr=dy_peak, color='black', linewidth=0., elinewidth=1.0, marker='^', label='Peak')
ax.set_title(f'Profiles uncorrected', fontsize=18) 
ax.set_xlabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.set_ylabel(r'Spectral radiance (W/sr/m$^2$/''\u212B)', color='black', fontsize=16)
ax.axvline(x=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.legend()
plt.show(block=False)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
for i in range(len(v_sumer_cropNeVIII_list)):
    x_peak = v_peak_corrected_list[i]
    dx_peak = ev_peak_corrected_list[i]
    y_peak = rad_peak_corrected_list[i]
    dy_peak = erad_peak_corrected_list[i]
    x_profile = v_sumer_cropNeVIII_list[i]
    dx_profile = ev_sumer_cropNeVIII_list[i]
    y_profile = rad_sumer_cropNeVIII_corrected_list[i]
    dy_profile = erad_sumer_cropNeVIII_corrected_list[i]
    ax.errorbar(x=x_profile, xerr=dx_profile, y=y_profile, yerr=dy_profile, color=color_list[i], linewidth=1.0)#, label='SUMER corrected')
    ax.errorbar(x=x_peak, xerr=dx_peak, y=y_peak, yerr=dy_peak, color=color_list[i], linewidth=0., elinewidth=1.0, marker='s')
ax.errorbar(x=x_peak, xerr=dx_peak, y=y_peak, yerr=dy_peak, color='black', linewidth=0., elinewidth=1.0, marker='s', label='Peak')
ax.set_title(f'Profiles corrected', fontsize=18) 
ax.set_xlabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.set_ylabel(r'Spectral radiance (W/sr/m$^2$/''\u212B)', color='black', fontsize=16)
ax.axvline(x=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.legend()
plt.show(block=False)


"""




