# INPUTS

# Binning
bin_lat = 4
bin_lon = 1

save_scalefactor_map = 'no'

show_spectral_image_binned = 'yes'
show_scaling_factor_maps = 'yes'
show_analysis_scalingfactor__row_col = [62,100] #'no' or list of 2 integers e.g. [23,56]

filename_eit = 'SOHO_EIT_195_19991107T042103_L1.fits' #for the binning of the coordinates (where solar rotation has been corrected)

## Ranges of wavelength
wavelength_range_scalefactor_left = [1537.7, 1539.5] #Angstrom
wavelength_range_scalefactor_right = [1542., 1544.] #Angstrom

## FWHM of the cold lines in SUMER:
fwhm_mean_weighted_sumer =  0.17740
fwhm_std_sumer =  0.02094
fwhm_unc_weighted_sumer =  0.00131
fwhm_synthetic_Si = 0.03
fwhm_sumer_to_convolve = fwhm_mean_weighted_sumer - fwhm_synthetic_Si
fwhm_to_convolve = fwhm_sumer_to_convolve #Usser can addapt this value
fwhm_to_convolve = 1.95 * 0.04215


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
from utils.data_path import path_data_soho 
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
exec(open("bin_data_interpolated.py").read())

# Outputs:
"""
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
# Import HRTS spectra

# QR A
from data.hrts.data__qqr_a_xdr import lambda__qqr_a_xdr, radiance__qqr_a_xdr #[nm]
lam_hrtsa, rad_hrtsa = 10.*lambda__qqr_a_xdr, 0.1*radiance__qqr_a_xdr #[Angstrom]
erad_hrtsa = np.zeros(len(rad_hrtsa))
label_hrtsa = r'QR-A close to the disk center ($\cos \theta = 0.92 - 0.98$)'
# QR B
from data.hrts.data__qqr_b_xdr import lambda__qqr_b_xdr, radiance__qqr_b_xdr #[nm]
lam_hrtsb, rad_hrtsb = 10.*lambda__qqr_b_xdr, 0.1*radiance__qqr_b_xdr #[Angstrom]
erad_hrtsb = np.zeros(len(rad_hrtsb))
label_hrtsb = r'QR-B ($\cos \theta \sim 0.68$)'
# QR L
from data.hrts.data__qqr_l_xdr import lambda__qqr_l_xdr, radiance__qqr_l_xdr #[nm]
lam_hrtsl, rad_hrtsl = 10.*lambda__qqr_l_xdr, 0.1*radiance__qqr_l_xdr #[Angstrom]
erad_hrtsl = np.zeros(len(rad_hrtsl))
label_hrtsl = r'QR-L close to the solar limb ($\cos \theta \sim 0.18$)'

############################################################
# Map of scaling factors binned


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

######################################################
# 1) Convolve HRTS spectrum with SUMER instrumental profile (gaussian profile of FWHM fwhm_to_convolve)

from scipy.ndimage import convolve1d

inst_fwhm_HRTSpx = convolution_FWHM_in_pixels(wavelength=lam_hrtsa, fwhm_wavelength=fwhm_to_convolve) #FWHM in pixels
inst_sigma_HRTSpx = inst_fwhm_HRTSpx/(2*np.sqrt(2*np.log(2))) # convert FWHM to sigma (pixels)

# build kernel
half_width_ = int(5*inst_sigma_HRTSpx) #Using ±5σ captures essentially all the Gaussian
xk = np.arange(-half_width_, half_width_+1)
kernel_ = np.exp(-0.5*(xk/inst_sigma_HRTSpx)**2)
kernel_ /= kernel_.sum() #Properly normalized kernel

# convolve flux
rad_hrtsa_conv = convolve1d(rad_hrtsa, kernel_, mode='constant', cval=0)
rad_hrtsb_conv = convolve1d(rad_hrtsb, kernel_, mode='constant', cval=0)
rad_hrtsl_conv = convolve1d(rad_hrtsl, kernel_, mode='constant', cval=0)

# propagate variance properly
varrad_hrtsa_conv = convolve1d(erad_hrtsa**2, kernel_**2, mode='constant', cval=0)
varrad_hrtsb_conv = convolve1d(erad_hrtsb**2, kernel_**2, mode='constant', cval=0)
varrad_hrtsl_conv = convolve1d(erad_hrtsl**2, kernel_**2, mode='constant', cval=0)
erad_hrtsa_conv = np.sqrt(varrad_hrtsa_conv)
erad_hrtsb_conv = np.sqrt(varrad_hrtsb_conv)
erad_hrtsl_conv = np.sqrt(varrad_hrtsl_conv)


######################################################
# 2) Create the interpolation function of the entire HRTS spectrum (after convolvolution)

from scipy.interpolate import interp1d

# Radiances
rad_hrtsa_conv = np.ma.filled(rad_hrtsa_conv, np.nan) # Convert masked values to NaNs
rad_hrtsb_conv = np.ma.filled(rad_hrtsb_conv, np.nan)
rad_hrtsl_conv = np.ma.filled(rad_hrtsl_conv, np.nan)
interp_func_hrtsa = interp1d(lam_hrtsa, rad_hrtsa_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
interp_func_hrtsb = interp1d(lam_hrtsb, rad_hrtsb_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
interp_func_hrtsl = interp1d(lam_hrtsl, rad_hrtsl_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
rad_hrtsa_conv_SUMERgrid = interp_func_hrtsa(lam_sumer)
rad_hrtsb_conv_SUMERgrid = interp_func_hrtsb(lam_sumer)
rad_hrtsl_conv_SUMERgrid = interp_func_hrtsl(lam_sumer)

# Uncertainties of the radiance
erad_hrtsa_conv = np.ma.filled(erad_hrtsa_conv, np.nan) # Convert masked values to NaNs
erad_hrtsb_conv = np.ma.filled(erad_hrtsb_conv, np.nan)
erad_hrtsl_conv = np.ma.filled(erad_hrtsl_conv, np.nan)
interp_func_hrtsa_err = interp1d(lam_hrtsa, erad_hrtsa_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
interp_func_hrtsb_err = interp1d(lam_hrtsb, erad_hrtsb_conv, kind='linear', bounds_error=False, fill_value=np.nan)
interp_func_hrtsl_err = interp1d(lam_hrtsl, erad_hrtsl_conv, kind='linear', bounds_error=False, fill_value=np.nan)
erad_hrtsa_conv_SUMERgrid = interp_func_hrtsa_err(lam_sumer)
erad_hrtsb_conv_SUMERgrid = interp_func_hrtsb_err(lam_sumer)
erad_hrtsl_conv_SUMERgrid = interp_func_hrtsl_err(lam_sumer)

######################################################
# 3) Crop regions at left and right of Ne VIII line to calculate the scaling factor of HRTS. Here for lam_sumer, lam_hrts and rad_hrts

################
# 3.1) Select range(s) of wavelength

# Left
idx_left_sumer_0, idx_left_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_scalefactor_left)
lam_sumer_cropleft = lam_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
rad_hrtsa_conv_SUMERgrid_cropleft = rad_hrtsa_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
rad_hrtsb_conv_SUMERgrid_cropleft = rad_hrtsb_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
rad_hrtsl_conv_SUMERgrid_cropleft = rad_hrtsl_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
erad_hrtsa_conv_SUMERgrid_cropleft = erad_hrtsa_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
erad_hrtsb_conv_SUMERgrid_cropleft = erad_hrtsb_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
erad_hrtsl_conv_SUMERgrid_cropleft = erad_hrtsl_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]

# Right
idx_right_sumer_0, idx_right_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_scalefactor_right)
lam_sumer_cropright = lam_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
rad_hrtsa_conv_SUMERgrid_cropright = rad_hrtsa_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
rad_hrtsb_conv_SUMERgrid_cropright = rad_hrtsb_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
rad_hrtsl_conv_SUMERgrid_cropright = rad_hrtsl_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
erad_hrtsa_conv_SUMERgrid_cropright = erad_hrtsa_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
erad_hrtsb_conv_SUMERgrid_cropright = erad_hrtsb_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
erad_hrtsl_conv_SUMERgrid_cropright = erad_hrtsl_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]

# Concatenate left and right
lam_sumer_cropscale  = np.concatenate([lam_sumer_cropleft,  lam_sumer_cropright])
rad_hrtsa_conv_SUMERgrid_cropscale  = np.concatenate([rad_hrtsa_conv_SUMERgrid_cropleft, rad_hrtsa_conv_SUMERgrid_cropright])
rad_hrtsb_conv_SUMERgrid_cropscale  = np.concatenate([rad_hrtsb_conv_SUMERgrid_cropleft, rad_hrtsb_conv_SUMERgrid_cropright])
rad_hrtsl_conv_SUMERgrid_cropscale  = np.concatenate([rad_hrtsl_conv_SUMERgrid_cropleft, rad_hrtsl_conv_SUMERgrid_cropright])
erad_hrtsa_conv_SUMERgrid_cropscale  = np.concatenate([erad_hrtsa_conv_SUMERgrid_cropleft, erad_hrtsa_conv_SUMERgrid_cropright])
erad_hrtsb_conv_SUMERgrid_cropscale  = np.concatenate([erad_hrtsb_conv_SUMERgrid_cropleft, erad_hrtsb_conv_SUMERgrid_cropright])
erad_hrtsl_conv_SUMERgrid_cropscale  = np.concatenate([erad_hrtsl_conv_SUMERgrid_cropleft, erad_hrtsl_conv_SUMERgrid_cropright])


######################################################
# 

# Create the map of scaling factors and the reduced chi^2
## HRTS QR-A
scaling_factor_map_binned_qra, chi2red_map_binned_qra = get_factor_SUMER_HRTS__previous_HRTS_preparation(lam_sumer_=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_binned_list, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_binned_list, lam_hrts_=lam_hrtsa, rad_hrts_=rad_hrtsa, rad_hrts_conv_SUMERgrid_cropscale_=rad_hrtsa_conv_SUMERgrid_cropscale, fwhm_conv=fwhm_to_convolve, wavelength_range_left=wavelength_range_scalefactor_left, wavelength_range_right=wavelength_range_scalefactor_right, indices_left_sumer=[idx_left_sumer_0, idx_left_sumer_1], indices_right_sumer=[idx_right_sumer_0, idx_right_sumer_1], show__row_col=show_analysis_scalingfactor__row_col, y_scale='linear', show_legend='yes')
## HRTS QR-B
scaling_factor_map_binned_qrb, chi2red_map_binned_qrb = get_factor_SUMER_HRTS__previous_HRTS_preparation(lam_sumer_=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_binned_list, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_binned_list, lam_hrts_=lam_hrtsb, rad_hrts_=rad_hrtsb, rad_hrts_conv_SUMERgrid_cropscale_=rad_hrtsb_conv_SUMERgrid_cropscale, fwhm_conv=fwhm_to_convolve, wavelength_range_left=wavelength_range_scalefactor_left, wavelength_range_right=wavelength_range_scalefactor_right, indices_left_sumer=[idx_left_sumer_0, idx_left_sumer_1], indices_right_sumer=[idx_right_sumer_0, idx_right_sumer_1], show__row_col=show_analysis_scalingfactor__row_col, y_scale='linear', show_legend='yes')
## HRTS QR-L
scaling_factor_map_binned_qrl, chi2red_map_binned_qrl = get_factor_SUMER_HRTS__previous_HRTS_preparation(lam_sumer_=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_binned_list, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_binned_list, lam_hrts_=lam_hrtsl, rad_hrts_=rad_hrtsl, rad_hrts_conv_SUMERgrid_cropscale_=rad_hrtsl_conv_SUMERgrid_cropscale, fwhm_conv=fwhm_to_convolve, wavelength_range_left=wavelength_range_scalefactor_left, wavelength_range_right=wavelength_range_scalefactor_right, indices_left_sumer=[idx_left_sumer_0, idx_left_sumer_1], indices_right_sumer=[idx_right_sumer_0, idx_right_sumer_1], show__row_col=show_analysis_scalingfactor__row_col, y_scale='linear', show_legend='yes')


############################################################
# show maps

if show_scaling_factor_maps=='yes':
    
    ## HRTS QR-A
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(scaling_factor_map_binned_qra, norm=LogNorm(), cmap='Greys_r', aspect='auto')
    ax.set_title('Scaling factor map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Helioprojective longitude (pixels)', fontsize=17)
    ax.set_ylabel('Helioprojective latitude (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    plt.show(block=False)

    ## HRTS QR-B
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(scaling_factor_map_binned_qrb, norm=LogNorm(), cmap='Greys_r', aspect='auto')
    ax.set_title('Scaling factor map, HRTS: QR-B', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Helioprojective longitude (pixels)', fontsize=17)
    ax.set_ylabel('Helioprojective latitude (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    plt.show(block=False)

    ## HRTS QR-L
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(scaling_factor_map_binned_qrl, norm=LogNorm(), cmap='Greys_r', aspect='auto')
    ax.set_title('Scaling factor map, HRTS: QR-L', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Helioprojective longitude (pixels)', fontsize=17)
    ax.set_ylabel('Helioprojective latitude (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    plt.show(block=False)

############################################################
#

if save_scalefactor_map == 'yes': # Save scale factor maps of the 3 QRs
    filename_profile = f'scalefactor_map_interpolated_binned_lon{bin_lon}_lat{bin_lat}.npz'
    foldepath_profile = '../outputs/'
    np.savez(foldepath_profile+filename_profile, scaling_factor_map_binned_qra=scaling_factor_map_binned_qra, scaling_factor_map_binned_qrb=scaling_factor_map_binned_qrb, scaling_factor_map_binned_qrl=scaling_factor_map_binned_qrl)


"""
In order to load the intensity map in another file (or this one), do the next:

scalefactor_loaded_dic = np.load(f'scalefactor_map_interpolated_binned_lon{bin_lon}_lat{bin_lat}.npz')
scaling_factor_map_binned_qra = scalefactor_loaded_dic['scaling_factor_map_binned_qra'] #[W/sr/m^2] 2D-array
scaling_factor_map_binned_qrb = scalefactor_loaded_dic['scaling_factor_map_binned_qrb'] #[W/sr/m^2] 2D-array
scaling_factor_map_binned_qrl = scalefactor_loaded_dic['scaling_factor_map_binned_qrl'] #[W/sr/m^2] 2D-array
"""
""

############################################################
# 

print('--------------------------------------')
print('scaling_factor_map_binned_qra, scaling_factor_map_binned_qrb, scaling_factor_map_binned_qrl')
print('chi2red_map_binned_qra, chi2red_map_binned_qrb, chi2red_map_binned_qrl')
print('# Radiances')
print('rad_hrtsa_conv, rad_hrtsb_conv, rad_hrtsl_conv')
print('interp_func_hrtsa, interp_func_hrtsb, interp_func_hrtsl')
print('rad_hrtsa_conv_SUMERgrid, rad_hrtsb_conv_SUMERgrid, rad_hrtsl_conv_SUMERgrid')
print('# Uncertainties of the radiance')
print('erad_hrtsa_conv, erad_hrtsb_conv, erad_hrtsl_conv')
print('interp_func_hrtsa_err, interp_func_hrtsb_err, interp_func_hrtsl_err')
print('erad_hrtsa_conv_SUMERgrid, erad_hrtsb_conv_SUMERgrid, erad_hrtsl_conv_SUMERgrid')
print('# Left')
print('idx_left_sumer_0, idx_left_sumer_1')
print('lam_sumer_cropleft')
print('rad_hrtsa_conv_SUMERgrid_cropleft, rad_hrtsb_conv_SUMERgrid_cropleft, rad_hrtsl_conv_SUMERgrid_cropleft')
print('erad_hrtsa_conv_SUMERgrid_cropleft, erad_hrtsb_conv_SUMERgrid_cropleft, erad_hrtsl_conv_SUMERgrid_cropleft')
print('# Right')
print('idx_right_sumer_0, idx_right_sumer_1')
print('lam_sumer_cropright')
print('rad_hrtsa_conv_SUMERgrid_cropright, rad_hrtsb_conv_SUMERgrid_cropright, rad_hrtsl_conv_SUMERgrid_cropright')
print('erad_hrtsa_conv_SUMERgrid_cropright, erad_hrtsb_conv_SUMERgrid_cropright, erad_hrtsl_conv_SUMERgrid_cropright')
print('# Concatenate left and right')
print('lam_sumer_cropscale')
print('rad_hrtsa_conv_SUMERgrid_cropscale, rad_hrtsb_conv_SUMERgrid_cropscale, rad_hrtsl_conv_SUMERgrid_cropscale')
print('erad_hrtsa_conv_SUMERgrid_cropscale, erad_hrtsb_conv_SUMERgrid_cropscale, erad_hrtsl_conv_SUMERgrid_cropscale')
print('--------------------------------------')

############################################################
# 




