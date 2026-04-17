
# Threshold value: label (type) and range of percentageRange percentage of the threshold value
## range_percentage: [float1, float2]
## threshold_value_type: 'max', 'min', 'mean', 'median'
## instrument: 'eit_195', 'sumer_NeVIII'
#range_percentage, threshold_value_type, instrument_line = [0., 4.], 'max', 'eit_195'
#range_percentage, threshold_value_type, instrument_line = [0., 3.42], 'max', 'eit_195'
#range_percentage, threshold_value_type, instrument_line = [0., 4.], 'max', 'sumer_NeVIII'
#range_percentage, threshold_value_type, instrument_line = [0., 5.], 'max', 'eit_195'
#range_percentage, threshold_value_type, instrument_line = [0., 60.], 'mean', 'eit_195'

range_percentage, threshold_value_type, instrument_line = [0., 60.], 'mean', 'eit_195'


#  Inputs
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'

# Threshold value: label (type) and range of percentageRange percentage of the threshold value
#range_percentage, threshold_value_type = [0., 4.], 'max' #'max', 'min', 'mean', 'median'
range_percentage, threshold_value_type = [0., 60.], 'mean'


color_sumer_uncorrected = 'black'
color_hrts_qra = 'blue'
color_hrts_qrb = 'red'
color_hrts_qrl = 'green'

# Parameters of the individual gaussians fits
components_linestyle = '-'
components_linewidth = 1.2
components_color = 'green'
fig_size = (12, 6)

# 
axislabel_size = 13
title_size = 17
legend_size = 13

# Wavelength ranges to crop spectra
wavelength_range_to_average = [1531.1147, 1551.7688]
wavelength_range_to_analyze_NeVIII = [1540.2, 1541.4]


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

############################################################################################################
############################################################################################################
############################################################################################################
# Average profiles of the intensity bin

# Load the intensity map and uncertainties
intensitymap_loaded_dic = np.load('../outputs/intensity_map_'+line_label+'_interpolated.npz')
intensity_map = intensitymap_loaded_dic['intensity_map'] #2D-array
intensity_map_unc = intensitymap_loaded_dic['intensity_map_unc'] #2D-array
intensity_map_croplat = intensitymap_loaded_dic['intensity_map_croplat'] #2D-array
intensity_map_unc_croplat = intensitymap_loaded_dic['intensity_map_unc_croplat'] #2D-array
line_center_label = intensitymap_loaded_dic['line_center_label'] 
vmin_sumer, vmax_sumer = intensitymap_loaded_dic['vmin_vmax'] 

######################################################
# Rest wavelength
print('rest_wavelength_label =', rest_wavelength_label)
print('Rest wavelength Ne VIII (2nd order): (lam_0 +- lam_unc_0) =', lam_0, r'$\pm$', lam_unc_0, '\u212B')
print('Uncertainty of the rest wavelengh in km/s =', v_unc_0, 'km/s')


######################################################

# Define intensity bin
lower_bound, upper_bound = get_bounds(intensitymap_croplat=intensity_map_croplat, range_percentage=range_percentage, threshold_value_type=threshold_value_type)
print('lower_bound, upper_bound =', lower_bound, ',', upper_bound)


# rows and columns inside the intensity bin
rowscols_croplat = np.argwhere((intensity_map_croplat>=lower_bound) & (intensity_map_croplat<=upper_bound))
y_row_list_plot = rowscols_croplat[:,0] # convert the list of pairs [row, column] into 2 lists of rows and columns (for the scatterplot)
x_col_list_plot = rowscols_croplat[:,1]
print('Number of pixels detected:', len(rowscols_croplat))

# Extent in pixels 
extent_sumer_px_contours = [0., intensity_map_croplat.shape[1]-1, intensity_map_croplat.shape[0]-1, 0.]
extent_sumer_px_image = [-0.5, intensity_map_croplat.shape[1]-1+0.5, intensity_map_croplat.shape[0]-1+0.5, -0.5]

######################################################

# Import SUMER data interpolated (wavelength calibrated)
data_interpolated_loaded = np.load('../data/data_modified/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)
# Average spectra of the pixels selected
lam_sumer_av, elam_sumer_av, rad_sumer_av, erad_sumer_av = average_profiles_from_pixels_selected_from_interpolated_data(wavelength_range_=wavelength_range_to_average, data_interpolated_loaded_=data_interpolated_loaded, rows_cols_of_spectroheliogram_croplat=rowscols_croplat)

############################################################################################################
############################################################################################################
############################################################################################################
# Profile of SUMER uncorrected and corrected (using the 3 spectra of HRTS)

range_numbers_to_string = '__'.join(f"{x:.2f}".replace('.', '_').rstrip('0') if f"{x:.2f}"[-1] != '0' else f"{x:.1f}".replace('.', '_') for x in range_percentage) 
filename_averaged_spectrum = 'average_profile__' + range_numbers_to_string + '__' + threshold_value_type + '_of_'+instrument_line+'.npz'


## Import dictionary
profiles_loaded_dic = np.load('../outputs/'+filename_averaged_spectrum)

## HRTS wavelength and Doppler velicity
lam_hrtsa_cropNeVIII = profiles_loaded_dic['lam_hrtsa_cropNeVIII']
lam_hrtsb_cropNeVIII = profiles_loaded_dic['lam_hrtsb_cropNeVIII']
lam_hrtsl_cropNeVIII = profiles_loaded_dic['lam_hrtsl_cropNeVIII']
v_hrtsa_cropNeVIII = vkms_doppler(lamb=lam_hrtsa_cropNeVIII, lamb_0=lam_0)
v_hrtsb_cropNeVIII = vkms_doppler(lamb=lam_hrtsb_cropNeVIII, lamb_0=lam_0)
v_hrtsl_cropNeVIII = vkms_doppler(lamb=lam_hrtsl_cropNeVIII, lamb_0=lam_0)

## HRTS convolved with SUMER instrumental profile, and scaled to SUMER
rad_hrtsa_conv_scaled_cropNeVIII = profiles_loaded_dic['rad_hrtsa_conv_scaled_cropNeVIII']
rad_hrtsb_conv_scaled_cropNeVIII = profiles_loaded_dic['rad_hrtsb_conv_scaled_cropNeVIII']
rad_hrtsl_conv_scaled_cropNeVIII = profiles_loaded_dic['rad_hrtsl_conv_scaled_cropNeVIII']
erad_hrtsa_conv_scaled_cropNeVIII = profiles_loaded_dic['erad_hrtsa_conv_scaled_cropNeVIII']
erad_hrtsb_conv_scaled_cropNeVIII = profiles_loaded_dic['erad_hrtsb_conv_scaled_cropNeVIII']
erad_hrtsl_conv_scaled_cropNeVIII = profiles_loaded_dic['erad_hrtsl_conv_scaled_cropNeVIII']


## SUMER wavelength and Doppler velocity (respect to the rest wavelength of Ne VIII 770 in 2nd order)
lam_sumer_cropNeVIII = profiles_loaded_dic['lam_sumer_cropNeVIII'] #Angstrom
v_sumer_cropNeVIII = vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0)

## SUMER radiance uncorrected
rad_sumer_cropNeVIII = profiles_loaded_dic['rad_sumer_cropNeVIII']
erad_sumer_cropNeVIII = profiles_loaded_dic['erad_sumer_cropNeVIII']

## SUMER radiance corrected
### QS-A
rad_sumer_cropNeVIII_corrected_qra = profiles_loaded_dic['rad_sumer_cropNeVIII_corrected_qra']
erad_sumer_cropNeVIII_corrected_qra = profiles_loaded_dic['erad_sumer_cropNeVIII_corrected_qra']
### QS-B
rad_sumer_cropNeVIII_corrected_qrb = profiles_loaded_dic['rad_sumer_cropNeVIII_corrected_qrb']
erad_sumer_cropNeVIII_corrected_qrb = profiles_loaded_dic['erad_sumer_cropNeVIII_corrected_qrb']
### QS-L
rad_sumer_cropNeVIII_corrected_qrl = profiles_loaded_dic['rad_sumer_cropNeVIII_corrected_qrl']
erad_sumer_cropNeVIII_corrected_qrl = profiles_loaded_dic['erad_sumer_cropNeVIII_corrected_qrl']



######################################################
######################################################
######################################################
# PAPER plot

x_lims = [max(min(v_hrtsa_cropNeVIII), min(v_sumer_cropNeVIII)),      min(max(v_hrtsa_cropNeVIII), max(v_sumer_cropNeVIII))]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
## HRTS scaled and convolved
ax.errorbar(x=v_hrtsa_cropNeVIII, y=rad_hrtsa_conv_scaled_cropNeVIII, linestyle='--', linewidth=1.2, color=color_hrts_qra, label='HRTS QS-A')
ax.errorbar(x=v_hrtsb_cropNeVIII, y=rad_hrtsb_conv_scaled_cropNeVIII, linestyle='--', linewidth=1.2, color=color_hrts_qrb, label='HRTS QS-B')
## SUMER uncorrected
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, linestyle='-', marker='.', markersize=10, linewidth=1.5, color=color_sumer_uncorrected, label='SUMER uncorrected')
## SUMER corrected
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII_corrected_qra, yerr=erad_sumer_cropNeVIII_corrected_qra, linestyle='-', linewidth=1.2, color=color_hrts_qra, label='SUMER corrected, QS-A')
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII_corrected_qrb, yerr=erad_sumer_cropNeVIII_corrected_qrb, linestyle='-', linewidth=1.2, color=color_hrts_qrb, label='SUMER corrected, QS-B')
## 
ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, label='Rest wavelength: 770.428 \u212B')
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_title(f'SUMER spectrum of the CH uncorrected and corrected with HRTS', fontsize=title_size)
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=axislabel_size)
ax.legend(fontsize=legend_size)
ax.set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size) #'Wavelength (nm)'
ax.set_xlim(x_lims)
plt.tight_layout()
plt.show(block=False)

#TODO: why is the radiance in nm??

############################################################################################################
############################################################################################################
############################################################################################################
# Fitting

######################################################
# Initial parameters of the fitting


#average_profile__0_0__60_0__mean_of_eit_195.npz

wavelength_range_NeVIII = [1540.32, 1541.43]
x_lims_fits = [min(v_sumer_cropNeVIII), max(v_sumer_cropNeVIII)]

bckg_fit_uncorrected = 0.2 #HRST not subtracted
init_parameters_uncorrected = [bckg_fit_uncorrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
	0.3-bckg_fit_uncorrected, -54., 20.,
	1.07-bckg_fit_uncorrected, -10, 45.,
	0.5-bckg_fit_uncorrected, 15., 45.,
	0.25-bckg_fit_uncorrected, 97., 30.
	]

bckg_fit_corrected_qra = -0.3
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
	0.1-bckg_fit_corrected_qra, -60., 20.,
	0.5-bckg_fit_corrected_qra, -30., 20.,
	3.-bckg_fit_corrected_qra, 0.0, 50.,
	#1.5-bckg_fit_corrected_qra, 25., 30.
	]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
	0.04-bckg_fit_corrected_qrb, -78., 30.,
	#0.3-bckg_fit_corrected_qrb, -50., 30.,
	0.5-bckg_fit_corrected_qrb, -30., 40.,
	0.8-bckg_fit_corrected_qrb, 0.0, 50.,
	#1.5-bckg_fit_corrected_qrb, 33., 40.
	]

bckg_fit_corrected_qrl = 0.
init_parameters_corrected_qrl = [bckg_fit_corrected_qrl, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
	0.08-bckg_fit_corrected_qrl, -75., 20.,
	0.3-bckg_fit_corrected_qrl, -35., 30.,
	0.8-bckg_fit_corrected_qrl, 7., 50.,
	0.022-bckg_fit_corrected_qrl, 77., 30.
	]



"""
#average_profile__0_0__4_0__max_of_EIT_195.npz

wavelength_range_NeVIII = [1540.32, 1541.43]

bckg_fit_uncorrected = 0.15 #HRST not subtracted
init_parameters_uncorrected = [bckg_fit_uncorrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
	0.2-bckg_fit_uncorrected, -54., 20.,
	0.5-bckg_fit_uncorrected, -10, 45.,
	0.4-bckg_fit_uncorrected, 15., 45.,
	#0.215-bckg_fit_uncorrected, 97., 30.
	]

bckg_fit_corrected_qra = 0.
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
	0.1-bckg_fit_corrected_qra, -60., 20.,
	#0.1-bckg_fit_corrected_qra, -50., 20.,
	0.88-bckg_fit_corrected_qra, -1.0, 50.,
	#0.15-bckg_fit_corrected_qra, 80., 30.
	]

bckg_fit_corrected_qrb = -0.3
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
	#0.05-bckg_fit_corrected_qrb, -77., 30.,
	1.-bckg_fit_corrected_qrb, -40., 30.,
	3.25-bckg_fit_corrected_qrb, 0.0, 50.,
	1.5-bckg_fit_corrected_qrb, 33., 40.
	]
           
bckg_fit_corrected_qrl = -0.3
init_parameters_corrected_qrl = [bckg_fit_corrected_qrl, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
	0.39-bckg_fit_corrected_qrl, -75., 20.,
	0.93-bckg_fit_corrected_qrl, -35., 30.,
	3.-bckg_fit_corrected_qrl, 7., 35.,
	1.-bckg_fit_corrected_qrl, 37., 50.
	]
"""

######################################################
# Fit uncorrected Ne VIII line

x_uncorrected = lam_sumer_cropNeVIII
y_uncorrected = rad_sumer_cropNeVIII
y_unc_uncorrected = erad_sumer_cropNeVIII


# Perform the fit
popt, pcov = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_uncorrected, lamb_0=lam_0), y_uncorrected, p0=init_parameters_uncorrected, sigma=y_unc_uncorrected, absolute_sigma=True) #popt are the optimized parameters. pcov is the covariance matrix of the parameters. 
perr = np.sqrt(np.diag(pcov)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_uncorrected = np.linspace(min(vkms_doppler(lamb=x_uncorrected, lamb_0=lam_0)), max(vkms_doppler(lamb=x_uncorrected, lamb_0=lam_0)), 300)
y_fit_uncorrected = multigaussian_function_for_curvefit(x_fit_uncorrected, *popt)


# Residuals
y_residuals = y_uncorrected - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_uncorrected, lamb_0=lam_0), *popt)
y_unc_fit_length_uncorrected = multi_gaussian_function_uncertainties(B=popt, B_unc=perr, x=vkms_doppler(lamb=x_uncorrected, lamb_0=lam_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_uncorrected, lamb_0=lam_0))))
y_unc_residuals = np.sqrt(y_unc_uncorrected**2 + y_unc_fit_length_uncorrected**2)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=fig_size, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].errorbar(x=vkms_doppler(lamb=x_uncorrected, lamb_0=lam_0) ,y=y_uncorrected, yerr=y_unc_uncorrected, color=color_sumer_uncorrected, marker='o', linewidth=0, elinewidth=1., label='SUMER uncorrected')
ax[0].plot(x_fit_uncorrected, y_fit_uncorrected, color=color_sumer_uncorrected, linestyle='-', label='Fit', zorder=1) 
#ax[0].plot(vkms_doppler(lamb=x_fit_uncorrected_singlegauss, lamb_0=lam_0), y_fit_uncorrected_singlegauss, color='magenta', linestyle='-', label='Individual gaussian', zorder=1)

bckg_fit = popt[0]
color_singlegauss_list = ['purple', 'brown', 'darkblue', 'darkred']
color_singlegauss_list = 5*['grey']
N_gaussians = (len(popt)-1)//3
for n_gauss in range(N_gaussians):
    color_i = color_singlegauss_list[n_gauss]
    amplitude_fit = popt[3*n_gauss+1]
    mean_fit = popt[3*n_gauss+2]
    fwhm_fit = popt[3*n_gauss+3]
    print('amplitude_fit', amplitude_fit)
    print('mean_fit', mean_fit)
    print('fwhm_fit', fwhm_fit)
    sigma_fit = fwhm_fit / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    x_fit_corrected_singlegauss = np.linspace(mean_fit-sigma_fit*3., mean_fit+sigma_fit*3., 200)
    y_fit_corrected_singlegauss = gaussian_function_with_background(x=x_fit_corrected_singlegauss, bckg=bckg_fit, amplitude=amplitude_fit, mean=mean_fit, fwhm=fwhm_fit)
    ax[0].plot(x_fit_corrected_singlegauss, y_fit_corrected_singlegauss, color=components_color, linestyle=components_linestyle, linewidth=components_linewidth)#, label='Individual gaussians')
ax[0].plot([], [], color=components_color, linestyle=components_linestyle, linewidth=components_linewidth, label='Individual gaussians')


ax[0].axvline(x=0, color='black', linestyle='--', label='Rest wavelength: 770.428 \u212B')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].set_title(f'SUMER spectrum uncorrected, multigaussian fit', fontsize=title_size)
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=title_size)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=axislabel_size)
ax[0].legend(fontsize=legend_size)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_uncorrected, lamb_0=lam_0), y=y_residuals, yerr=y_unc_residuals, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)#'Wavelength (nm)'
ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
ax[0].set_xlim(x_lims_fits)
ax[1].set_xlim(x_lims_fits)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
plt.show(block=False)

           
           
######################################################
# Fit corrected Ne VIII line, QS-A

x_corrected_qra = lam_sumer_cropNeVIII
y_corrected_qra = rad_sumer_cropNeVIII_corrected_qra
y_unc_corrected_qra = erad_sumer_cropNeVIII_corrected_qra


# Perform the fit
popt_qra, pcov_qra = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), y_corrected_qra, p0=init_parameters_corrected_qra, sigma=y_unc_corrected_qra, absolute_sigma=True) #popt_qra are the optimized parameters. pcov_qra is the covariance matrix of the parameters. 
perr_qra = np.sqrt(np.diag(pcov_qra)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_corrected_qra = np.linspace(min(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0)), max(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0)), 300)
y_fit_corrected_qra = multigaussian_function_for_curvefit(x_fit_corrected_qra, *popt_qra)


# Residuals
y_residuals_qra = y_corrected_qra - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), *popt_qra)
y_unc_fit_length_corrected_qra = multi_gaussian_function_uncertainties(B=popt_qra, B_unc=perr_qra, x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0))))
y_unc_residuals_qra = np.sqrt(y_unc_corrected_qra**2 + y_unc_fit_length_corrected_qra**2)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=fig_size, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0) ,y=y_corrected_qra, yerr=y_unc_corrected_qra, color=color_hrts_qra, marker='o', linewidth=0, elinewidth=1., label='SUMER corrected QS-A')
ax[0].plot(x_fit_corrected_qra, y_fit_corrected_qra, color=color_hrts_qra, linestyle='-', label='Fit', zorder=1) 
#ax[0].plot(vkms_doppler(lamb=x_fit_corrected_qra_singlegauss, lamb_0=lam_0), y_fit_corrected_qra_singlegauss, color='magenta', linestyle='-', label='Individual gaussian', zorder=1)

bckg_fit = popt_qra[0]
color_singlegauss_list = ['purple', 'brown', 'darkblue', 'darkred']
color_singlegauss_list = 5*['grey']
N_gaussians = (len(popt_qra)-1)//3
for n_gauss in range(N_gaussians):
    color_i = color_singlegauss_list[n_gauss]
    amplitude_fit = popt_qra[3*n_gauss+1]
    mean_fit = popt_qra[3*n_gauss+2]
    fwhm_fit = popt_qra[3*n_gauss+3]
    print('amplitude_fit', amplitude_fit)
    print('mean_fit', mean_fit)
    print('fwhm_fit', fwhm_fit)
    sigma_fit = fwhm_fit / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    x_fit_corrected_qra_singlegauss = np.linspace(mean_fit-sigma_fit*3., mean_fit+sigma_fit*3., 200)
    y_fit_corrected_qra_singlegauss = gaussian_function_with_background(x=x_fit_corrected_qra_singlegauss, bckg=bckg_fit, amplitude=amplitude_fit, mean=mean_fit, fwhm=fwhm_fit)
    ax[0].plot(x_fit_corrected_qra_singlegauss, y_fit_corrected_qra_singlegauss, color=components_color, linestyle=components_linestyle, linewidth=components_linewidth)#, label='Individual gaussians')
ax[0].plot([], [], color=components_color, linestyle=components_linestyle, linewidth=components_linewidth, label='Individual gaussians')

ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII/2')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-A, multigaussian fit', fontsize=title_size)
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=title_size)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=axislabel_size)
ax[0].legend(fontsize=legend_size)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), y=y_residuals_qra, yerr=y_unc_residuals_qra, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)#'Wavelength (nm)'
ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
ax[0].set_xlim(x_lims_fits)
ax[1].set_xlim(x_lims_fits)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
plt.show(block=False)


######################################################
# Fit corrected Ne VIII line, QS-B

x_corrected_qrb = lam_sumer_cropNeVIII
y_corrected_qrb = rad_sumer_cropNeVIII_corrected_qrb
y_unc_corrected_qrb = erad_sumer_cropNeVIII_corrected_qrb


# Perform the fit
popt_qrb, pcov_qrb = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), y_corrected_qrb, p0=init_parameters_corrected_qrb, sigma=y_unc_corrected_qrb, absolute_sigma=True) #popt_qrb are the optimized parameters. pcov_qrb is the covariance matrix of the parameters. 
perr_qrb = np.sqrt(np.diag(pcov_qrb)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_corrected_qrb = np.linspace(min(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0)), max(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0)), 300)
y_fit_corrected_qrb = multigaussian_function_for_curvefit(x_fit_corrected_qrb, *popt_qrb)


# Residuals
y_residuals_qrb = y_corrected_qrb - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), *popt_qrb)
y_unc_fit_length_corrected_qrb = multi_gaussian_function_uncertainties(B=popt_qrb, B_unc=perr_qrb, x=vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0))))
y_unc_residuals_qrb = np.sqrt(y_unc_corrected_qrb**2 + y_unc_fit_length_corrected_qrb**2)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=fig_size, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0) ,y=y_corrected_qrb, yerr=y_unc_corrected_qrb, color=color_hrts_qrb, marker='o', linewidth=0, elinewidth=1., label='SUMER corrected QS-B')
ax[0].plot(x_fit_corrected_qrb, y_fit_corrected_qrb, color=color_hrts_qrb, linestyle='-', label='Fit', zorder=1) 
#ax[0].plot(vkms_doppler(lamb=x_fit_corrected_qrb_singlegauss, lamb_0=lam_0), y_fit_corrected_qrb_singlegauss, color='magenta', linestyle='-', label='Individual gaussian', zorder=1)

bckg_fit = popt_qrb[0]
color_singlegauss_list = ['purple', 'brown', 'darkblue', 'darkred']
color_singlegauss_list = 5*['grey']
N_gaussians = (len(popt_qrb)-1)//3
for n_gauss in range(N_gaussians):
    color_i = color_singlegauss_list[n_gauss]
    amplitude_fit = popt_qrb[3*n_gauss+1]
    mean_fit = popt_qrb[3*n_gauss+2]
    fwhm_fit = popt_qrb[3*n_gauss+3]
    print('amplitude_fit', amplitude_fit)
    print('mean_fit', mean_fit)
    print('fwhm_fit', fwhm_fit)
    sigma_fit = fwhm_fit / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    x_fit_corrected_qrb_singlegauss = np.linspace(mean_fit-sigma_fit*3., mean_fit+sigma_fit*3., 200)
    y_fit_corrected_qrb_singlegauss = gaussian_function_with_background(x=x_fit_corrected_qrb_singlegauss, bckg=bckg_fit, amplitude=amplitude_fit, mean=mean_fit, fwhm=fwhm_fit)
    ax[0].plot(x_fit_corrected_qrb_singlegauss, y_fit_corrected_qrb_singlegauss, color=components_color, linestyle=components_linestyle, linewidth=components_linewidth)#, label='Individual gaussians')
ax[0].plot([], [], color=components_color, linestyle=components_linestyle, linewidth=components_linewidth, label='Individual gaussians')

ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII/2')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-B, multigaussian fit', fontsize=title_size)
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=title_size)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=axislabel_size)
ax[0].legend(fontsize=legend_size)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), y=y_residuals_qrb, yerr=y_unc_residuals_qrb, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)#'Wavelength (nm)'
ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
ax[0].set_xlim(x_lims_fits)
ax[1].set_xlim(x_lims_fits)
plt.show(block=False)

######################################################
# Fit corrected Ne VIII line, QS-L

x_corrected_qrl = lam_sumer_cropNeVIII
y_corrected_qrl = rad_sumer_cropNeVIII_corrected_qrl
y_unc_corrected_qrl = erad_sumer_cropNeVIII_corrected_qrl


# Perform the fit
popt_qrl, pcov_qrl = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0), y_corrected_qrl, p0=init_parameters_corrected_qrl, sigma=y_unc_corrected_qrl, absolute_sigma=True) #popt_qrl are the optimized parameters. pcov_qrl is the covariance matrix of the parameters. 
perr_qrl = np.sqrt(np.diag(pcov_qrl)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_corrected_qrl = np.linspace(min(vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0)), max(vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0)), 300)
y_fit_corrected_qrl = multigaussian_function_for_curvefit(x_fit_corrected_qrl, *popt_qrl)


# Residuals
y_residuals_qrl = y_corrected_qrl - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0), *popt_qrl)
y_unc_fit_length_corrected_qrl = multi_gaussian_function_uncertainties(B=popt_qrl, B_unc=perr_qrl, x=vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0))))
y_unc_residuals_qrl = np.sqrt(y_unc_corrected_qrl**2 + y_unc_fit_length_corrected_qrl**2)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=fig_size, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0) ,y=y_corrected_qrl, yerr=y_unc_corrected_qrl, color=color_hrts_qrl, marker='o', linewidth=0, elinewidth=1., label='SUMER corrected QS-L')
ax[0].plot(x_fit_corrected_qrl, y_fit_corrected_qrl, color='green', linestyle='-', label='Fit', zorder=1) 
#ax[0].plot(vkms_doppler(lamb=x_fit_corrected_qrl_singlegauss, lamb_0=lam_0), y_fit_corrected_qrl_singlegauss, color='magenta', linestyle='-', label='Individual gaussian', zorder=1)

bckg_fit = popt_qrl[0]
color_singlegauss_list = ['purple', 'brown', 'darkblue', 'darkred']
color_singlegauss_list = 5*['grey']
N_gaussians = (len(popt_qrl)-1)//3
for n_gauss in range(N_gaussians):
    color_i = color_singlegauss_list[n_gauss]
    amplitude_fit = popt_qrl[3*n_gauss+1]
    mean_fit = popt_qrl[3*n_gauss+2]
    fwhm_fit = popt_qrl[3*n_gauss+3]
    print('amplitude_fit', amplitude_fit)
    print('mean_fit', mean_fit)
    print('fwhm_fit', fwhm_fit)
    sigma_fit = fwhm_fit / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    x_fit_corrected_qrl_singlegauss = np.linspace(mean_fit-sigma_fit*3., mean_fit+sigma_fit*3., 200)
    y_fit_corrected_qrl_singlegauss = gaussian_function_with_background(x=x_fit_corrected_qrl_singlegauss, bckg=bckg_fit, amplitude=amplitude_fit, mean=mean_fit, fwhm=fwhm_fit)
    ax[0].plot(x_fit_corrected_qrl_singlegauss, y_fit_corrected_qrl_singlegauss, color=components_color, linestyle=components_linestyle, linewidth=components_linewidth)#, label='Individual gaussians')
ax[0].plot([], [], color=components_color, linestyle=components_linestyle, linewidth=components_linewidth, label='Individual gaussians')

ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII/2')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-L, multigaussian fit', fontsize=title_size)
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=title_size)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=axislabel_size)
ax[0].legend(fontsize=legend_size)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0), y=y_residuals_qrl, yerr=y_unc_residuals_qrl, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)#'Wavelength (nm)'
ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
ax[0].set_xlim(x_lims_fits)
ax[1].set_xlim(x_lims_fits)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
plt.show(block=False)


#############################################


#xb_uncorrected, yb_uncorrected = find_bisector(x_data=x_uncorrected[3:-7], y_data=y_uncorrected[3:-7], y_unc_data=y_unc_uncorrected[3:-7], y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')

#xb_corrected, yb_corrected = find_bisector(x_data=x_corrected, y_data=y_corrected, y_unc_data=y_unc_corrected, y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')







