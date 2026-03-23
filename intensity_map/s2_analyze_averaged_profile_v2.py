
filename_averaged_spectrum = 'average_profile__0_0__4_0__max_of_sumer_NeVIII.npz'
filename_averaged_spectrum = 'average_profile__0_0__3_42__max_of_sumer_195___.npz'

#  Inputs
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'

# Threshold value: label (type) and range of percentageRange percentage of the threshold value
threshold_value_type = 'max' #'max', 'min', 'mean', 'median'
range_percentage = [0., 4.]

fwhm_conv = 1.95*0.04215 #Angstrom

color_sumer = 'blue'
color_hrts = 'green'
color_sumer_uncorrected = 'red'
color_sumer_corrected = 'blue'

sun_region = 'qqr_a'

# Wavelength ranges to crop spectra
wavelength_range_to_average = [1531.1147, 1551.7688]
wavelength_range_to_analyze_NeVIII = [1540.2, 1541.4]

## Ranges of wavelength
wavelength_range_scalefactor_left = [1537.7, 1539.5] #Angstrom
wavelength_range_scalefactor_right = [1542., 1544.] #Angstrom

# save average profile as .npy?
save_average_profile = 'no' 

show_plots_correction = 'no'

######################################################
# Initial parameters of the fitting

bckg_fit_uncorrected = 0.15 #HRST not subtracted
init_parameters_uncorrected = [bckg_fit_uncorrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
           0.2-bckg_fit_uncorrected, -54., 20.,
           0.5-bckg_fit_uncorrected, -10, 45.,
           0.4-bckg_fit_uncorrected, 15., 45.,
           #0.215-bckg_fit_uncorrected, 97., 30.
           ]

if sun_region=='qqr_a':
    wavelength_range_NeVIII = [1540.32, 1541.43]
    bckg_fit_corrected = -0.3
    init_parameters_corrected = [bckg_fit_corrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
                   0.1-bckg_fit_corrected, -60., 20.,
                   #0.1-bckg_fit_corrected, -50., 20.,
                   0.3-bckg_fit_corrected, 0.0, 50.,
                   #0.15-bckg_fit_corrected, 80., 30.
                   ]
               
elif sun_region=='qqr_b':
    wavelength_range_NeVIII = [1540.3, 1541.43]
    bckg_fit_corrected = -0.3
    init_parameters_corrected = [bckg_fit_corrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
                   1.-bckg_fit_corrected, -40., 30.,
                   3.25-bckg_fit_corrected, 0.0, 50.,
                   1.5-bckg_fit_corrected, 33., 40.
                   ]
                   
elif sun_region=='qqr_l':
    bckg_fit_corrected = -0.3
    init_parameters_corrected = [bckg_fit_corrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
                   0.39-bckg_fit_corrected, -75., 20.,
                   0.93-bckg_fit_corrected, -35., 30.,
                   3.-bckg_fit_corrected, 7., 35.,
                   1.-bckg_fit_corrected, 37., 50.
                   ]


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
intensitymap_loaded_dic = np.load('../data/data_modified/intensity_map_'+line_label+'_interpolated.npz')
intensity_map = intensitymap_loaded_dic['intensity_map'] #2D-array
intensity_map_unc = intensitymap_loaded_dic['intensity_map_unc'] #2D-array
intensity_map_croplat = intensitymap_loaded_dic['intensity_map_croplat'] #2D-array
intensity_map_unc_croplat = intensitymap_loaded_dic['intensity_map_unc_croplat'] #2D-array
line_center_label = intensitymap_loaded_dic['line_center_label'] 
vmin_sumer, vmax_sumer = intensitymap_loaded_dic['vmin_vmax'] 

######################################################

# Rest wavelength used
rest_wavelength_label = 'Peter_and_Judge_1999' #'SUMER_atlas', 'Peter_1998', 'Dammasch_1999', 'Peter_and_Judge_1999', 'Kelly_database'
lamb_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][0] #Angstrom
lamb_unc_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][1] #Angstrom
print('Rest wavelength Ne VIII (2nd order):', lamb_0, r'$\pm$', lamb_unc_0, '\u212B')

# uncertainty of the rest wavelength in km/s
v_unc_0 = vkms_doppler_unc(lamb=lamb_0, lamb_unc=lamb_unc_0, lamb_0=lamb_0, lamb_0_unc=lamb_unc_0) 


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
# Substract HRTS

profiles_loaded_dic = np.load('../data/data_modified/'+filename_averaged_spectrum)
lam_sumer_cropNeVIII = profiles_loaded_dic['lam_sumer_cropNeVIII'] #Angstrom
rad_sumer_cropNeVIII = profiles_loaded_dic['rad_sumer_cropNeVIII']
erad_sumer_cropNeVIII = profiles_loaded_dic['erad_sumer_cropNeVIII']
rad_sumer_cropNeVIII_corrected_qra = profiles_loaded_dic['rad_sumer_cropNeVIII_corrected_qra']
erad_sumer_cropNeVIII_corrected_qra = profiles_loaded_dic['erad_sumer_cropNeVIII_corrected_qra']
lam_hrtsa_cropNeVIII = profiles_loaded_dic['lam_hrtsa_cropNeVIII']
rad_hrtsa_conv_scaled_cropNeVIII = profiles_loaded_dic['rad_hrtsa_conv_scaled_cropNeVIII']
erad_hrtsa_conv_scaled_cropNeVIII = profiles_loaded_dic['erad_hrtsa_conv_scaled_cropNeVIII']
rad_sumer_cropNeVIII_corrected_qrb = profiles_loaded_dic['rad_sumer_cropNeVIII_corrected_qrb']
erad_sumer_cropNeVIII_corrected_qrb = profiles_loaded_dic['erad_sumer_cropNeVIII_corrected_qrb']
lam_hrtsb_cropNeVIII = profiles_loaded_dic['lam_hrtsb_cropNeVIII']
rad_hrtsb_conv_scaled_cropNeVIII = profiles_loaded_dic['rad_hrtsb_conv_scaled_cropNeVIII']
erad_hrtsb_conv_scaled_cropNeVIII = profiles_loaded_dic['erad_hrtsb_conv_scaled_cropNeVIII']
rad_sumer_cropNeVIII_corrected_qrl = profiles_loaded_dic['rad_sumer_cropNeVIII_corrected_qrl']
erad_sumer_cropNeVIII_corrected_qrl = profiles_loaded_dic['erad_sumer_cropNeVIII_corrected_qrl']
lam_hrtsl_cropNeVIII = profiles_loaded_dic['lam_hrtsl_cropNeVIII']
rad_hrtsl_conv_scaled_cropNeVIII = profiles_loaded_dic['rad_hrtsl_conv_scaled_cropNeVIII']
erad_hrtsl_conv_scaled_cropNeVIII = profiles_loaded_dic['erad_hrtsl_conv_scaled_cropNeVIII']


######################################################
######################################################
######################################################
# Fitting

           
######################################################
# Fit uncorrected Ne VIII line

x_uncorrected = lam_sumer_cropNeVIII
y_uncorrected = rad_sumer_cropNeVIII
y_unc_uncorrected = erad_sumer_cropNeVIII


# Perform the fit
popt, pcov = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0), y_uncorrected, p0=init_parameters_uncorrected, sigma=y_unc_uncorrected, absolute_sigma=True) #popt are the optimized parameters. pcov is the covariance matrix of the parameters. 
perr = np.sqrt(np.diag(pcov)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_uncorrected = np.linspace(min(vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0)), max(vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0)), 300)
y_fit_uncorrected = multigaussian_function_for_curvefit(x_fit_uncorrected, *popt)


# Residuals
y_residuals = y_uncorrected - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0), *popt)
y_unc_fit_length_uncorrected = multi_gaussian_function_uncertainties(B=popt, B_unc=perr, x=vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0))))
y_unc_residuals = np.sqrt(y_unc_uncorrected**2 + y_unc_fit_length_uncorrected**2)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].errorbar(x=vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0) ,y=y_uncorrected, yerr=y_unc_uncorrected, color=color_sumer_uncorrected, marker='o', linewidth=0, elinewidth=1., label='SUMER uncorrected')
ax[0].plot(x_fit_uncorrected, y_fit_uncorrected, color='orange', linestyle='-', label='Fit', zorder=1) 
#ax[0].plot(vkms_doppler(lamb=x_fit_uncorrected_singlegauss, lamb_0=lamb_0), y_fit_uncorrected_singlegauss, color='magenta', linestyle='-', label='Individual gaussian', zorder=1)

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
    ax[0].plot(x_fit_corrected_singlegauss, y_fit_corrected_singlegauss, color=color_i, linestyle=':')#, label='Individual gaussians')
ax[0].plot([], [], color='grey', linestyle=':', label='Individual gaussians')

ax[0].axvline(x=0, color='black', linestyle='--', label='Rest wavelength: 770.428 \u212B')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].set_title(f'SUMER spectrum lowest 4%, multigaussian fit', fontsize=18)
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=18)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=15)
ax[0].legend(fontsize=12)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0), y=y_residuals, yerr=y_unc_residuals, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
ax[1].set_ylabel('Residuals', fontsize=15)
plt.tight_layout()
plt.show(block=False)

           
           
######################################################
# Fit corrected Ne VIII line

if sun_region == 'qqr_a':
    x_corrected = lam_sumer_cropNeVIII
    y_corrected = rad_sumer_cropNeVIII_corrected_qra
    y_unc_corrected = erad_sumer_cropNeVIII_corrected_qra
if sun_region == 'qqr_b':
    x_corrected = lam_sumer_cropNeVIII
    y_corrected = rad_sumer_cropNeVIII_corrected_qrb
    y_unc_corrected = erad_sumer_cropNeVIII_corrected_qrb
elif sun_region == 'qqr_l':
    x_corrected = lam_sumer_cropNeVIII
    y_corrected = rad_sumer_cropNeVIII_corrected_qrl
    y_unc_corrected = erad_sumer_cropNeVIII_corrected_qrl


# Perform the fit
popt, pcov = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected, lamb_0=lamb_0), y_corrected, p0=init_parameters_corrected, sigma=y_unc_corrected, absolute_sigma=True) #popt are the optimized parameters. pcov is the covariance matrix of the parameters. 
perr = np.sqrt(np.diag(pcov)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_corrected = np.linspace(min(vkms_doppler(lamb=x_corrected, lamb_0=lamb_0)), max(vkms_doppler(lamb=x_corrected, lamb_0=lamb_0)), 300)
y_fit_corrected = multigaussian_function_for_curvefit(x_fit_corrected, *popt)


# Residuals
y_residuals = y_corrected - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected, lamb_0=lamb_0), *popt)
y_unc_fit_length_corrected = multi_gaussian_function_uncertainties(B=popt, B_unc=perr, x=vkms_doppler(lamb=x_corrected, lamb_0=lamb_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected, lamb_0=lamb_0))))
y_unc_residuals = np.sqrt(y_unc_corrected**2 + y_unc_fit_length_corrected**2)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected, lamb_0=lamb_0) ,y=y_corrected, yerr=y_unc_corrected, color=color_sumer_corrected, marker='o', linewidth=0, elinewidth=1., label='SUMER corrected')
ax[0].plot(x_fit_corrected, y_fit_corrected, color='green', linestyle='-', label='Fit', zorder=1) 
#ax[0].plot(vkms_doppler(lamb=x_fit_corrected_singlegauss, lamb_0=lamb_0), y_fit_corrected_singlegauss, color='magenta', linestyle='-', label='Individual gaussian', zorder=1)

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
    ax[0].plot(x_fit_corrected_singlegauss, y_fit_corrected_singlegauss, color=color_i, linestyle=':')#, label='Individual gaussians')
ax[0].plot([], [], color='grey', linestyle=':', label='Individual gaussians')

ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII/2')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].set_title(f'SUMER spectrum lowest 4%, multigaussian fit', fontsize=18)
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=18)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=15)
ax[0].legend(fontsize=12)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected, lamb_0=lamb_0), y=y_residuals, yerr=y_unc_residuals, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
ax[1].set_ylabel('Residuals', fontsize=15)
plt.tight_layout()
plt.show(block=False)



#############################################


xb_uncorrected, yb_uncorrected = find_bisector(x_data=x_uncorrected[3:-7], y_data=y_uncorrected[3:-7], y_unc_data=y_unc_uncorrected[3:-7], y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')

xb_corrected, yb_corrected = find_bisector(x_data=x_corrected, y_data=y_corrected, y_unc_data=y_unc_corrected, y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')







