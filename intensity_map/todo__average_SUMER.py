
#  Inputs
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'

# Threshold value: label (type) and range of percentageRange percentage of the threshold value
threshold_value_type = 'max' #'max', 'min', 'mean', 'median'
range_percentage = [0., 4.]

fwhm_conv = 1.95*0.04215 #Angstrom

hrts_qr = 'a' #'a', 'b', 'l'

color_sumer = 'blue'
color_hrts = 'green'
color_sumer_uncorrected = 'red'
color_sumer_corrected = 'blue'


# Wavelength ranges to crop spectra
wavelength_range_to_average = [1531.1147, 1551.7688]
wavelength_range_to_analyze_NeVIII = [1540.2, 1541.4]

## Ranges of wavelength
wavelength_range_scalefactor_left = [1537.7, 1539.5] #Angstrom
wavelength_range_scalefactor_right = [1542., 1544.] #Angstrom

# save average profile as .npy?
save_average_profile_map = 'no' 

show_plots_correction = 'no'

######################################################
# Initial parameters of the fitting

bckg_fit_uncorrected = 1.5 #HRST not subtracted
init_parameters_uncorrected = [bckg_fit_uncorrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
           2.-bckg_fit_uncorrected, -54., 20.,
           5.-bckg_fit_uncorrected, -10, 45.,
           4.-bckg_fit_uncorrected, 15., 45.,
           2.15-bckg_fit_uncorrected, 97., 30.
           ]

if hrts_qr=='a':
    wavelength_range_NeVIII = [1540.32, 1541.43]
    bckg_fit_corrected = -0.3
    init_parameters_corrected = [bckg_fit_corrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
                   0.2-bckg_fit_corrected, -60., 20.,
                   0.4-bckg_fit_corrected, 0.0, 50.,
                   #1.5-bckg_fit_corrected, 25., 30.
                   ]
               
elif hrts_qr=='b':
    wavelength_range_NeVIII = [1540.3, 1541.43]
    bckg_fit_corrected = -0.3
    init_parameters_corrected = [bckg_fit_corrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
                   0.1-bckg_fit_corrected, -40., 30.,
                   0.3-bckg_fit_corrected, 0.0, 50.,
                   0.15-bckg_fit_corrected, 33., 40.,
                   0.02-bckg_fit_corrected, -80., 40.
                   ]
                   
elif hrts_qr=='l':
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
from auxiliar_functions.data_path import path_data_soho 
from auxiliar_functions.SOHO_aux_functions import *
from auxiliar_functions.calibration_parameters__output import *
from auxiliar_functions.spectroheliogram_functions import *
from auxiliar_functions.solar_rotation_variables import *
from auxiliar_functions.aux_functions import *
from auxiliar_functions.general_variables import *
from auxiliar_functions.NeVIII_rest_wavelength import *
from scale_hrts import *

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
lamb_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][0] #Angstrom
lamb_unc_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][1] #Angstrom
print('Rest wavelength Ne VIII (2nd order):', lamb_0, r'$\pm$', lamb_unc_0, '\u212B')

# uncertainty of the rest wavelength in km/s
v_unc_0 = vkms_doppler_unc(lamb=lamb_0, lamb_unc=lamb_unc_0, lamb_0=lamb_0, lamb_0_unc=lamb_unc_0) 


######################################################

# Import SUMER data interpolated (wavelength calibrated)
data_interpolated_loaded = np.load('../auxiliar_functions/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)
spectral_image_interpolated_list = data_interpolated_loaded['spectral_image_interpolated_list']
spectral_image_unc_interpolated_list = data_interpolated_loaded['spectral_image_unc_interpolated_list']
lam_sumer = data_interpolated_loaded['reference_wavelength']          # scalar (0‑d array; use a_loaded.item() for Python float)
lam_sumer_unc = data_interpolated_loaded['unc_reference_wavelength'] #uncertainty of lam_sumer
row_reference = int(data_interpolated_loaded['row_reference'])        # becomes a NumPy array or object array, so I conver it to integer again

######################################################
######################################################
######################################################

# Define intensity bin
lower_bound, upper_bound = get_bounds(intensitymap_croplat=intensity_map_croplat, range_percentage=range_percentage, threshold_value_type=threshold_value_type)
print('lower_bound, upper_bound =', lower_bound, ',', upper_bound)


# rows and columns inside the intensity bin
rowscols_inside_range = range_intensity_addresses_of_SUMER_spectroheliogram(intensitymap_croplat=intensity_map_croplat, lower_bound=lower_bound, upper_bound=upper_bound, slit_top_px=0)
print('Number of pixels detected:', len(rowscols_inside_range))

# convert the list of pairs [row, column] into 2 lists of rows and columns (for the scatterplot)
y_row_list_plot, x_col_list_plot = convert_list_of_pairs_to_2_lists(list_of_pairs=rowscols_inside_range)

# Extent in pixels 
extent_sumer_px_contours = [0., intensity_map_croplat.shape[1]-1, intensity_map_croplat.shape[0]-1, 0.]
extent_sumer_px_image = [-0.5, intensity_map_croplat.shape[1]-1+0.5, intensity_map_croplat.shape[0]-1+0.5, -0.5]

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
contour_lower = ax.contour(intensity_map_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_px_contours)
contour_upper = ax.contour(intensity_map_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_px_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{lower_bound}'),
    mlines.Line2D([],[],color='blue', label=f'{upper_bound}')]
plt.show(block=False)


######################################################
######################################################
######################################################


# Average spectra of the pixels selected
lam_sumer_av, elam_sumer_av, rad_sumer_av, erad_sumer_av = average_profiles_from_pixels_selected_from_interpolated_data(wavelength_range_=wavelength_range_to_average, data_interpolated_loaded_=data_interpolated_loaded, rows_cols_of_spectroheliogram=rowscols_inside_range)

# crop near Ne VIII
lam_sumer_avNeVIII, idx_sumer_crop_ = crop_range(list_to_crop=lam_sumer_av, range_values=wavelength_range_to_analyze_NeVIII)
elam_sumer_avNeVIII = elam_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
rad_sumer_avNeVIII = rad_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
erad_sumer_avNeVIII = erad_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]


fsh = fun_scale_hrts(hrts_qr=hrts_qr, lamb_0=lamb_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_conv, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction)
lam_sumer_cropNeVIII = fsh['lam_sumer_cropNeVIII']
rad_sumer_cropNeVIII = fsh['rad_sumer_cropNeVIII']
erad_sumer_cropNeVIII = fsh['erad_sumer_cropNeVIII']
rad_sumer_cropNeVIII_corrected = fsh['rad_sumer_cropNeVIII_corrected']
erad_sumer_cropNeVIII_corrected = fsh['erad_sumer_cropNeVIII_corrected']
lam_hrts = fsh['lam_hrts']
rad_hrts = fsh['rad_hrts']
erad_hrts = fsh['erad_hrts']
rad_hrts_conv = fsh['rad_hrts_conv']
erad_hrts_conv = fsh['erad_hrts_conv']
rad_hrts_conv_scaled = fsh['rad_hrts_conv_scaled']
erad_hrts_conv_scaled = fsh['erad_hrts_conv_scaled']
lam_hrts_cropNeVIII = fsh['lam_hrts_cropNeVIII']
rad_hrts_cropNeVIII = fsh['rad_hrts_cropNeVIII']
erad_hrts_cropNeVIII = fsh['erad_hrts_cropNeVIII']
rad_hrts_conv_scaled_cropNeVIII = fsh['rad_hrts_conv_scaled_cropNeVIII']
erad_hrts_conv_scaled_cropNeVIII = fsh['erad_hrts_conv_scaled_cropNeVIII']


######################################################
# Show averaged spectra

# Full wavelength range
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
ax.errorbar(x=lam_sumer_av, y=rad_sumer_av, yerr=erad_sumer_av, color='blue', linewidth=1., label='SUMER data')
ax.set_title(f'SOHO/SUMER, profile averaged', fontsize=18) 
ax.set_xlabel('Wavelength (\u212B)', color='black', fontsize=16)
ax.set_ylabel(f'Av. spectral radiance [W/sr/m^2/Angstroem]', color='black', fontsize=16)
ax.axvline(lamb_0, color='green', linewidth=1., label=f'Rest wavelength ({lamb_0})'' \u212B')
ax.axvspan(lamb_0-lamb_unc_0, lamb_0+lamb_unc_0, color='green', alpha=0.2)
ax.legend()
plt.show(block=False)


# Wavelength range cropped around Ne VIII
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
ax.errorbar(x=lam_sumer_avNeVIII, y=rad_sumer_avNeVIII, yerr=erad_sumer_avNeVIII, color='blue', linewidth=1., label='SUMER data')
ax.set_title(f'SOHO/SUMER, profile averaged', fontsize=18) 
ax.set_xlabel('Wavelength (\u212B)', color='black', fontsize=16)
ax.set_ylabel(f'Av. spectral radiance [W/sr/m^2/Angstroem]', color='black', fontsize=16)
ax.axvline(lamb_0, color='green', linewidth=1., label=f'Rest wavelength ({lamb_0})'' \u212B')
ax.axvspan(lamb_0-lamb_unc_0, lamb_0+lamb_unc_0, color='green', alpha=0.2)
ax.legend()
plt.show(block=False)


# Plot: Comparison SUMER corrected and uncorrected
fig, ax = plt.subplots(figsize=(12, 5))
#ax.errorbar(x=vkms_doppler(lamb=lam_crop, lamb_0=lamb_0), y=rad_crop, yerr=erad_crop, color='black', linewidth=0.6, label='SUMER box') #Real spectrum (SUMER) 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER lowest {range_percentage}%, not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII_corrected, yerr=erad_sumer_cropNeVIII_corrected, color=color_sumer_corrected, linestyle='-', linewidth=2., label=f'SUMER {range_percentage} of the maximum%, corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_hrts_cropNeVIII, lamb_0=lamb_0), y=rad_hrts_conv_scaled_cropNeVIII, yerr=erad_hrts_conv_scaled_cropNeVIII, color='black', linestyle='--', linewidth=2., label='HRST - QR A') #Real spectrum (SUMER)
ax.axvline(x=0, color='brown', linestyle=':', linewidth=2., label='Rest wavelength of Ne VIII/2')
ax.axvspan(-v_unc_0, v_unc_0, color='brown', alpha=0.15)
ax.set_title(f'Comparison SUMER before and after correction with HRTS', fontsize=18)
ax.set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
ax.set_xlim([vkms_doppler(lamb=min(lam_hrts_cropNeVIII), lamb_0=lamb_0), vkms_doppler(lamb=max(lam_hrts_cropNeVIII), lamb_0=lamb_0)])
ax.legend(fontsize=12)
ax.set_yscale('linear')
plt.show(block=False)


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

ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII/2')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].set_title(f'SUMER {range_percentage}% of the average of the spectroheliogram, uncorrected', fontsize=18)
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

x_corrected = lam_sumer_cropNeVIII
y_corrected = rad_sumer_cropNeVIII_corrected
y_unc_corrected = erad_sumer_cropNeVIII_corrected


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
ax[0].set_title(f'SUMER {range_percentage}% of the average of the spectroheliogram, corrected', fontsize=18)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=15)
ax[0].legend(fontsize=12)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected, lamb_0=lamb_0), y=y_residuals, yerr=y_unc_residuals, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
ax[1].set_ylabel('Residuals', fontsize=15)
plt.tight_layout()
plt.show(block=False)



#############################################


x_corrected = lam_sumer_cropNeVIII
x_uncorrected = x_corrected
y_corrected = rad_sumer_cropNeVIII_corrected
y_unc_corrected = erad_sumer_cropNeVIII_corrected
y_uncorrected = rad_sumer_cropNeVIII
y_unc_uncorrected = erad_sumer_cropNeVIII



xb_uncorrected, yb_uncorrected = find_bisector(x_data=x_uncorrected[3:-7], y_data=y_uncorrected[3:-7], y_unc_data=y_unc_uncorrected[3:-7], y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')

xb_corrected, yb_corrected = find_bisector(x_data=x_corrected, y_data=y_corrected, y_unc_data=y_unc_corrected, y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')


######################################################
######################################################
######################################################




"""
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=rad_peak_uncorrected_list, xerr=erad_peak_uncorrected_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='.', label='SUMER uncorrected')
ax.errorbar(x=rad_peak_uncorrected_list, xerr=erad_peak_uncorrected_list, y=v_peak_corrected_list, yerr=ev_peak_corrected_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Spectral radiance peak (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lamb_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.legend()
plt.show(block=False)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=rad_percentage_mean_sumer, xerr=erad_percentage_mean_sumer, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='.', label='SUMER uncorrected')
ax.errorbar(x=rad_percentage_mean_sumer, xerr=erad_percentage_mean_sumer, y=v_peak_corrected_list, yerr=ev_peak_corrected_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Percentage of the mean (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lamb_0}''\u212B')#, label=label_i) 
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
ax.axvline(x=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lamb_0}''\u212B')#, label=label_i) 
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
ax.axvline(x=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lamb_0}''\u212B')#, label=label_i) 
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.legend()
plt.show(block=False)


"""




