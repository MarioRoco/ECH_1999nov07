save_paper_images = 'yes'
folder_name = '../outputs/paper_figures/sumer_contours/v4' #name of the folder where you save the images
save_dpi = 100 #resolution: number of pixels per inch. ChatGPT gave me 300 by default. 


line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'

# Threshold value: label (type) and range of percentageRange percentage of the threshold value
#range_percentage, threshold_value_type = [0., 4.], 'max' #'max', 'min', 'mean', 'median'
#range_percentage, threshold_value_type = [0., 5.], 'max' #'max', 'min', 'mean', 'median'
range_percentage, threshold_value_type = [0., 6.5], 'max' #'max', 'min', 'mean', 'median'


#color_sumer = 'blue'
#color_hrts = 'green'
#color_sumer_uncorrected = 'red'
#color_sumer_corrected = 'blue'


# Parameters of the individual gaussians fits
components_linestyle = '-'
components_linewidth = 1.2
components_color = 'green'
fig_size = (12, 6)

# 
axislabel_size = 13
title_size = 17
legend_size = 13
line_width = 2.


# Wavelength ranges to crop spectra
wavelength_range_to_average = [1531.1147, 1551.7688]
wavelength_range_to_analyze_NeVIII = [1540.2, 1541.4]


# save average profile as .npy?
save_average_profile = 'no' 

show_secondary_plots = 'no'
show_plots_correction = 'yes'


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
from PAPER_scale_hrts import *
from PAPER_fig_params import * 

rest_wavelength_label_figures = f'Rest wavelength ({lam_0/2.}'' \u212B)'
#rest_wavelength_label_figures = 'Rest wavelength: 770.428 \u212B'

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

print('##################################')
print('N PIXELS IN SUMER:', len(rowscols_croplat), '=', len(rowscols_croplat)*150./3600., 'hrs')
print('##################################')


# Extent in pixels 
extent_sumer_px_contours = [0., intensity_map_croplat.shape[1]-1, intensity_map_croplat.shape[0]-1, 0.]
extent_sumer_px_image = [-0.5, intensity_map_croplat.shape[1]-1+0.5, intensity_map_croplat.shape[0]-1+0.5, -0.5]


#################################################
#################################################
#################################################

# Import SUMER data interpolated (wavelength calibrated)
data_interpolated_loaded = np.load('../data/data_modified/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)
# Average spectra of the pixels selected
lam_sumer_av, elam_sumer_av, rad_sumer_av, erad_sumer_av = average_profiles_from_pixels_selected_from_interpolated_data(wavelength_range_=wavelength_range_to_average, data_interpolated_loaded_=data_interpolated_loaded, rows_cols_of_spectroheliogram_croplat=rowscols_croplat)


#################################################
#################################################
#################################################
# Variation of parameters

# Variation of SUMER instrumental profile
#variation_instrumental_profile = 0.1
#fwhm_to_convolve = (variation_instrumental_profile+1.)*fwhm_to_convolve

"""
l_hrts_left = 1531.609
l_hrts_right = 1551.248
l_sumer_left = 1531.550
l_sumer_right = 1551.358
lam_shift = (l_hrts_left - l_sumer_left)
lam_delta = 1.0 + (l_hrts_right-l_sumer_right)/(l_hrts_right-l_hrts_left)
lam_sumer_modified = (lam_sumer_av-l_hrts_left+lam_shift) * lam_delta + l_hrts_left

lam_sumer_av = lam_sumer_modified
"""

"""
lam_delta * lam_sumer_av + (-l_hrts_left*lam_delta + lam_shift*lam_delta + l_hrts_left)
y = mx+b
x = lam_sumer_av
m = lam_delta = 0.9943989001476705
b = (-l_hrts_left*lam_delta + lam_shift*lam_delta + l_hrts_left) = 8.637364478835252
"""
"""
mm = 0.9943989001476705
bb = 8.637364478835252 
lam_sumer_av = mm * lam_sumer_av + bb
"""
#################################################
#################################################
#################################################

############################################################################################################
############################################################################################################
############################################################################################################
# Substract HRTS

x_lims_ranges = [1535.80, 1546.53]
y_lims_ranges_a = [0.09, 1.65]
y_lims_ranges_b = [0.07, 1.65]
y_lims_ranges_l = [0.04, 1.65]

#subtract HRTS QR-A
fsh_qra = fun_scale_hrts(hrts_qr='a', lamb_0=lam_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_to_convolve, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction, title_fit_radiances='auto', title_scaled_HRTSspectrum='auto', title_ranges='auto', x_lims_ranges=x_lims_ranges, y_lims_ranges=y_lims_ranges_a, save_paper_images=save_paper_images, folder_name=folder_name, save_dpi=save_dpi, show_secondary_plots=show_secondary_plots)
lam_sumer_cropNeVIII = fsh_qra['lam_sumer_cropNeVIII']
rad_sumer_cropNeVIII = fsh_qra['rad_sumer_cropNeVIII']
erad_sumer_cropNeVIII = fsh_qra['erad_sumer_cropNeVIII']
rad_sumer_cropNeVIII_corrected_qra = fsh_qra['rad_sumer_cropNeVIII_corrected']
erad_sumer_cropNeVIII_corrected_qra = fsh_qra['erad_sumer_cropNeVIII_corrected']
lam_hrtsa = fsh_qra['lam_hrts']
rad_hrtsa = fsh_qra['rad_hrts']
erad_hrtsa = fsh_qra['erad_hrts']
rad_hrtsa_conv = fsh_qra['rad_hrts_conv']
erad_hrtsa_conv = fsh_qra['erad_hrts_conv']
rad_hrtsa_conv_scaled = fsh_qra['rad_hrts_conv_scaled']
erad_hrtsa_conv_scaled = fsh_qra['erad_hrts_conv_scaled']
lam_hrtsa_cropNeVIII = fsh_qra['lam_hrts_cropNeVIII']
rad_hrtsa_cropNeVIII = fsh_qra['rad_hrts_cropNeVIII']
erad_hrtsa_cropNeVIII = fsh_qra['erad_hrts_cropNeVIII']
rad_hrtsa_conv_scaled_cropNeVIII = fsh_qra['rad_hrts_conv_scaled_cropNeVIII']
erad_hrtsa_conv_scaled_cropNeVIII = fsh_qra['erad_hrts_conv_scaled_cropNeVIII']

#subtract HRTS QR-B
fsh_qrb = fun_scale_hrts(hrts_qr='b', lamb_0=lam_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_to_convolve, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction, title_fit_radiances='auto', title_scaled_HRTSspectrum='auto', title_ranges='auto', x_lims_ranges=x_lims_ranges, y_lims_ranges=y_lims_ranges_b, save_paper_images=save_paper_images, folder_name=folder_name, save_dpi=save_dpi, show_secondary_plots=show_secondary_plots)
rad_sumer_cropNeVIII_corrected_qrb = fsh_qrb['rad_sumer_cropNeVIII_corrected']
erad_sumer_cropNeVIII_corrected_qrb = fsh_qrb['erad_sumer_cropNeVIII_corrected']
lam_hrtsb = fsh_qrb['lam_hrts']
rad_hrtsb = fsh_qrb['rad_hrts']
erad_hrtsb = fsh_qrb['erad_hrts']
rad_hrtsb_conv = fsh_qrb['rad_hrts_conv']
erad_hrtsb_conv = fsh_qrb['erad_hrts_conv']
rad_hrtsb_conv_scaled = fsh_qrb['rad_hrts_conv_scaled']
erad_hrtsb_conv_scaled = fsh_qrb['erad_hrts_conv_scaled']
lam_hrtsb_cropNeVIII = fsh_qrb['lam_hrts_cropNeVIII']
rad_hrtsb_cropNeVIII = fsh_qrb['rad_hrts_cropNeVIII']
erad_hrtsb_cropNeVIII = fsh_qrb['erad_hrts_cropNeVIII']
rad_hrtsb_conv_scaled_cropNeVIII = fsh_qrb['rad_hrts_conv_scaled_cropNeVIII']
erad_hrtsb_conv_scaled_cropNeVIII = fsh_qrb['erad_hrts_conv_scaled_cropNeVIII']

#subtract HRTS QR-L
fsh_qrl = fun_scale_hrts(hrts_qr='l', lamb_0=lam_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_to_convolve, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction, title_fit_radiances='auto', title_scaled_HRTSspectrum='auto', title_ranges='auto', x_lims_ranges=x_lims_ranges, y_lims_ranges=y_lims_ranges_l, save_paper_images=save_paper_images, folder_name=folder_name, save_dpi=save_dpi, show_secondary_plots=show_secondary_plots)
rad_sumer_cropNeVIII_corrected_qrl = fsh_qrl['rad_sumer_cropNeVIII_corrected']
erad_sumer_cropNeVIII_corrected_qrl = fsh_qrl['erad_sumer_cropNeVIII_corrected']
lam_hrtsl = fsh_qrl['lam_hrts']
rad_hrtsl = fsh_qrl['rad_hrts']
erad_hrtsl = fsh_qrl['erad_hrts']
rad_hrtsl_conv = fsh_qrl['rad_hrts_conv']
erad_hrtsl_conv = fsh_qrl['erad_hrts_conv']
rad_hrtsl_conv_scaled = fsh_qrl['rad_hrts_conv_scaled']
erad_hrtsl_conv_scaled = fsh_qrl['erad_hrts_conv_scaled']
lam_hrtsl_cropNeVIII = fsh_qrl['lam_hrts_cropNeVIII']
rad_hrtsl_cropNeVIII = fsh_qrl['rad_hrts_cropNeVIII']
erad_hrtsl_cropNeVIII = fsh_qrl['erad_hrts_cropNeVIII']
rad_hrtsl_conv_scaled_cropNeVIII = fsh_qrl['rad_hrts_conv_scaled_cropNeVIII']
erad_hrtsl_conv_scaled_cropNeVIII = fsh_qrl['erad_hrts_conv_scaled_cropNeVIII']


######################################################
# Show image of the intensity map with the contours and the pixels inside the contours


# Extents in arcsec
from utils.solar_rotation_variables import HPlat, HPlon_rotcomp_dic
HPlat_croplat = HPlat[slit_top_px:slit_bottom_px+1]
HPlon_rotcomp = HPlon_rotcomp_dic['SOHO_EIT_195_19991107T063706_L1.fits']
lat_half_bottom = abs((HPlat_croplat[1]-HPlat_croplat[0])/2.)
lat_half_top = abs((HPlat_croplat[-1]-HPlat_croplat[-2])/2.)
lon_half_left = abs((HPlon_rotcomp[1]-HPlon_rotcomp[0])/2.)
lon_half_right = abs((HPlon_rotcomp[-1]-HPlon_rotcomp[-2])/2.)
extent_eit_sumer_arcsec_image = [HPlon_rotcomp[0]-lon_half_left, HPlon_rotcomp[-1]+lon_half_right, HPlat_croplat[-1]-lat_half_bottom, HPlat_croplat[0]+lat_half_top] #arcsec
extent_eit_sumer_arcsec_contours = [HPlon_rotcomp[0], HPlon_rotcomp[-1], HPlat_croplat[-1], HPlat_croplat[0]] #arcsec


if show_secondary_plots == 'yes':
	# contours and pixels
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5.7))
	img = ax.imshow(intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer), extent=extent_sumer_px_image)
	cbar = fig.colorbar(img, ax=ax, pad=0.03)
	ax.set_title(f'SOHO/SUMER intensity map {line_center_label}, solar rotation NOT compensated')
	ax.set_xlabel('Helioprojective longitude (arcsec), rotation compensated')
	ax.set_ylabel('Helioprojective latitude (arcsec)')
	ax.axis('auto') # Ensures equal scaling of axis x and y
	ax.scatter(x=x_col_list_plot, y=y_row_list_plot, s=1, color='yellow')
	contour_lower = ax.contour(intensity_map_croplat[::-1], levels=[lower_bound], colors='red', linewidths=1, extent=extent_sumer_px_contours)
	contour_upper = ax.contour(intensity_map_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=1, extent=extent_sumer_px_contours)
	legend_elements = [
	mlines.Line2D([],[],color='red', label=f'{lower_bound}'),
	mlines.Line2D([],[],color='blue', label=f'{upper_bound}')]
	plt.show(block=False)
	 
	 
	 
	#Ne VIII intensity map with contours
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5.7))
	img = ax.imshow(intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer), extent=extent_eit_sumer_arcsec_image)
	cbar = fig.colorbar(img, ax=ax, pad=0.03)
	ax.set_title(f'SOHO/SUMER intensity map {line_center_label}, solar rotation NOT compensated')
	ax.set_xlabel('Helioprojective longitude (arcsec), rotation compensated')
	ax.set_ylabel('Helioprojective latitude (arcsec)')
	ax.axis('auto') # Ensures equal scaling of axis x and y
	contour_lower = ax.contour(intensity_map_croplat[::-1], levels=[lower_bound], colors='red', linewidths=1, extent=extent_eit_sumer_arcsec_contours)
	contour_upper = ax.contour(intensity_map_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=1, extent=extent_eit_sumer_arcsec_contours)
	legend_elements = [
	mlines.Line2D([],[],color='red', label=f'{lower_bound}'),
	mlines.Line2D([],[],color='blue', label=f'{upper_bound}')]
	plt.show(block=False)


######################################################
# Show averaged spectra


if show_secondary_plots == 'yes':
	# Full wavelength range
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
	ax.errorbar(x=lam_sumer_av, y=rad_sumer_av, yerr=erad_sumer_av, color='blue', linewidth=1., label='SUMER data')
	ax.set_title(f'SOHO/SUMER, profile averaged', fontsize=18) 
	ax.set_xlabel('Wavelength (\u212B)', color='black', fontsize=16)
	ax.set_ylabel(f'Av. spectral radiance [W/sr/m^2/Angstroem]', color='black', fontsize=16)
	ax.axvline(lam_0, color='green', linewidth=1., label=rest_wavelength_label_figures)
	ax.axvspan(lam_0-lam_unc_0, lam_0+lam_unc_0, color='green', alpha=0.2)
	# legend in desired order:
	handles, labels = ax.get_legend_handles_labels()
	order = [
	labels.index('SUMER data'),
	labels.index(rest_wavelength_label_figures),]
	ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size)
	plt.show(block=False)


	# Wavelength range cropped around Ne VIII
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
	ax.errorbar(x=lam_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color='blue', linewidth=1., label='SUMER data')
	ax.set_title(f'SOHO/SUMER, profile averaged', fontsize=18) 
	ax.set_xlabel('Wavelength (\u212B)', color='black', fontsize=16)
	ax.set_ylabel(f'Av. spectral radiance [W/sr/m^2/Angstroem]', color='black', fontsize=16)
	ax.axvline(lam_0, color='green', linewidth=1., label=rest_wavelength_label_figures)
	ax.axvspan(lam_0-lam_unc_0, lam_0+lam_unc_0, color='green', alpha=0.2)
	# legend in desired order:
	handles, labels = ax.get_legend_handles_labels()
	order = [
	labels.index('SUMER data'),
	labels.index(rest_wavelength_label_figures),]
	ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size)
	plt.show(block=False)


# Plot: Comparison SUMER uncorrected and corrected (QR-A)
fig, ax = plt.subplots(figsize=(12, 5))
#ax.errorbar(x=vkms_doppler(lamb=lam_crop, lamb_0=lam_0), y=rad_crop, yerr=erad_crop, color='black', linewidth=0.6, label='SUMER box') #Real spectrum (SUMER) 
#ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER lowest {range_percentage}%, not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label='SUMER not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0), y=rad_sumer_cropNeVIII_corrected_qra, yerr=erad_sumer_cropNeVIII_corrected_qra, color=color_sumer_corrected_qra, linestyle='-', linewidth=2., label='SUMER corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_hrtsa_cropNeVIII, lamb_0=lam_0), y=rad_hrtsa_conv_scaled_cropNeVIII, yerr=erad_hrtsa_conv_scaled_cropNeVIII, color=color_sumer_corrected_qra, linestyle='--', linewidth=2., label='HRST - QS A') #Real spectrum (SUMER)
ax.axvline(x=0, color='black', linestyle=':', linewidth=2., label=rest_wavelength_label_figures)
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_title(f'Comparison SUMER before and after correction with HRTS QS-A', fontsize=18)
ax.set_xlabel('Doppler shift (km/s)', fontsize=15)
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
ax.set_xlim([vkms_doppler(lamb=min(lam_hrtsa_cropNeVIII), lamb_0=lam_0), vkms_doppler(lamb=max(lam_hrtsa_cropNeVIII), lamb_0=lam_0)])
# legend in desired order:
handles, labels = ax.get_legend_handles_labels()
order = [
labels.index('SUMER not corrected'),
labels.index('SUMER corrected'),
labels.index('HRST - QS A'),
labels.index(rest_wavelength_label_figures),]
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size)
ax.set_yscale('linear')
plt.show(block=False)


# Plot: Comparison SUMER uncorrected and corrected (QR-B)
fig, ax = plt.subplots(figsize=(12, 5))
#ax.errorbar(x=vkms_doppler(lamb=lam_crop, lamb_0=lam_0), y=rad_crop, yerr=erad_crop, color='black', linewidth=0.6, label='SUMER box') #Real spectrum (SUMER) 
#ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER lowest {range_percentage}%, not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0), y=rad_sumer_cropNeVIII_corrected_qrb, yerr=erad_sumer_cropNeVIII_corrected_qrb, color=color_sumer_corrected_qrb, linestyle='-', linewidth=2., label=f'SUMER corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_hrtsb_cropNeVIII, lamb_0=lam_0), y=rad_hrtsb_conv_scaled_cropNeVIII, yerr=erad_hrtsb_conv_scaled_cropNeVIII, color=color_sumer_corrected_qrb, linestyle='--', linewidth=2., label='HRST - QS B') #Real spectrum (SUMER)
ax.axvline(x=0, color='black', linestyle=':', linewidth=2., label=rest_wavelength_label_figures)
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_title(f'Comparison SUMER before and after correction with HRTS QS-B', fontsize=18)
ax.set_xlabel('Doppler shift (km/s)', fontsize=15)
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
ax.set_xlim([vkms_doppler(lamb=min(lam_hrtsb_cropNeVIII), lamb_0=lam_0), vkms_doppler(lamb=max(lam_hrtsb_cropNeVIII), lamb_0=lam_0)])
# legend in desired order:
handles, labels = ax.get_legend_handles_labels()
order = [
labels.index('SUMER not corrected'),
labels.index('SUMER corrected'),
labels.index('HRST - QS B'),
labels.index(rest_wavelength_label_figures),]
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size)
ax.set_yscale('linear')
plt.show(block=False)


# Plot: Comparison SUMER uncorrected and corrected (QR-L)
fig, ax = plt.subplots(figsize=(12, 5))
#ax.errorbar(x=vkms_doppler(lamb=lam_crop, lamb_0=lam_0), y=rad_crop, yerr=erad_crop, color='black', linewidth=0.6, label='SUMER box') #Real spectrum (SUMER) 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label='SUMER not corrected')#, label=f'SUMER lowest {range_percentage}%, not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0), y=rad_sumer_cropNeVIII_corrected_qrl, yerr=erad_sumer_cropNeVIII_corrected_qrl, color=color_sumer_corrected_qrl, linestyle='-', linewidth=2., label='SUMER corrected')#, label=f'SUMER {range_percentage} of the maximum%, corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_hrtsl_cropNeVIII, lamb_0=lam_0), y=rad_hrtsl_conv_scaled_cropNeVIII, yerr=erad_hrtsl_conv_scaled_cropNeVIII, color=color_sumer_corrected_qrl, linestyle='--', linewidth=2., label='HRST - QS L') #Real spectrum (SUMER)
ax.axvline(x=0, color='black', linestyle=':', linewidth=2., label=rest_wavelength_label_figures)
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_title(f'Comparison SUMER before and after correction with HRTS QS-L', fontsize=18)
ax.set_xlabel('Doppler shift (km/s)', fontsize=15)
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
ax.set_xlim([vkms_doppler(lamb=min(lam_hrtsl_cropNeVIII), lamb_0=lam_0), vkms_doppler(lamb=max(lam_hrtsl_cropNeVIII), lamb_0=lam_0)])
# legend in desired order:
handles, labels = ax.get_legend_handles_labels()
order = [
labels.index('SUMER not corrected'),
labels.index('SUMER corrected'),
labels.index('HRST - QS L'),
labels.index(rest_wavelength_label_figures),]
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size)
ax.set_yscale('linear')
plt.show(block=False)



# save average profile as .npy
if save_average_profile == 'yes':
    range_numbers_to_string = '__'.join(f"{x:.2f}".replace('.', '_').rstrip('0') if f"{x:.2f}"[-1] != '0' else f"{x:.1f}".replace('.', '_') for x in range_percentage) 
    filename_profile = 'average_profile__' + range_numbers_to_string + '__' + threshold_value_type + '_of_sumer_' + line_label
    foldepath_profile = '../outputs/'
    np.savez(foldepath_profile+filename_profile, lam_sumer_cropNeVIII=lam_sumer_cropNeVIII, rad_sumer_cropNeVIII=rad_sumer_cropNeVIII, erad_sumer_cropNeVIII=erad_sumer_cropNeVIII, rad_sumer_cropNeVIII_corrected_qra=rad_sumer_cropNeVIII_corrected_qra, erad_sumer_cropNeVIII_corrected_qra=erad_sumer_cropNeVIII_corrected_qra, lam_hrtsa_cropNeVIII=lam_hrtsa_cropNeVIII, rad_hrtsa_conv_scaled_cropNeVIII=rad_hrtsa_conv_scaled_cropNeVIII, erad_hrtsa_conv_scaled_cropNeVIII=erad_hrtsa_conv_scaled_cropNeVIII, rad_sumer_cropNeVIII_corrected_qrb=rad_sumer_cropNeVIII_corrected_qrb, erad_sumer_cropNeVIII_corrected_qrb=erad_sumer_cropNeVIII_corrected_qrb, lam_hrtsb_cropNeVIII=lam_hrtsb_cropNeVIII, rad_hrtsb_conv_scaled_cropNeVIII=rad_hrtsb_conv_scaled_cropNeVIII, erad_hrtsb_conv_scaled_cropNeVIII=erad_hrtsb_conv_scaled_cropNeVIII, rad_sumer_cropNeVIII_corrected_qrl=rad_sumer_cropNeVIII_corrected_qrl, erad_sumer_cropNeVIII_corrected_qrl=erad_sumer_cropNeVIII_corrected_qrl, lam_hrtsl_cropNeVIII=lam_hrtsl_cropNeVIII, rad_hrtsl_conv_scaled_cropNeVIII=rad_hrtsl_conv_scaled_cropNeVIII, erad_hrtsl_conv_scaled_cropNeVIII=erad_hrtsl_conv_scaled_cropNeVIII)


"""
In order to load the intensity map in another file (or this one), do the next:

profiles_loaded_dic = np.load(filename_profile)
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

"""
""


##############################################################
##############################################################
##############################################################
# Dopplermap and BR asymmetry map

# Load the intensity map and uncertainties
dopplermap_BRmap_loaded_dic = np.load('../outputs/dopplermap_BRmap.npz')
ddopplershift_map_binned_HRTSsub_lessmedian = dopplermap_BRmap_loaded_dic['ddopplershift_map_binned_HRTSsub_lessmedian']
BR_asymmetry_map_gaussian_binned_corrected_normalized = dopplermap_BRmap_loaded_dic['BR_map']


### Dopplermap without contours
vmin_vmax = [-12., 12.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
label_size = 18
img = ax.imshow(ddopplershift_map_binned_HRTSsub_lessmedian, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap='seismic', extent=extent_eit_sumer_arcsec_image)
cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cax)
cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
ax.set_title(r'Doppler map, blends corrected', fontsize=20)
#ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
ax.set_aspect('auto')
if save_paper_images == 'yes':
	fig_name = 'dopplermap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)

### PAPER image: Dopplermap with SUMER contours
vmin_vmax = [-12., 12.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
label_size = 18
img = ax.imshow(ddopplershift_map_binned_HRTSsub_lessmedian, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap='seismic', extent=extent_eit_sumer_arcsec_image)
cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cax)
cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
ax.set_title(r'Doppler map, blends corrected', fontsize=20)
#ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
#plt.subplots_adjust(left=0.1, right=0.90, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_lower = ax.contour(intensity_map_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax.contour(intensity_map_croplat[::-1], levels=[upper_bound], colors='black', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
if save_paper_images == 'yes':
	fig_name = 'contours_SUMER__dopplermap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



### B-R asymmetry map without contours
vmin_vmax_BR = [-1.,1.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
img = ax.imshow(BR_asymmetry_map_gaussian_binned_corrected_normalized, vmin=vmin_vmax_BR[0], vmax=vmin_vmax_BR[1], cmap='seismic', extent=extent_eit_sumer_arcsec_image)
cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cax)
cbar.set_label('Red-blue asymmetry normalized', fontsize=16)
ax.set_title('R-B normalized, blends corrected', fontsize=20)
#ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
ax.set_aspect('auto')
if save_paper_images == 'yes':
	fig_name = 'asymmetrymap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



### PAPER image: B-R asymmetry map with contours of SUMER
vmin_vmax_BR = [-1.,1.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
img = ax.imshow(BR_asymmetry_map_gaussian_binned_corrected_normalized, vmin=vmin_vmax_BR[0], vmax=vmin_vmax_BR[1], cmap='seismic', extent=extent_eit_sumer_arcsec_image)
cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cax)
cbar.set_label('Red-blue asymmetry normalized', fontsize=16)
ax.set_title('R-B normalized, blends corrected', fontsize=20)
#ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
#plt.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_lower = ax.contour(intensity_map_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax.contour(intensity_map_croplat[::-1], levels=[upper_bound], colors='black', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
if save_paper_images == 'yes':
	fig_name = 'contours_SUMER__asymmetrymap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)


##############################################################
##############################################################
##############################################################
# 


## HRTS wavelength and Doppler velicity
v_hrtsa_cropNeVIII = vkms_doppler(lamb=lam_hrtsa_cropNeVIII, lamb_0=lam_0)
v_hrtsb_cropNeVIII = vkms_doppler(lamb=lam_hrtsb_cropNeVIII, lamb_0=lam_0)
v_hrtsl_cropNeVIII = vkms_doppler(lamb=lam_hrtsl_cropNeVIII, lamb_0=lam_0)




## SUMER wavelength and Doppler velocity (respect to the rest wavelength of Ne VIII 770 in 2nd order)
v_sumer_cropNeVIII = vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0)



######################################################
######################################################
######################################################
# PAPER plot

x_lims = [max(min(v_hrtsa_cropNeVIII), min(v_sumer_cropNeVIII)),      min(max(v_hrtsa_cropNeVIII), max(v_sumer_cropNeVIII))]


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
0.04-bckg_fit_corrected_qra, -67., 20.,
#0.13-bckg_fit_corrected_qra, -31., 20.,
0.34-bckg_fit_corrected_qra, 0.0, 50.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.04-bckg_fit_corrected_qrb, -80., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
#0.1-bckg_fit_corrected_qrb, -40., 40.,
0.37-bckg_fit_corrected_qrb, 0.0, 50.,
#1.5-bckg_fit_corrected_qrb, 33., 40.
]

bckg_fit_corrected_qrl = 0.
init_parameters_corrected_qrl = [bckg_fit_corrected_qrl, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.08-bckg_fit_corrected_qrl, -75., 20.,
0.3-bckg_fit_corrected_qrl, -35., 30.,
0.8-bckg_fit_corrected_qrl, 7., 50.,
0.022-bckg_fit_corrected_qrl, 77., 30.
]



######################################################

### PAPER image: spectra sumer and hrts together

x_lims = [max(min(v_hrtsa_cropNeVIII), min(v_sumer_cropNeVIII)),      min(max(v_hrtsa_cropNeVIII), max(v_sumer_cropNeVIII))]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
## HRTS scaled and convolved
ax.errorbar(x=v_hrtsa_cropNeVIII, y=rad_hrtsa_conv_scaled_cropNeVIII, linestyle='--', linewidth=line_width, color=color_hrts_qra, label='HRTS QS-A')
ax.errorbar(x=v_hrtsb_cropNeVIII, y=rad_hrtsb_conv_scaled_cropNeVIII, linestyle='--', linewidth=line_width, color=color_hrts_qrb, label='HRTS QS-B')
## SUMER uncorrected
#ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, linestyle='-', marker='.', markersize=10, linewidth=line_width, color=color_sumer_uncorrected, label='SUMER uncorrected')
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, linestyle='-', linewidth=line_width, color=color_sumer_uncorrected, label='SUMER uncorrected')
## SUMER corrected
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII_corrected_qra, yerr=erad_sumer_cropNeVIII_corrected_qra, linestyle='-', linewidth=line_width, color=color_hrts_qra, label='SUMER corrected, QS-A')
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII_corrected_qrb, yerr=erad_sumer_cropNeVIII_corrected_qrb, linestyle='-', linewidth=line_width, color=color_hrts_qrb, label='SUMER corrected, QS-B')
## 
ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, label=rest_wavelength_label_figures)
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_title(f'SUMER spectrum of the CH uncorrected and corrected with HRTS', fontsize=title_size)
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
#ax.legend(fontsize=legend_size)
# legend in desired order:
handles, labels = ax.get_legend_handles_labels()
order = [
    labels.index('SUMER uncorrected'),
    labels.index('HRTS QS-A'),
    labels.index('HRTS QS-B'),
    labels.index('SUMER corrected, QS-A'),
    labels.index('SUMER corrected, QS-B'),
    labels.index(rest_wavelength_label_figures),]
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size)
ax.set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size) 
ax.set_xlim(x_lims)
plt.tight_layout()
if save_paper_images == 'yes':
    fig_name = 'contours_SUMER__spectra_sumer_and_hrts_together'
    plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)


######################################################


lines_id = [[1539.705 , 'Si I'],
[1539.738 , 'Si I'],
[1539.849 , 'Fe II'],
[1539.951 , 'Fe II'],
[1540.287 , 'Si I'], 
[1540.369 , 'Fe II'],
[1540.706 , 'Si I'],
[1540.707 , 'Si I'],
[1540.782 , 'Si I'],
[1540.963 , 'Si I'],
[1540.985 , 'Si I'],
[1541.026 , 'Fe II'],
[1541.033 , 'Fe II'],
[1541.322 , 'Si I'],
[1541.415 , 'Si I'],
[1541.455 , 'Fe II'],
[1542.186 , 'Si I'],
[1542.269 , 'Si I'],
[1542.340 , 'Si I'],
[1542.432 , 'Si I']]


lines_id = [
#[1540.354 , 'Si I'],
[1540.544 , 'Si I'],
[1540.707 , 'Si I'],
[1540.782 , 'Si I'],
[1540.963 , 'Si I'],
[1540.978 , 'Si I'],
#[1540.985 , 'Si I'],
[1541.026 , 'Fe II'],
#[1541.033 , 'Fe II'],
[1541.064 , 'Si I'],
#[1541.178 , 'Si I'],
#[1541.198 , 'Si I'],
#[1541.322 , 'Si I']
]


"""
### PAPER image: spectra sumer and hrts together, and identified lines
x_lims = [max(min(v_hrtsa_cropNeVIII), min(v_sumer_cropNeVIII)),      min(max(v_hrtsa_cropNeVIII), max(v_sumer_cropNeVIII))]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
## HRTS scaled and convolved
ax.errorbar(x=v_hrtsa_cropNeVIII, y=rad_hrtsa_conv_scaled_cropNeVIII, linestyle='--', linewidth=line_width, color=color_hrts_qra, label='HRTS QS-A')
ax.errorbar(x=v_hrtsb_cropNeVIII, y=rad_hrtsb_conv_scaled_cropNeVIII, linestyle='--', linewidth=line_width, color=color_hrts_qrb, label='HRTS QS-B')
## SUMER uncorrected
#ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, linestyle='-', marker='.', markersize=10, linewidth=line_width, color=color_sumer_uncorrected, label='SUMER uncorrected')
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, linestyle='-', linewidth=line_width, color=color_sumer_uncorrected, label='SUMER uncorrected')
## SUMER corrected
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII_corrected_qra, yerr=erad_sumer_cropNeVIII_corrected_qra, linestyle='-', linewidth=line_width, color=color_hrts_qra, label='SUMER corrected, QS-A')
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII_corrected_qrb, yerr=erad_sumer_cropNeVIII_corrected_qrb, linestyle='-', linewidth=line_width, color=color_hrts_qrb, label='SUMER corrected, QS-B')
## 
ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, label=rest_wavelength_label_figures)
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_title(f'SUMER spectrum of the CH uncorrected and corrected with HRTS', fontsize=title_size)
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
#ax.legend(fontsize=legend_size)
# legend in desired order:
handles, labels = ax.get_legend_handles_labels()
order = [
    labels.index('SUMER uncorrected'),
    labels.index('HRTS QS-A'),
    labels.index('HRTS QS-B'),
    labels.index('SUMER corrected, QS-A'),
    labels.index('SUMER corrected, QS-B'),
    labels.index(rest_wavelength_label_figures),]
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size-2)
ax.set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size) 
ax.set_xlim(x_lims)
plt.tight_layout()
# Add vertical lines and annotations at the upper part of the panel
_, ymax = ax.get_ylim() # Get the current Y-axis limits
for wavelength_i, symbol_i in lines_id:
	v_i = vkms_doppler(lamb=wavelength_i, lamb_0=lam_0)
	ax.axvline(x=v_i, color='green', linestyle='--')
	ax.text(v_i+0.2, ymax * 0.55, f'{symbol_i} - {wavelength_i}', rotation=90, verticalalignment='top', fontsize=10, color='green') # Position the text near the top of the plot panel, slightly below the max y-limit
if save_paper_images == 'yes':
    fig_name = 'contours_SUMER__spectra_sumer_and_hrts_together_and_line_identification'
    plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)
"""


######################################################
######################################################
######################################################
######################################################
######################################################
# Expand x limits

wavelength_range_to_analyze_NeVIII_v2 = [1540.2-2., 1541.4+2.]


#subtract HRTS QR-A
fsh_qra_v2 = fun_scale_hrts(hrts_qr='a', lamb_0=lam_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_to_convolve, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII_v2, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction, title_fit_radiances='auto', title_scaled_HRTSspectrum='auto', title_ranges='auto', x_lims_ranges=x_lims_ranges, y_lims_ranges=y_lims_ranges_a, save_paper_images=save_paper_images, folder_name=folder_name, save_dpi=save_dpi, show_secondary_plots=show_secondary_plots)
lam_sumer_cropNeVIII_v2 = fsh_qra_v2['lam_sumer_cropNeVIII']
rad_sumer_cropNeVIII_v2 = fsh_qra_v2['rad_sumer_cropNeVIII']
erad_sumer_cropNeVIII_v2 = fsh_qra_v2['erad_sumer_cropNeVIII']
rad_sumer_cropNeVIII_corrected_qra_v2 = fsh_qra_v2['rad_sumer_cropNeVIII_corrected']
erad_sumer_cropNeVIII_corrected_qra_v2 = fsh_qra_v2['erad_sumer_cropNeVIII_corrected']
lam_hrtsa_v2 = fsh_qra_v2['lam_hrts']
rad_hrtsa_v2 = fsh_qra_v2['rad_hrts']
erad_hrtsa_v2 = fsh_qra_v2['erad_hrts']
rad_hrtsa_conv_v2 = fsh_qra_v2['rad_hrts_conv']
erad_hrtsa_conv_v2 = fsh_qra_v2['erad_hrts_conv']
rad_hrtsa_conv_scaled_v2 = fsh_qra_v2['rad_hrts_conv_scaled']
erad_hrtsa_conv_scaled_v2 = fsh_qra_v2['erad_hrts_conv_scaled']
lam_hrtsa_cropNeVIII_v2 = fsh_qra_v2['lam_hrts_cropNeVIII']
rad_hrtsa_cropNeVIII_v2 = fsh_qra_v2['rad_hrts_cropNeVIII']
erad_hrtsa_cropNeVIII_v2 = fsh_qra_v2['erad_hrts_cropNeVIII']
rad_hrtsa_conv_scaled_cropNeVIII_v2 = fsh_qra_v2['rad_hrts_conv_scaled_cropNeVIII']
erad_hrtsa_conv_scaled_cropNeVIII_v2 = fsh_qra_v2['erad_hrts_conv_scaled_cropNeVIII']

#subtract HRTS QR-B
fsh_qrb_v2 = fun_scale_hrts(hrts_qr='b', lamb_0=lam_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_to_convolve, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII_v2, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction, title_fit_radiances='auto', title_scaled_HRTSspectrum='auto', title_ranges='auto', x_lims_ranges=x_lims_ranges, y_lims_ranges=y_lims_ranges_b, save_paper_images=save_paper_images, folder_name=folder_name, save_dpi=save_dpi, show_secondary_plots=show_secondary_plots)
rad_sumer_cropNeVIII_corrected_qrb_v2 = fsh_qrb_v2['rad_sumer_cropNeVIII_corrected']
erad_sumer_cropNeVIII_corrected_qrb_v2 = fsh_qrb_v2['erad_sumer_cropNeVIII_corrected']
lam_hrtsb_v2 = fsh_qrb_v2['lam_hrts']
rad_hrtsb_v2 = fsh_qrb_v2['rad_hrts']
erad_hrtsb_v2 = fsh_qrb_v2['erad_hrts']
rad_hrtsb_conv_v2 = fsh_qrb_v2['rad_hrts_conv']
erad_hrtsb_conv_v2 = fsh_qrb_v2['erad_hrts_conv']
rad_hrtsb_conv_scaled_v2 = fsh_qrb_v2['rad_hrts_conv_scaled']
erad_hrtsb_conv_scaled_v2 = fsh_qrb_v2['erad_hrts_conv_scaled']
lam_hrtsb_cropNeVIII_v2 = fsh_qrb_v2['lam_hrts_cropNeVIII']
rad_hrtsb_cropNeVIII_v2 = fsh_qrb_v2['rad_hrts_cropNeVIII']
erad_hrtsb_cropNeVIII_v2 = fsh_qrb_v2['erad_hrts_cropNeVIII']
rad_hrtsb_conv_scaled_cropNeVIII_v2 = fsh_qrb_v2['rad_hrts_conv_scaled_cropNeVIII']
erad_hrtsb_conv_scaled_cropNeVIII_v2 = fsh_qrb_v2['erad_hrts_conv_scaled_cropNeVIII']

#subtract HRTS QR-L
fsh_qrl_v2 = fun_scale_hrts(hrts_qr='l', lamb_0=lam_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_to_convolve, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII_v2, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction, title_fit_radiances='auto', title_scaled_HRTSspectrum='auto', title_ranges='auto', x_lims_ranges=x_lims_ranges, y_lims_ranges=y_lims_ranges_l, save_paper_images=save_paper_images, folder_name=folder_name, save_dpi=save_dpi, show_secondary_plots=show_secondary_plots)
rad_sumer_cropNeVIII_corrected_qrl_v2 = fsh_qrl_v2['rad_sumer_cropNeVIII_corrected']
erad_sumer_cropNeVIII_corrected_qrl_v2 = fsh_qrl_v2['erad_sumer_cropNeVIII_corrected']
lam_hrtsl_v2 = fsh_qrl_v2['lam_hrts']
rad_hrtsl_v2 = fsh_qrl_v2['rad_hrts']
erad_hrtsl_v2 = fsh_qrl_v2['erad_hrts']
rad_hrtsl_conv_v2 = fsh_qrl_v2['rad_hrts_conv']
erad_hrtsl_conv_v2 = fsh_qrl_v2['erad_hrts_conv']
rad_hrtsl_conv_scaled_v2 = fsh_qrl_v2['rad_hrts_conv_scaled']
erad_hrtsl_conv_scaled_v2 = fsh_qrl_v2['erad_hrts_conv_scaled']
lam_hrtsl_cropNeVIII_v2 = fsh_qrl_v2['lam_hrts_cropNeVIII']
rad_hrtsl_cropNeVIII_v2 = fsh_qrl_v2['rad_hrts_cropNeVIII']
erad_hrtsl_cropNeVIII_v2 = fsh_qrl_v2['erad_hrts_cropNeVIII']
rad_hrtsl_conv_scaled_cropNeVIII_v2 = fsh_qrl_v2['rad_hrts_conv_scaled_cropNeVIII']
erad_hrtsl_conv_scaled_cropNeVIII_v2 = fsh_qrl_v2['erad_hrts_conv_scaled_cropNeVIII']



## HRTS wavelength and Doppler velicity
v_hrtsa_cropNeVIII_v2 = vkms_doppler(lamb=lam_hrtsa_cropNeVIII_v2, lamb_0=lam_0)
v_hrtsb_cropNeVIII_v2 = vkms_doppler(lamb=lam_hrtsb_cropNeVIII_v2, lamb_0=lam_0)
v_hrtsl_cropNeVIII_v2 = vkms_doppler(lamb=lam_hrtsl_cropNeVIII_v2, lamb_0=lam_0)

## SUMER wavelength and Doppler velocity (respect to the rest wavelength of Ne VIII 770 in 2nd order)
v_sumer_cropNeVIII_v2 = vkms_doppler(lamb=lam_sumer_cropNeVIII_v2, lamb_0=lam_0)

lines_Si = [
#[1540.354 , 'Si I'],
[1540.544 , 'Si I'],
[1540.707 , 'Si I'],
[1540.782 , 'Si I'],
[1540.963 , 'Si I'],
[1540.978 , 'Si I'],
#[1540.985 , 'Si I'],
[1541.064 , 'Si I'],
#[1541.178 , 'Si I'],
#[1541.198 , 'Si I'],
#[1541.322 , 'Si I']
]

lines_Fe = [
[1541.026 , 'Fe II'],
#[1541.033 , 'Fe II'],
]

"""
### PAPER image: spectra sumer and hrts together, and identified lines
line_width_v2 = 1.2
x_lims = [max(min(v_hrtsa_cropNeVIII_v2), min(v_sumer_cropNeVIII_v2)),      min(max(v_hrtsa_cropNeVIII_v2), max(v_sumer_cropNeVIII_v2))]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
## HRTS scaled and convolved
ax.errorbar(x=v_hrtsa_cropNeVIII_v2, y=rad_hrtsa_conv_scaled_cropNeVIII_v2, linestyle='--', linewidth=line_width_v2, color=color_hrts_qra, label='HRTS QS-A')
ax.errorbar(x=v_hrtsb_cropNeVIII_v2, y=rad_hrtsb_conv_scaled_cropNeVIII_v2, linestyle='--', linewidth=line_width_v2, color=color_hrts_qrb, label='HRTS QS-B')
## SUMER uncorrected
#ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, linestyle='-', marker='.', markersize=10, linewidth=line_width_v2, color=color_sumer_uncorrected, label='SUMER uncorrected')
ax.errorbar(x=v_sumer_cropNeVIII_v2, y=rad_sumer_cropNeVIII_v2, yerr=erad_sumer_cropNeVIII_v2, linestyle='-', linewidth=line_width_v2, color=color_sumer_uncorrected, label='SUMER uncorrected')
## SUMER corrected
ax.errorbar(x=v_sumer_cropNeVIII_v2, y=rad_sumer_cropNeVIII_corrected_qra_v2, yerr=erad_sumer_cropNeVIII_corrected_qra_v2, linestyle='-', linewidth=line_width_v2, color=color_hrts_qra, label='SUMER corrected, QS-A')
ax.errorbar(x=v_sumer_cropNeVIII_v2, y=rad_sumer_cropNeVIII_corrected_qrb_v2, yerr=erad_sumer_cropNeVIII_corrected_qrb_v2, linestyle='-', linewidth=line_width_v2, color=color_hrts_qrb, label='SUMER corrected, QS-B')
## 
ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, label=rest_wavelength_label_figures)
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax.set_title(f'SUMER spectrum of the CH uncorrected and corrected with HRTS', fontsize=title_size)
#ax.set_title(f'CH-2', fontsize=title_size)
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
#ax.legend(fontsize=legend_size)
ax.set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size) 
ax.set_xlim(x_lims)
plt.tight_layout()
# Add vertical lines and annotations at the upper part of the panel
_, ymax = ax.get_ylim() # Get the current Y-axis limits
for wavelength_i, symbol_i in lines_Si:
	v_i = vkms_doppler(lamb=wavelength_i, lamb_0=lam_0)
	ax.axvline(x=v_i, color='green', linestyle='--', linewidth=line_width_v2)
	#ax.text(v_i+0.2, ymax * 0.55, f'{symbol_i} - {wavelength_i}', rotation=90, verticalalignment='top', fontsize=10, color='green') # Position the text near the top of the plot panel, slightly below the max y-limit
ax.plot([],[], color='green', linestyle='--', linewidth=line_width_v2, label='Si I lines')
for wavelength_i, symbol_i in lines_Fe:
	v_i = vkms_doppler(lamb=wavelength_i, lamb_0=lam_0)
	ax.axvline(x=v_i, color='gold', linestyle='--', linewidth=line_width_v2)
	#ax.text(v_i+0.2, ymax * 0.55, f'{symbol_i} - {wavelength_i}', rotation=90, verticalalignment='top', fontsize=10, color='green') # Position the text near the top of the plot panel, slightly below the max y-limit
ax.plot([],[], color='gold', linestyle='--', linewidth=line_width_v2, label='Fe II lines')
# Big text at the right inside the panel
ax.text(
    0.8, 0.5,               # x, y in axes coordinates (0–1); >1 is outside data area but still in axes
    'CH-2',    # your text (multiline if you want)
    transform=ax.transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=28,             # “big” text; adjust as needed
    fontweight='bold', color=color_contours_sumer_Imap)#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
# legend in desired order:
handles, labels = ax.get_legend_handles_labels()
order = [
    labels.index('SUMER uncorrected'),
    labels.index('HRTS QS-A'),
    labels.index('HRTS QS-B'),
    labels.index('SUMER corrected, QS-A'),
    labels.index('SUMER corrected, QS-B'),
    labels.index(rest_wavelength_label_figures),
    labels.index('Si I lines'),
    labels.index('Fe II lines')]
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size-2, loc='upper left')
if save_paper_images == 'yes':
    fig_name = 'contours_EIT__spectra_sumer_and_hrts_together_and_line_identification'
    plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)
"""


### PAPER image: spectra sumer and hrts together, and identified lines
line_width_v2 = 1.2
x_lims = [max(min(lam_hrtsa_cropNeVIII_v2), min(lam_sumer_cropNeVIII_v2)),      min(max(lam_hrtsa_cropNeVIII_v2), max(lam_sumer_cropNeVIII_v2))]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
## HRTS scaled and convolved
ax.errorbar(x=lam_hrtsa_cropNeVIII_v2, y=rad_hrtsa_conv_scaled_cropNeVIII_v2, linestyle='--', linewidth=line_width_v2, color=color_hrts_qra, label='HRTS QS-A')
ax.errorbar(x=lam_hrtsb_cropNeVIII_v2, y=rad_hrtsb_conv_scaled_cropNeVIII_v2, linestyle='--', linewidth=line_width_v2, color=color_hrts_qrb, label='HRTS QS-B')
## SUMER uncorrected
#ax.errorbar(x=lam_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, linestyle='-', marker='.', markersize=10, linewidth=line_width_v2, color=color_sumer_uncorrected, label='SUMER uncorrected')
ax.errorbar(x=lam_sumer_cropNeVIII_v2, y=rad_sumer_cropNeVIII_v2, yerr=erad_sumer_cropNeVIII_v2, linestyle='-', linewidth=line_width_v2, color=color_sumer_uncorrected, label='SUMER uncorrected')
## SUMER corrected
ax.errorbar(x=lam_sumer_cropNeVIII_v2, y=rad_sumer_cropNeVIII_corrected_qra_v2, yerr=erad_sumer_cropNeVIII_corrected_qra_v2, linestyle='-', linewidth=line_width_v2, color=color_hrts_qra, label='SUMER corrected, QS-A')
ax.errorbar(x=lam_sumer_cropNeVIII_v2, y=rad_sumer_cropNeVIII_corrected_qrb_v2, yerr=erad_sumer_cropNeVIII_corrected_qrb_v2, linestyle='-', linewidth=line_width_v2, color=color_hrts_qrb, label='SUMER corrected, QS-B')
## 
ax.axvline(x=lam_0, color='black', linestyle=':', linewidth=1.5, label=rest_wavelength_label_figures)
ax.axvspan(lam_0-lam_unc_0, lam_0+lam_unc_0, color='grey', alpha=0.15)
#ax.set_title(f'SUMER spectrum of the CH uncorrected and corrected with HRTS', fontsize=title_size)
#ax.set_title(f'CH-2', fontsize=title_size)
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
#ax.legend(fontsize=legend_size)
ax.set_xlabel('Wavelength (\u212B)', fontsize=axislabel_size) 
ax.set_xlim(x_lims)
plt.tight_layout()
# Add vertical lines and annotations at the upper part of the panel
_, ymax = ax.get_ylim() # Get the current Y-axis limits
for wavelength_i, symbol_i in lines_Si:
	v_i = wavelength_i
	ax.axvline(x=v_i, color='green', linestyle='--', linewidth=line_width_v2)
	#ax.text(v_i+0.2, ymax * 0.55, f'{symbol_i} - {wavelength_i}', rotation=90, verticalalignment='top', fontsize=10, color='green') # Position the text near the top of the plot panel, slightly below the max y-limit
ax.plot([],[], color='green', linestyle='--', linewidth=line_width_v2, label='Si I lines')
for wavelength_i, symbol_i in lines_Fe:
	v_i = wavelength_i
	ax.axvline(x=v_i, color='gold', linestyle='--', linewidth=line_width_v2)
	#ax.text(v_i+0.2, ymax * 0.55, f'{symbol_i} - {wavelength_i}', rotation=90, verticalalignment='top', fontsize=10, color='green') # Position the text near the top of the plot panel, slightly below the max y-limit
ax.plot([],[], color='gold', linestyle='--', linewidth=line_width_v2, label='Fe II lines')
# Big text at the right inside the panel
ax.text(
    0.8, 0.5,               # x, y in axes coordinates (0–1); >1 is outside data area but still in axes
    'CH-2',    # your text (multiline if you want)
    transform=ax.transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color=color_contours_sumer_Imap)#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
# legend in desired order:
handles, labels = ax.get_legend_handles_labels()
order = [
    labels.index('SUMER uncorrected'),
    labels.index('HRTS QS-A'),
    labels.index('HRTS QS-B'),
    labels.index('SUMER corrected, QS-A'),
    labels.index('SUMER corrected, QS-B'),
    labels.index(rest_wavelength_label_figures),
    labels.index('Si I lines'),
    labels.index('Fe II lines')]
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size-2, loc='upper left')
if save_paper_images == 'yes':
    fig_name = 'contours_SUMER__spectra_sumer_and_hrts_together_and_line_identification'
    plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



y_lims_fit = [-0.02, 1.7]


######################################################
######################################################
######################################################
######################################################
######################################################

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



# Multigaussian fit to the uncorrected profile
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=fig_size, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].errorbar(x=vkms_doppler(lamb=x_uncorrected, lamb_0=lam_0) ,y=y_uncorrected, yerr=y_unc_uncorrected, color=color_sumer_uncorrected, marker='o', linewidth=0, elinewidth=1., label='SUMER not corrected')
ax[0].plot(x_fit_uncorrected, y_fit_uncorrected, color=color_sumer_uncorrected, linestyle='-', label='Multigaussian fit', zorder=1) 
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
ax[0].axvline(x=0, color='black', linestyle='--', label=rest_wavelength_label_figures)
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax[0].set_title(f'SUMER spectrum uncorrected, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'CH-2 uncorrected', fontsize=title_size)
ax[0].set_ylim(y_lims_fit)
ax[1].axhline(y=0, linestyle=':', color='black')
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=title_size)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
ax[0].set_yscale('linear')
# legend in desired order:
handles, labels = ax[0].get_legend_handles_labels()
order = [
labels.index('SUMER not corrected'),
labels.index('Multigaussian fit'),
labels.index('Individual gaussians'),
labels.index(rest_wavelength_label_figures),]
ax[0].legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size)
ax[1].errorbar(x=vkms_doppler(lamb=x_uncorrected, lamb_0=lam_0), y=y_residuals, yerr=y_unc_residuals, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)
ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
ax[0].set_xlim(x_lims_fits)
ax[1].set_xlim(x_lims_fits)
# Big text at the right inside the panel
ax[0].text(
    0.8, 0.5,               # x, y in axes coordinates (0–1); >1 is outside data area but still in axes
    'CH-2',    # your text (multiline if you want)
    transform=ax[0].transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color=color_contours_sumer_Imap)#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
if save_paper_images == 'yes':
	fig_name = 'contours_SUMER__spectrum_multigaussian_fit_uncorrected'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)

######################################################
# Single Gaussian fit corrected Ne VIII line, QS-A

bckg_fit1_corrected_qra = -0.3
init_parameters1_corrected_qra = [bckg_fit1_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.34-bckg_fit1_corrected_qra, 0.0, 50.
]

x_corrected_qra = lam_sumer_cropNeVIII
y_corrected_qra = rad_sumer_cropNeVIII_corrected_qra
y_unc_corrected_qra = erad_sumer_cropNeVIII_corrected_qra


# Perform the fit
popt_qra, pcov_qra = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), y_corrected_qra, p0=init_parameters1_corrected_qra, sigma=y_unc_corrected_qra, absolute_sigma=True) #popt_qra are the optimized parameters. pcov_qra is the covariance matrix of the parameters. 
perr_qra = np.sqrt(np.diag(pcov_qra)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_corrected_qra = np.linspace(min(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0)), max(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0)), 300)
y_fit_corrected_qra = multigaussian_function_for_curvefit(x_fit_corrected_qra, *popt_qra)
y_fit_SUMERgrid_corrected_qra = multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), *popt_qra)

# Chi squared
chi2red_qra_single = chi2red_function(y_fit_=y_fit_SUMERgrid_corrected_qra, y_data_=y_corrected_qra, y_unc_data_=y_unc_corrected_qra, popt_=popt_qra)[0]

# Residuals
y_residuals_qra = y_corrected_qra - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), *popt_qra)
y_unc_fit_length_corrected_qra = multi_gaussian_function_uncertainties(B=popt_qra, B_unc=perr_qra, x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0))))
y_unc_residuals_qra = np.sqrt(y_unc_corrected_qra**2 + y_unc_fit_length_corrected_qra**2)


### PAPER image: multigaussian fit to the corrected profile (QR-A)
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
#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-A, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'CH-2 corrected with QS-A', fontsize=title_size)
ax[0].set_ylim(y_lims_fit)
ax[1].axhline(y=0, linestyle=':', color='black')
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=title_size)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
ax[0].legend(fontsize=legend_size)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), y=y_residuals_qra, yerr=y_unc_residuals_qra, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)
ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
ax[0].set_xlim(x_lims_fits)
ax[1].set_xlim(x_lims_fits)
# Big text at the right inside the panel
ax[0].text(
    0.8, 0.5,               # x, y in axes coordinates (0–1); >1 is outside data area but still in axes
    'CH-2',    # your text (multiline if you want)
    transform=ax[0].transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color=color_contours_sumer_Imap)#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
if save_paper_images == 'yes':
	fig_name = 'contours_SUMER__spectrum_multigaussian_fit_corrected_qra'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)
           
######################################################
# Double Gaussian fit corrected Ne VIII line, QS-A

x_corrected_qra = lam_sumer_cropNeVIII
y_corrected_qra = rad_sumer_cropNeVIII_corrected_qra
y_unc_corrected_qra = erad_sumer_cropNeVIII_corrected_qra


# Perform the fit
popt_qra, pcov_qra = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), y_corrected_qra, p0=init_parameters_corrected_qra, sigma=y_unc_corrected_qra, absolute_sigma=True) #popt_qra are the optimized parameters. pcov_qra is the covariance matrix of the parameters. 
perr_qra = np.sqrt(np.diag(pcov_qra)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_corrected_qra = np.linspace(min(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0)), max(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0)), 300)
y_fit_corrected_qra = multigaussian_function_for_curvefit(x_fit_corrected_qra, *popt_qra)
y_fit_SUMERgrid_corrected_qra = multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), *popt_qra)

# Chi squared
chi2red_qra_double = chi2red_function(y_fit_=y_fit_SUMERgrid_corrected_qra, y_data_=y_corrected_qra, y_unc_data_=y_unc_corrected_qra, popt_=popt_qra)[0]


# Residuals
y_residuals_qra = y_corrected_qra - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), *popt_qra)
y_unc_fit_length_corrected_qra = multi_gaussian_function_uncertainties(B=popt_qra, B_unc=perr_qra, x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0))))
y_unc_residuals_qra = np.sqrt(y_unc_corrected_qra**2 + y_unc_fit_length_corrected_qra**2)


### PAPER image: multigaussian fit to the corrected profile (QR-A)
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
#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-A, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'CH-2 corrected with QS-A', fontsize=title_size)
ax[0].set_ylim(y_lims_fit)
ax[1].axhline(y=0, linestyle=':', color='black')
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=title_size)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
ax[0].legend(fontsize=legend_size)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), y=y_residuals_qra, yerr=y_unc_residuals_qra, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)
ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
ax[0].set_xlim(x_lims_fits)
ax[1].set_xlim(x_lims_fits)
# Big text at the right inside the panel
ax[0].text(
    0.8, 0.5,               # x, y in axes coordinates (0–1); >1 is outside data area but still in axes
    'CH-2',    # your text (multiline if you want)
    transform=ax[0].transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color=color_contours_sumer_Imap)#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
# Reduced chi-square text on the left
ax[0].text(0.02, 0.98,  # x, y in axes coordinates (0–1); top-left corner
    r'Double: $\chi^2_{\rm red} = ' + f'{chi2red_qra_double:.1f}' + r'$' + '\n'
    r'Single: $\chi^2_{\rm red} = ' + f'{chi2red_qra_single:.1f}' + r'$', transform=ax[0].transAxes, ha='left', va='top', fontsize=chi2red_fontsize, fontweight='normal', color='black')
if save_paper_images == 'yes':
	fig_name = 'contours_SUMER__spectrum_multigaussian_fit_corrected_qra'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)




######################################################
# Single Gaussian fit corrected Ne VIII line, QS-B

x_corrected_qrb = lam_sumer_cropNeVIII
y_corrected_qrb = rad_sumer_cropNeVIII_corrected_qrb
y_unc_corrected_qrb = erad_sumer_cropNeVIII_corrected_qrb

bckg_fit1_corrected_qrb = -0.3
init_parameters1_corrected_qrb = [bckg_fit1_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.34-bckg_fit1_corrected_qrb, 0.0, 50.
]

# Perform the fit
popt_qrb, pcov_qrb = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), y_corrected_qrb, p0=init_parameters1_corrected_qrb, sigma=y_unc_corrected_qrb, absolute_sigma=True) #popt_qrb are the optimized parameters. pcov_qrb is the covariance matrix of the parameters. 
perr_qrb = np.sqrt(np.diag(pcov_qrb)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_corrected_qrb = np.linspace(min(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0)), max(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0)), 300)
y_fit_corrected_qrb = multigaussian_function_for_curvefit(x_fit_corrected_qrb, *popt_qrb)
y_fit_SUMERgrid_corrected_qrb = multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), *popt_qrb)

# Chi squared
chi2red_qrb_single = chi2red_function(y_fit_=y_fit_SUMERgrid_corrected_qrb, y_data_=y_corrected_qrb, y_unc_data_=y_unc_corrected_qrb, popt_=popt_qrb)[0]



# Residuals
y_residuals_qrb = y_corrected_qrb - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), *popt_qrb)
y_unc_fit_length_corrected_qrb = multi_gaussian_function_uncertainties(B=popt_qrb, B_unc=perr_qrb, x=vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0))))
y_unc_residuals_qrb = np.sqrt(y_unc_corrected_qrb**2 + y_unc_fit_length_corrected_qrb**2)



### PAPER image: multigaussian fit to the corrected profile (QR-B)
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
#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-B, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'CH-2 corrected with QS-B', fontsize=title_size)
ax[0].set_ylim(y_lims_fit)
ax[1].axhline(y=0, linestyle=':', color='black')
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=title_size)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
ax[0].legend(fontsize=legend_size)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), y=y_residuals_qrb, yerr=y_unc_residuals_qrb, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)
ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
ax[0].set_xlim(x_lims_fits)
ax[1].set_xlim(x_lims_fits)
# Big text at the right inside the panel
ax[0].text(
    0.8, 0.5,               # x, y in axes coordinates (0–1); >1 is outside data area but still in axes
    'CH-2',    # your text (multiline if you want)
    transform=ax[0].transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color=color_contours_sumer_Imap)#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
if save_paper_images == 'yes':
	fig_name = 'contours_SUMER__spectrum_multigaussian_fit_corrected_qrb'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)


######################################################
# Double Gaussian fit corrected Ne VIII line, QS-B

x_corrected_qrb = lam_sumer_cropNeVIII
y_corrected_qrb = rad_sumer_cropNeVIII_corrected_qrb
y_unc_corrected_qrb = erad_sumer_cropNeVIII_corrected_qrb


# Perform the fit
popt_qrb, pcov_qrb = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), y_corrected_qrb, p0=init_parameters_corrected_qrb, sigma=y_unc_corrected_qrb, absolute_sigma=True) #popt_qrb are the optimized parameters. pcov_qrb is the covariance matrix of the parameters. 
perr_qrb = np.sqrt(np.diag(pcov_qrb)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_corrected_qrb = np.linspace(min(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0)), max(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0)), 300)
y_fit_corrected_qrb = multigaussian_function_for_curvefit(x_fit_corrected_qrb, *popt_qrb)
y_fit_SUMERgrid_corrected_qrb = multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), *popt_qrb)

# Chi squared
chi2red_qrb_double = chi2red_function(y_fit_=y_fit_SUMERgrid_corrected_qrb, y_data_=y_corrected_qrb, y_unc_data_=y_unc_corrected_qrb, popt_=popt_qrb)[0]


# Residuals
y_residuals_qrb = y_corrected_qrb - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), *popt_qrb)
y_unc_fit_length_corrected_qrb = multi_gaussian_function_uncertainties(B=popt_qrb, B_unc=perr_qrb, x=vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0))))
y_unc_residuals_qrb = np.sqrt(y_unc_corrected_qrb**2 + y_unc_fit_length_corrected_qrb**2)



### PAPER image: multigaussian fit to the corrected profile (QR-B)
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
#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-B, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'CH-2 corrected with QS-B', fontsize=title_size)
ax[0].set_ylim(y_lims_fit)
ax[1].axhline(y=0, linestyle=':', color='black')
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=title_size)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
ax[0].legend(fontsize=legend_size)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), y=y_residuals_qrb, yerr=y_unc_residuals_qrb, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)
ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
ax[0].set_xlim(x_lims_fits)
ax[1].set_xlim(x_lims_fits)
# Big text at the right inside the panel
ax[0].text(
    0.8, 0.5,               # x, y in axes coordinates (0–1); >1 is outside data area but still in axes
    'CH-2',    # your text (multiline if you want)
    transform=ax[0].transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color=color_contours_sumer_Imap)#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
    # Reduced chi-square text on the left
ax[0].text(0.02, 0.98,  # x, y in axes coordinates (0–1); top-left corner
    r'Double: $\chi^2_{\rm red} = ' + f'{chi2red_qra_double:.1f}' + r'$' + '\n'
    r'Single: $\chi^2_{\rm red} = ' + f'{chi2red_qra_single:.1f}' + r'$', transform=ax[0].transAxes, ha='left', va='top', fontsize=chi2red_fontsize, fontweight='normal', color='black')
if save_paper_images == 'yes':
	fig_name = 'contours_SUMER__spectrum_multigaussian_fit_corrected_qrb'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
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



### PAPER image: multigaussian fit to the corrected profile (QR-L)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=fig_size, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0) ,y=y_corrected_qrl, yerr=y_unc_corrected_qrl, color=color_hrts_qrl, marker='o', linewidth=0, elinewidth=1., label='SUMER corrected QS-L')
ax[0].plot(x_fit_corrected_qrl, y_fit_corrected_qrl, color=color_hrts_qrl, linestyle='-', label='Fit', zorder=1) 
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
#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-L, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'CH-2 corrected with QS-L', fontsize=title_size)
ax[0].set_ylim(y_lims_fit)
ax[1].axhline(y=0, linestyle=':', color='black')
#ax[0].set_title(f'SUMER {range_percentage}%, corrected', fontsize=title_size)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
ax[0].legend(fontsize=legend_size)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0), y=y_residuals_qrl, yerr=y_unc_residuals_qrl, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)
ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
ax[0].set_xlim(x_lims_fits)
ax[1].set_xlim(x_lims_fits)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
if save_paper_images == 'yes':
	fig_name = 'contours_SUMER__spectrum_multigaussian_fit_corrected_qrl'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)


#############################################


#xb_uncorrected, yb_uncorrected = find_bisector(x_data=x_uncorrected[3:-7], y_data=y_uncorrected[3:-7], y_unc_data=y_unc_uncorrected[3:-7], y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')

#xb_corrected, yb_corrected = find_bisector(x_data=x_corrected, y_data=y_corrected, y_unc_data=y_unc_corrected, y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')


#############################################

print('chi2red_qra_single: ', chi2red_qra_single)
print('chi2red_qra_double: ', chi2red_qra_double)
print('chi2red_qrb_single: ', chi2red_qrb_single)
print('chi2red_qrb_double: ', chi2red_qrb_double)




