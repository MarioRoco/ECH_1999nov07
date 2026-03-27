
# Binning
bin_lat = 4
bin_lon = 1


lat_img__indices_binned = 46, 156
lat_img__indices = bin_lat*lat_img__indices_binned[0], bin_lon*lat_img__indices_binned[1]

wavelength_range_analysis = [1539., 1543.] #Angstroem

BR_distance_centroid = 50. #[km/s] Distance of the center of the range from the centroid of the fitted gaussian
BR_width = 50. #[km/s]

line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', 'cold_line'

# EIT
eit_wavelength = 195 #171, 195, 284, or 304 [Angstrom]
eit_time = 'late' #'early' or 'late' (early: around 1 or 4 am; late: around 6 or 7 am)

############################################################
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
# Import NOT binned data

## Spectral images
data_interpolated_loaded = np.load('../data/data_modified/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)
spectral_image_interpolated_list = data_interpolated_loaded['spectral_image_interpolated_list']
spectral_image_unc_interpolated_list = data_interpolated_loaded['spectral_image_unc_interpolated_list']
spectral_image_interpolated_croplat_list = data_interpolated_loaded['spectral_image_interpolated_croplat_list']
spectral_image_unc_interpolated_croplat_list = data_interpolated_loaded['spectral_image_unc_interpolated_croplat_list']
lam_sumer = data_interpolated_loaded['reference_wavelength']          # scalar (0‑d array; use a_loaded.item() for Python float)
lam_sumer_unc = data_interpolated_loaded['unc_reference_wavelength'] #uncertainty of lam_sumer
row_reference = int(data_interpolated_loaded['row_reference'])        # becomes a NumPy array or object array, so I conver it to integer again


## Intensity map
intensitymap_loaded_dic = np.load('../outputs/intensity_map_'+line_label+'_interpolated.npz')
intensity_map = intensitymap_loaded_dic['intensity_map'] #2D-array
intensity_map_unc = intensitymap_loaded_dic['intensity_map_unc'] #2D-array
intensity_map_croplat = intensitymap_loaded_dic['intensity_map_croplat'] #2D-array
intensity_map_unc_croplat = intensitymap_loaded_dic['intensity_map_unc_croplat'] #2D-array
line_center_label = intensitymap_loaded_dic['line_center_label'] 
vmin_sumer, vmax_sumer = intensitymap_loaded_dic['vmin_vmax'] 

############################################################
# Import binned data

## Spectral images binned
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


## Intensity map binned
intensitymap_loaded_dic = np.load('../outputs/intensity_map_'+line_label+f'_binned_lon{bin_lon}_lat{bin_lat}.npz')
intensity_map_croplat_binned = intensitymap_loaded_dic['intensity_map_croplat_binned'] #[W/sr/m^2] 2D-array
intensity_map_unc_croplat_binned = intensitymap_loaded_dic['intensity_map_unc_croplat_binned'] #[W/sr/m^2] 2D-array


# Dopplershift map binned
dopplershift_map_loaded_dic = np.load('../outputs/dopplershift_map_'+line_label+f'_binned_lon{bin_lon}_lat{bin_lat}.npz')
dopplershift_map_binned = dopplershift_map_loaded_dic['dopplershift_map_binned'] #2D-array
dopplershift_map_binned_HRTSsub_qra = dopplershift_map_loaded_dic['dopplershift_map_binned_HRTSsub_qra'] #2D-array
dopplershift_map_binned_HRTSsub_qrb = dopplershift_map_loaded_dic['dopplershift_map_binned_HRTSsub_qrb'] #2D-array
dopplershift_map_binned_HRTSsub_qrl = dopplershift_map_loaded_dic['dopplershift_map_binned_HRTSsub_qrl'] #2D-array
rest_wavelength_label = dopplershift_map_loaded_dic['rest_wavelength_label']
rest_wavelength = dopplershift_map_loaded_dic['rest_wavelength']
rest_wavelength_unc = dopplershift_map_loaded_dic['rest_wavelength_unc']
rest_velocity_unc = dopplershift_map_loaded_dic['rest_velocity_unc']
# dopplershift_map_binned_HRTSsub_qra_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qra)


## Subtract median for every row
dopplershift_map_binned_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned)
dopplershift_map_binned_HRTSsub_qra_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qra)
dopplershift_map_binned_HRTSsub_qrb_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qrb)
dopplershift_map_binned_HRTSsub_qrl_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qrl)



## BR asymmetry map binned
data_id = f'_binned_lon{bin_lon}_lat{bin_lat}__d{int(BR_distance_centroid)}_w{int(BR_width)}'
### (R-B)/(R+B)
BRasymmetry_map_subtraction_loaded_dic = np.load('../outputs/BRasymmetry_map_subtraction_'+line_label+data_id+'.npz')
BRasymmetry_map_gaussian_binned_normalized = BRasymmetry_map_subtraction_loaded_dic['BRasymmetry_map_gaussian_binned_normalized'] # 2D-array
BRasymmetry_map_gaussian_binned_HRTSsub_qra_normalized = BRasymmetry_map_subtraction_loaded_dic['BRasymmetry_map_gaussian_binned_HRTSsub_qra_normalized'] # 2D-array
BRasymmetry_map_gaussian_binned_HRTSsub_qrb_normalized = BRasymmetry_map_subtraction_loaded_dic['BRasymmetry_map_gaussian_binned_HRTSsub_qrb_normalized'] # 2D-array
BRasymmetry_map_gaussian_binned_HRTSsub_qrl_normalized = BRasymmetry_map_subtraction_loaded_dic['BRasymmetry_map_gaussian_binned_HRTSsub_qrl_normalized'] # 2D-array
### R/B
BRasymmetry_map_division_loaded_dic = np.load('../outputs/BRasymmetry_map_division_'+line_label+data_id+'.npz')
BRasymmetry_map_gaussian_binned_division = BRasymmetry_map_division_loaded_dic['BRasymmetry_map_gaussian_binned_division'] # 2D-array
BRasymmetry_map_gaussian_binned_HRTSsub_qra_division = BRasymmetry_map_division_loaded_dic['BRasymmetry_map_gaussian_binned_HRTSsub_qra_division'] # 2D-array
BRasymmetry_map_gaussian_binned_HRTSsub_qrb_division = BRasymmetry_map_division_loaded_dic['BRasymmetry_map_gaussian_binned_HRTSsub_qrb_division'] # 2D-array
BRasymmetry_map_gaussian_binned_HRTSsub_qrl_division = BRasymmetry_map_division_loaded_dic['BRasymmetry_map_gaussian_binned_HRTSsub_qrl_division'] # 2D-array


## Scale factor map
scalefactor_loaded_dic = np.load(f'../outputs/scalefactor_map_interpolated_binned_lon{bin_lon}_lat{bin_lat}.npz')

scaling_factor_map_binned_qra = scalefactor_loaded_dic['scaling_factor_map_binned_qra'] #[W/sr/m^2] 2D-array
scaling_factor_map_binned_qrb = scalefactor_loaded_dic['scaling_factor_map_binned_qrb'] #[W/sr/m^2] 2D-array
scaling_factor_map_binned_qrl = scalefactor_loaded_dic['scaling_factor_map_binned_qrl'] #[W/sr/m^2] 2D-array

wavelength_range_scalefactor_left = scalefactor_loaded_dic['wavelength_range_scalefactor_left'] #Angstrom
wavelength_range_scalefactor_right = scalefactor_loaded_dic['wavelength_range_scalefactor_right'] #Angstrom
fwhm_to_convolve = scalefactor_loaded_dic['fwhm_to_convolve'] #Angstrom
rad_hrtsa_conv = scalefactor_loaded_dic['rad_hrtsa_conv']
rad_hrtsb_conv = scalefactor_loaded_dic['rad_hrtsb_conv']
rad_hrtsl_conv = scalefactor_loaded_dic['rad_hrtsl_conv']
lam_hrts_SUMERgrid = scalefactor_loaded_dic['lam_sumer']
rad_hrtsa_conv_SUMERgrid = scalefactor_loaded_dic['rad_hrtsa_conv_SUMERgrid']
rad_hrtsb_conv_SUMERgrid = scalefactor_loaded_dic['rad_hrtsb_conv_SUMERgrid']
rad_hrtsl_conv_SUMERgrid = scalefactor_loaded_dic['rad_hrtsl_conv_SUMERgrid']

############################################################
# Import EIT image in the FOV of SUMER

exec(open("bin2_EIT_map.py").read())

"""
# Outputs:
header_eit
data_eit
data_eit_crop_corrected

extent_eit_px_uncorrected_contours
extent_eit_px_contours

extent_eit_sumer_arcsec_image
extent_eit_sumer_arcsec_contours

closest_index
closest_time_sumer
time_eit
hour_eit
HPlon_rotcomp
HPlon
HPlat
HPlat_croplat
"""

############################################################
#  Extent in pixels 

extent_sumer_px_contours = [0., intensity_map_croplat.shape[1]-1, intensity_map_croplat.shape[0]-1, 0.]
extent_sumer_px_image = [-0.5, intensity_map_croplat.shape[1]-1+0.5, intensity_map_croplat.shape[0]-1+0.5, -0.5]

extent_sumer_binned_px_contours = [0., intensity_map_croplat_binned.shape[1]-1, intensity_map_croplat_binned.shape[0]-1, 0.]
extent_sumer_binned_px_image = [-0.5, intensity_map_croplat_binned.shape[1]-1+0.5, intensity_map_croplat_binned.shape[0]-1+0.5, -0.5]

extent_eit_px_contours = [0., intensity_map_croplat.shape[1]-1, intensity_map_croplat.shape[0]-1, 0.]
extent_eit_px_image = [-0.5, intensity_map_croplat.shape[1]-1+0.5, intensity_map_croplat.shape[0]-1+0.5, -0.5]


# Extents in arcsec
lat_half_bottom = abs((HPlat_croplat[1]-HPlat_croplat[0])/2.)
lat_half_top = abs((HPlat_croplat[-1]-HPlat_croplat[-2])/2.)
lon_half_left = abs((HPlon_rotcomp[1]-HPlon_rotcomp[0])/2.)
lon_half_right = abs((HPlon_rotcomp[-1]-HPlon_rotcomp[-2])/2.)
extent_eit_sumer_arcsec_image = [HPlon_rotcomp[0]-lon_half_left, HPlon_rotcomp[-1]+lon_half_right, HPlat_croplat[-1]-lat_half_bottom, HPlat_croplat[0]+lat_half_top] #arcsec
extent_eit_sumer_arcsec_contours = [HPlon_rotcomp[0], HPlon_rotcomp[-1], HPlat_croplat[-1], HPlat_croplat[0]] #arcsec


############################################################
############################################################
############################################################
# 

color_pixel = 'lime'

vmin_vmax__dopplermap = [-12, 12]
vmin_vmax__BRmap = [-1., 1.]
label_size = 18


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
ax.imshow(dopplershift_map_binned_HRTSsub_qra_lessmedian, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
ax.set_title('Dopplershift map, HRTS: QR-A, median subtracted', fontsize=18)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
row, col = lat_img__indices_binned
rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
ax.add_patch(rect)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
plt.show(block=False)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
ax.imshow(BRasymmetry_map_gaussian_binned_HRTSsub_qra_normalized, cmap='seismic', vmin=vmin_vmax__BRmap[0], vmax=vmin_vmax__BRmap[1], aspect='auto')
ax.set_title('BR asymmetry map, HRTS: QR-A', fontsize=18)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
row, col = lat_img__indices_binned
rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
ax.add_patch(rect)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
plt.show(block=False)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
ax.imshow(intensity_map_croplat_binned, cmap='Greys_r', norm=LogNorm(), aspect='auto')
ax.set_title('Intensity map binned, HRTS: QR-A', fontsize=18)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
row, col = lat_img__indices_binned
rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
ax.add_patch(rect)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
plt.show(block=False)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
ax.imshow(intensity_map_croplat, cmap='Greys_r', norm=LogNorm(), aspect='auto')
ax.set_title('Intensity map, HRTS: QR-A', fontsize=18)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
row, col = lat_img__indices
rect = patches.Rectangle((col-0.5, row-0.5), 1, 1, linewidth=1.5, edgecolor=color_pixel, facecolor='none', label=f'row, col: {row}, {col}')
ax.add_patch(rect)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
plt.show(block=False)

############################################################
# 

r_binned, c_binned = lat_img__indices_binned
r, c = lat_img__indices

## binned
spectral_image_binned = spectral_image_interpolated_croplat_binned_list[c_binned]
unc_spectral_image_binned = spectral_image_unc_interpolated_croplat_binned_list[c_binned]
x_profile_binned = lam_sumer
y_profile_binned = spectral_image_binned[r_binned, :]
yerr_profile_binned = unc_spectral_image_binned[r_binned, :]

## not binned
spectral_image = spectral_image_interpolated_croplat_list[c]
unc_spectral_image = spectral_image_unc_interpolated_croplat_list[c]
x_profile = x_profile_binned = lam_sumer
y_profile = spectral_image[r, :]
yerr_profile = unc_spectral_image[r, :]

## Scaling the HRTS convolved profiles
scaling_factor_qra = scaling_factor_map_binned_qra[r_binned,c_binned]
scaling_factor_qrb = scaling_factor_map_binned_qrb[r_binned,c_binned]
scaling_factor_qrl = scaling_factor_map_binned_qrl[r_binned,c_binned]
rad_hrtsa_conv_scaled = scaling_factor_qra * rad_hrtsa_conv
rad_hrtsb_conv_scaled = scaling_factor_qrb * rad_hrtsb_conv
rad_hrtsl_conv_scaled = scaling_factor_qrl * rad_hrtsl_conv
rad_hrtsa_conv_SUMERgrid_scaled = scaling_factor_qra * rad_hrtsa_conv_SUMERgrid
rad_hrtsb_conv_SUMERgrid_scaled = scaling_factor_qrb * rad_hrtsb_conv_SUMERgrid
rad_hrtsl_conv_SUMERgrid_scaled = scaling_factor_qrl * rad_hrtsl_conv_SUMERgrid

## Crop spectra in a shorter range around Ne VIII
### SUMER
lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_analysis)
#### Not binned
x_profile_crop = x_profile[idx_sumer_crop[0]:idx_sumer_crop[1]+1]
y_profile_crop = y_profile[idx_sumer_crop[0]:idx_sumer_crop[1]+1]
yerr_profile_crop = yerr_profile[idx_sumer_crop[0]:idx_sumer_crop[1]+1]
#### Binned
x_profile_binned_crop = x_profile_binned[idx_sumer_crop[0]:idx_sumer_crop[1]+1]
y_profile_binned_crop = y_profile_binned[idx_sumer_crop[0]:idx_sumer_crop[1]+1]
yerr_profile_binned_crop = yerr_profile_binned[idx_sumer_crop[0]:idx_sumer_crop[1]+1]
### HRTS
lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrtsa, range_values=wavelength_range_analysis)
rad_hrtsa_conv_scaled_crop = rad_hrtsa_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]
rad_hrtsb_conv_scaled_crop = rad_hrtsb_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]
rad_hrtsl_conv_scaled_crop = rad_hrtsl_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]
### HRTS in SUMER grid
lam_hrts_SUMERgrid_crop, idx_hrtsSG_crop = crop_range(list_to_crop=lam_hrts_SUMERgrid, range_values=wavelength_range_analysis)
rad_hrtsa_conv_SUMERgrid_scaled_crop = rad_hrtsa_conv_SUMERgrid_scaled[idx_hrtsSG_crop[0]:idx_hrtsSG_crop[1]+1]
rad_hrtsb_conv_SUMERgrid_scaled_crop = rad_hrtsb_conv_SUMERgrid_scaled[idx_hrtsSG_crop[0]:idx_hrtsSG_crop[1]+1]
rad_hrtsl_conv_SUMERgrid_scaled_crop = rad_hrtsl_conv_SUMERgrid_scaled[idx_hrtsSG_crop[0]:idx_hrtsSG_crop[1]+1]

## Substract HRTS to SUMER
#### Binned
y_profile_binned_crop_corrected_qra = y_profile_binned_crop - rad_hrtsa_conv_SUMERgrid_scaled_crop
y_profile_binned_crop_corrected_qrb = y_profile_binned_crop - rad_hrtsb_conv_SUMERgrid_scaled_crop
y_profile_binned_crop_corrected_qrl = y_profile_binned_crop - rad_hrtsl_conv_SUMERgrid_scaled_crop
yerr_profile_binned_crop_corrected_qra = yerr_profile_binned_crop
yerr_profile_binned_crop_corrected_qrb = yerr_profile_binned_crop
yerr_profile_binned_crop_corrected_qrl = yerr_profile_binned_crop



############################################################
# 

lamb_0 = rest_wavelength
def lam_to_vkms(lamb): return 299792.4580*(lamb-lamb_0)/lamb_0 #[km/s]
def vkms_to_lam(v_kms): return lamb_0*((v_kms/299792.4580)+1) #velocity in [km/s]



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
ax.errorbar(x=x_profile_binned_crop, y=y_profile_binned_crop, yerr=yerr_profile_binned_crop, color='black', linewidth=1, label='SUMER uncorrected')
ax.errorbar(x=x_profile_binned_crop, y=y_profile_binned_crop_corrected_qra, yerr=yerr_profile_binned_crop_corrected_qra, color='blue', linewidth=1, label='SUMER corrected')
ax.errorbar(x=lam_hrts_crop, y=rad_hrtsa_conv_scaled_crop, color='green', linewidth=1, label='HRTS convolved and scaled')
ax.errorbar(x=lam_hrts_SUMERgrid_crop, y=rad_hrtsa_conv_SUMERgrid_scaled_crop, color='green', linewidth=0, marker='.', markersize=5, label='HRTS, SUMER grid')
ax.set_title(f'Profile of one pixel binned. row, col = ({r_binned}, {c_binned})', fontsize=18)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
ax2 = ax.secondary_xaxis('top', functions=(lam_to_vkms, vkms_to_lam))
ax2.set_xlabel('Doppler velocity [km/s]', fontsize=17)
ax.axvline(x=rest_wavelength, color='brown', label=f'Rest wavelength: {vkms_to_lam}'r' $\pm$ 'f'{rest_wavelength}'' \u212B')
ax.axvspan(rest_wavelength-rest_wavelength_unc, rest_wavelength+rest_wavelength_unc, color='brown', alpha=0.15)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
plt.show(block=False)




# TODO: if you want to subract HRTS for the not binned data, you have to calculate the scaling factor
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
ax.errorbar(x=x_profile_crop, y=y_profile_crop, yerr=yerr_profile_crop, color='black', linewidth=1, label='SUMER uncorrected') 
ax.set_title(f'Profile of one pixel (not binned). row, col = ({r}, {c})', fontsize=18)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
ax2 = ax.secondary_xaxis('top', functions=(lam_to_vkms, vkms_to_lam))
ax2.set_xlabel('Doppler velocity [km/s]', fontsize=17)
ax.axvline(x=rest_wavelength, label=f'Rest wavelength: {vkms_to_lam}'r' $\pm$ 'f'{rest_wavelength}'' \u212B')
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
plt.show(block=False)


############################################################
# 






############################################################
# 









