# INPUTS

# Binning
bin_lat = 4
bin_lon = 1

line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', 'cold_line'

wavelength_range_analysis = [1540.1, 1541.5] #Angstroem

save_intensity_map = 'no'

show_spectral_image_binned = 'yes'
show_wavelength_range = 'yes'
show_intensitymap_binned = 'yes'

show_spectral_image_binned = 'no'
show_spectral_ranges = 'no'
show_intensity_map_binned = 'yes'
show_scaling_factor_maps = 'no'
show_dopplermaps = 'yes'
show_dopplermaps_lessmedian = 'yes'
show_chi2red_of_dopplermaps = 'no'

line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line':

filename_eit = 'SOHO_EIT_195_19991107T042103_L1.fits' #for the binning of the coordinates (where solar rotation has been corrected)

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
# Scaling factor array
exec(open("bin_scalefactor_map.py").read())

# Outputs:
"""
scaling_factor_map_binned_qra, scaling_factor_map_binned_qrb, scaling_factor_map_binned_qrl
chi2red_map_binned_qra, chi2red_map_binned_qrb, chi2red_map_binned_qrl
# Radiances
rad_hrtsa_conv, rad_hrtsb_conv, rad_hrtsl_conv
interp_func_hrtsa, interp_func_hrtsb, interp_func_hrtsl
rad_hrtsa_conv_SUMERgrid, rad_hrtsb_conv_SUMERgrid, rad_hrtsl_conv_SUMERgrid
# Uncertainties of the radiance
erad_hrtsa_conv, erad_hrtsb_conv, erad_hrtsl_conv
interp_func_hrtsa_err, interp_func_hrtsb_err, interp_func_hrtsl_err
erad_hrtsa_conv_SUMERgrid, erad_hrtsb_conv_SUMERgrid, erad_hrtsl_conv_SUMERgrid
# Left
idx_left_sumer_0, idx_left_sumer_1
lam_sumer_cropleft
rad_hrtsa_conv_SUMERgrid_cropleft, rad_hrtsb_conv_SUMERgrid_cropleft, rad_hrtsl_conv_SUMERgrid_cropleft
erad_hrtsa_conv_SUMERgrid_cropleft, erad_hrtsb_conv_SUMERgrid_cropleft, erad_hrtsl_conv_SUMERgrid_cropleft
# Right
idx_right_sumer_0, idx_right_sumer_1
lam_sumer_cropright
rad_hrtsa_conv_SUMERgrid_cropright, rad_hrtsb_conv_SUMERgrid_cropright, rad_hrtsl_conv_SUMERgrid_cropright
erad_hrtsa_conv_SUMERgrid_cropright, erad_hrtsb_conv_SUMERgrid_cropright, erad_hrtsl_conv_SUMERgrid_cropright
# Concatenate left and right
lam_sumer_cropscale
rad_hrtsa_conv_SUMERgrid_cropscale, rad_hrtsb_conv_SUMERgrid_cropscale, rad_hrtsl_conv_SUMERgrid_cropscale
erad_hrtsa_conv_SUMERgrid_cropscale, erad_hrtsb_conv_SUMERgrid_cropscale, erad_hrtsl_conv_SUMERgrid_cropscale
"""

############################################################

## Ranges of wavelength
if line_label == 'NeVIII':
    wavelength_range_intensity_map = [1540.45, 1541.2] #Angstroem
    wavelength_range_intensity_map_bckg = [1539.8, 1540.2] #Angstroem
    line_center_label = 'Ne VIII - 770.428 \u212B'
elif line_label == 'SiII':
    wavelength_range_intensity_map = [1533.075, 1533.805] #Angstroem
    #wavelength_range_intensity_map = [1533.17, 1533.725] #Angstroem
    wavelength_range_intensity_map_bckg = [1535.10, 1536.60] #Angstroem
    line_center_label = 'Si II - 1533.43 \u212B'
elif line_label == 'CIV':
    wavelength_range_intensity_map = [1547.90, 1548.66] #Angstroem
    wavelength_range_intensity_map_bckg = [1545.93, 1547.65] #Angstroem
    line_center_label = 'C IV - 1548.21 \u212B'
elif line_label == 'cold_line':
    wavelength_range_intensity_map = [1537.80, 1538.10] #Angstroem
    wavelength_range_intensity_map_bckg = [1538.30, 1538.39] #Angstroem
    line_center_label = 'Si I - 1537.94 \u212B'


############################################################

# Rest wavelength used
rest_wavelength_label = 'Peter_and_Judge_1999' #'SUMER_atlas', 'Peter_1998', 'Dammasch_1999', 'Peter_and_Judge_1999', 'Kelly_database'
lam_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][0] #Angstrom
lam_unc_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][1] #Angstrom
print('Rest wavelength Ne VIII (2nd order):', lam_0, r'$\pm$', lam_unc_0, '\u212B')

# uncertainty of the rest wavelength in km/s
v_unc_0 = vkms_doppler_unc(lamb=lam_0, lamb_unc=lam_unc_0, lamb_0=lam_0, lamb_0_unc=lam_unc_0) 

############################################################
# 1) Calculate Dopplermap, 2) Plot Dopplermap (fitting a single Gaussian) 3) Plot map of chi^2_red

## HRTS not subtracted
dopplershift_map_binned, chi2red_map_binned = create_SUMER_dopplermap_single_gaussianfit(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_binned_list, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_binned_list, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', show__row_col='no', y_scale='linear', show_legend='no') #show__row_col=lat_img__indices_binned
## HRTS subtracted, QR-A
dopplershift_map_binned_HRTSsub_qra, chi2red_map_binned_HRTSsub_qra = create_SUMER_dopplermap_single_gaussianfit(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_binned_list, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_binned_list, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, subtract_HRTS='yes', lam_hrts=lam_hrtsa, rad_hrts=rad_hrtsa, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qra, show__row_col='no', y_scale='linear', show_legend='no') #show__row_col=lat_img__indices_binned
## HRTS subtracted, QR-B
dopplershift_map_binned_HRTSsub_qrb, chi2red_map_binned_HRTSsub_qrb = create_SUMER_dopplermap_single_gaussianfit(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_binned_list, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_binned_list, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, subtract_HRTS='yes', lam_hrts=lam_hrtsb, rad_hrts=rad_hrtsb, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qrb, show__row_col='no', y_scale='linear', show_legend='no') #show__row_col=lat_img__indices_binned
## HRTS subtracted, QR-L
dopplershift_map_binned_HRTSsub_qrl, chi2red_map_binned_HRTSsub_qrl = create_SUMER_dopplermap_single_gaussianfit(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_binned_list, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_binned_list, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, subtract_HRTS='yes', lam_hrts=lam_hrtsl, rad_hrts=rad_hrtsl, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qrl, show__row_col='no', y_scale='linear', show_legend='no') #show__row_col=lat_img__indices_binned


############################################################
# Subtract median for each row
dopplershift_map_binned_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned)
dopplershift_map_binned_HRTSsub_qra_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qra)
dopplershift_map_binned_HRTSsub_qrb_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qrb)
dopplershift_map_binned_HRTSsub_qrl_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qrl)

############################################################
# 
lower_bound, upper_bound = get_bounds(intensitymap_croplat=intensitymap_croplat, range_percentage=range_percentage, threshold_value_type=threshold_value_type)

############################################################
#  Extent in pixels 
#extent_sumer_px_contours = [0., intensitymap_croplat.shape[1]-1, intensitymap_croplat.shape[0]-1, 0.]
#extent_sumer_px_image = [-0.5, intensitymap_croplat.shape[1]-1+0.5, intensitymap_croplat.shape[0]-1+0.5, -0.5]
#extent_sumer_binned_px_contours = [0., intensitymap_croplat_binned.shape[1]-1, intensitymap_croplat_binned.shape[0]-1, 0.]
#extent_sumer_binned_px_image = [-0.5, intensitymap_croplat_binned.shape[1]-1+0.5, intensitymap_croplat_binned.shape[0]-1+0.5, -0.5]

extent_binned_px_image = [-0.5, dopplershift_map_binned.shape[1]-1+0.5, dopplershift_map_binned.shape[0]-1+0.5, -0.5]

############################################################
# Show images

if show_dopplermaps == 'yes':
    vmin_vmax__dopplermap = [-12, 12]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qra, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qrb, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-B', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qrl, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-L', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)
    

if show_dopplermaps_lessmedian == 'yes':
    vmin_vmax__dopplermap = [-12, 12]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_lessmedian, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qra_lessmedian, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-A, median subtracted', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qrb_lessmedian, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-B, median subtracted', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qrl_lessmedian, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-L, median subtracted', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)



if show_chi2red_of_dopplermaps == 'yes':

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(chi2red_map_binned, cmap='Greys_r', aspect='auto')
    ax.set_title(r'$chi^2_{\rm red}$ of the Dopplershift map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(chi2red_map_binned_HRTSsub_qra, cmap='Greys_r', aspect='auto')
    ax.set_title(r'$chi^2_{\rm red}$ of the Dopplershift map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(chi2red_map_binned_HRTSsub_qrb, cmap='Greys_r', aspect='auto')
    ax.set_title(r'$chi^2_{\rm red}$ of the Dopplershift map, HRTS: QR-B', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(chi2red_map_binned_HRTSsub_qrl, cmap='Greys_r', aspect='auto')
    ax.set_title(r'$chi^2_{\rm red}$ of the Dopplershift map, HRTS: QR-L', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)



############################################################

print('--------------------------------------')
print('rest_wavelength_label')
print('lam_0')
print('lam_unc_0')
print('v_unc_0')
print('dopplershift_map_binned')
print('chi2red_map_binned')
print('dopplershift_map_binned_HRTSsub_qra')
print('dopplershift_map_binned_HRTSsub_qrb')
print('dopplershift_map_binned_HRTSsub_qrl')
print('dopplershift_map_binned_HRTSsub_qra_lessmedian')
print('dopplershift_map_binned_HRTSsub_qrb_lessmedian')
print('dopplershift_map_binned_HRTSsub_qrl_lessmedian')
print('chi2red_map_binned_HRTSsub_qra')
print('chi2red_map_binned_HRTSsub_qrb')
print('chi2red_map_binned_HRTSsub_qrl')
print('--------------------------------------')

############################################################



