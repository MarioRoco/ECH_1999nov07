
save_paper_images = 'yes'
folder_name = '../outputs/paper_figures/profiles_from_polygons/v4' #name of the folder where you save the images
save_dpi = 100 #resolution: number of pixels per inch. ChatGPT gave me 300 by default. 

#  Inputs

# polygon vertices are given as (row, col)
#poly_rc = [[], [], [], []]

## Quiet Sun regions
#poly_rc, region_id = [[68,104], [28,104], [28,130], [68,130]], 'QSa1' #QS a1
#poly_rc, region_id = [[28,130], [68,130], [68,160], [28,160]], 'QSa2' #QS a2
#poly_rc, region_id = [[68,160], [28,160], [28,190], [68,190]], 'QSa3' #QS a3
#poly_rc, region_id = [[68,104], [28,104], [28,190], [68,190]], 'QSa4' #QS a4
#poly_rc, region_id = [[66,122], [16,122], [16,176], [66,178]], 'QSa5' #QS a5
poly_rc, region_id = [[12,104], [12,199], [83,180], [83,127]], 'QSa6' #QS a6
#poly_rc, region_id = [[115,3], [140,27], [185,27], [167,3]], 'QSb' #QS b
#poly_rc, region_id = [[302,178], [221,199], [221,231], [302,231]], 'QSc1' #QS c1
#poly_rc, region_id = [[221,199], [221,231], [97,231]], 'QSc2' #QS c2
#poly_rc, region_id = [[302,178], [97,231], [302,231]], 'QSc3' #QS c3


## From Dopplermap
#poly_rc, region_id = [[285,144], [272,145], [272,148], [282,148]], 'bluein' #blue patch inside CH
#poly_rc, region_id = [[272,155], [263,155], [265,159], [271,158]], 'redin' #red patch in border of CH
#poly_rc, region_id = [[79,94], [81,97], [91,97], [91,94]], 'blueborder' #blue patch in border of CH
#poly_rc, region_id = [[17,15], [18,19], [30,18], [30,14]], 'bluein' #blue patch inside CH
#poly_rc, region_id = [[274,159], [278,165], [292,168], [290,160]], 'redout' #red patch outside CH
#poly_rc, region_id = [[236,5], [236,10], [250,10], [251,7]], 'redout' #red patch outside CH
#poly_rc, region_id = [[199,225], [203,228], [205,228], [205,224]], 'blueout' #blue patch outside CH
#poly_rc, region_id = [[183,141], [182,145], [167,145], [167,143]], 'redin' #red patch in border of CH
#poly_rc, region_id = [[203,143], [202,144], [208,144], [208,142]], 'blueout' #blue patch outside CH
#poly_rc, region_id = [[168,143], [169,146], [180,145], [180,142]], 'redin' #red patch in border of CH
#poly_rc, region_id = [[29,133], [29,136], [37,136], [37,133]], 


##From BR asymmetry map
#poly_rc, region_id = [[223,80], [223,81], [227,81], [228,79]], 
#poly_rc, region_id = [[232-2,79], [232-2,80], [233-2,80], [233-2,79]], 




line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'

eit_wavelength = 195 #171, 195, 284, or 304 [Angstrom]
eit_time = 'late' #'early' or 'late' (early: around 1 or 4 am; late: around 6 or 7 am)


# Threshold value: label (type) and range of percentageRange percentage of the threshold value
#range_percentage, threshold_value_type, instrument_line = [0., 3.42], 'max', 'eit_195'
#range_percentage, threshold_value_type, instrument_line = [0., 4.], 'max', 'sumer_NeVIII'
#range_percentage, threshold_value_type, instrument_line = [0., 5.], 'max', 'eit_195'
#range_percentage, threshold_value_type, instrument_line = [0., 60.], 'mean', 'eit_195'
#range_percentage, threshold_value_type, instrument_line = [0., 30.], 'max', 'eit_195'
range_percentage, threshold_value_type, instrument_line = [0., 4.], 'max', 'eit_195'



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
show_plots_correction = 'no'

# Full Sun
show_sumer_FOV = 'yes'
show_contours_fullsun = 'yes'
vmin_eit_fullsun, vmax_eit_fullsun = 5.5e1, 5e3 
contour_intensity_eit_fullsun = 135. #110.
#contour_intensity_eit_fullsun = 'upper_bound'

legend_size = 13



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
# Import EIT data

# Select the name of the EIT file according to the above inputs
if eit_time=='early':
    if eit_wavelength==171: filename_eit = 'SOHO_EIT_171_19991107T010032_L1.fits'
    elif eit_wavelength==195: filename_eit = 'SOHO_EIT_195_19991107T042103_L1.fits'
    elif eit_wavelength==284: filename_eit = 'SOHO_EIT_284_19991107T011231_L1.fits'
    elif eit_wavelength==304: filename_eit = 'SOHO_EIT_304_19991107T013601_L1.fits'

elif eit_time=='late':
    if eit_wavelength==171: filename_eit = 'SOHO_EIT_171_19991107T070017_L1.fits'
    elif eit_wavelength==195: filename_eit = 'SOHO_EIT_195_19991107T063706_L1.fits'
    elif eit_wavelength==284: filename_eit = 'SOHO_EIT_284_19991107T070704_L1.fits'
    elif eit_wavelength==304: filename_eit = 'SOHO_EIT_304_19991107T073030_L1.fits'

# Path of EIT file
filepath_eit = path_data_soho + 'eit/' + filename_eit

# Extract data and header
data_eit = fits.getdata(filepath_eit)[::-1]
header_eit = fits.getheader(filepath_eit)

######################################################

from utils.solar_rotation_variables import *
closest_index = closest_index_EIT_SUMER_dic[filename_eit]
closest_time_sumer = closest_time_SUMER_to_EIT_dic[filename_eit]
time_eit = time_EIT_dic[filename_eit]
hour_eit = hour_EIT_dic[filename_eit]
HPlon_rotcomp = HPlon_rotcomp_dic[filename_eit]
HPlon
HPlat
HPlat_croplat = HPlat[slit_top_px:slit_bottom_px+1]

print("Time of EIT image:.......................", time_eit)
print("Closest time of the SUMER raster:........", closest_time_sumer)
print("Index of that SUMER file in the list:....", closest_index)


######################################################
# Crop EIT data

# Find row index in EIT corresponding to these extremes
y_px_crop_top = int(np.round(Y__HP_to_pixel(y_HP=HPlat[slit_top_px], header_eit=header_eit)))
y_px_crop_bottom = int(np.round(Y__HP_to_pixel(y_HP=HPlat[slit_bottom_px], header_eit=header_eit)))
x_px_crop_left = int(np.round(X__HP_to_pixel(x_HP=HPlon_rotcomp[0], header_eit=header_eit)))
x_px_crop_right = int(np.round(X__HP_to_pixel(x_HP=HPlon_rotcomp[-1], header_eit=header_eit)))

# Crop EIT array
data_eit_crop = data_eit[y_px_crop_top:y_px_crop_bottom+1, x_px_crop_left:x_px_crop_right+1]

"""
# Corrected alignment
dx_px = 0
dy_px = -6
data_eit_crop_corrected = data_eit[y_px_crop_top+dy_px : y_px_crop_bottom+dy_px, x_px_crop_left+dx_px : x_px_crop_right+dx_px]

# Slit position
HPlon_slit_rotcomp_corrected = HPlon_rotcomp[closest_index + dx_px]
HPlat_slit_croplat_corrected = HPlat_croplat[[0,-1]]
"""
# Corrected alignment ('NeVIII', 195, late)
dx_px_left = -1
dx_px_right = -1
dy_px_top = -4
dy_px_bottom = -7
vmin_eit, vmax_eit = 5e1, 1e3
vmin_eit, vmax_eit = 3e1, 5e3

data_eit_crop_corrected = data_eit[y_px_crop_top+dy_px_top : y_px_crop_bottom+dy_px_bottom, x_px_crop_left+dx_px_left : x_px_crop_right+dx_px_right]

# Slit position
HPlon_slit_rotcomp_corrected = HPlon_rotcomp[closest_index + dx_px_left]
HPlat_slit_croplat_corrected = HPlat_croplat[[0,-1]]














closest_index = closest_index_EIT_SUMER_dic[filename_eit]
closest_time_sumer = closest_time_SUMER_to_EIT_dic[filename_eit]
time_eit = time_EIT_dic[filename_eit]
hour_eit = hour_EIT_dic[filename_eit]
HPlon_rotcomp = HPlon_rotcomp_dic[filename_eit]
HPlon
HPlat
HPlat_croplat = HPlat[slit_top_px:slit_bottom_px+1]


##############################################################
# Physical units of x and y axis of the spectroheliogram

# Get HP longitude for all the raster
x_HPlon = HPlon #Solar rotation not compensated
x_HPlon_rotcomp = HPlon_rotcomp_dic[filename_eit] #solar rotation compensated (depends on the EIT file chosen)


# Closest SUMER spectrum (index) in time to EIT image
I = closest_index = closest_index_EIT_SUMER_dic[filename_eit]

# Times of EIT and SUMER: format the date in a more readable way
label_date_eit = time_EIT_dic[filename_eit].strftime("%d/%b/%Y %H:%M:%S")

##############################################################

# Extent to convert the axes to helioprojective units (arcseconds)
extent_eit_fullsun_HP = helioprojective_extent_EIT(header_eit=header_eit)

# Projected position and dimension of the slit over the EIT image (at the time of the EIT image)
x_slit = [HPlon_rotcomp[closest_index], HPlon_rotcomp[closest_index]]
y_slit = [HPlat_croplat[0], HPlat_croplat[-1]]

# Projected FOV of the raster over the EIT image with solar rotation compensated and not compensated
x_FOV_rotcomp, y_FOV_rotcomp = create_rectangle(x_left=HPlon_rotcomp[0], x_right=HPlon_rotcomp[-1], y_low=HPlat_croplat[0], y_high=HPlat_croplat[-1], N=100)
#x_FOV_NOrotcomp, y_FOV_NOrotcomp = create_rectangle(x_left=HPlon[0], x_right=HPlon[-1], y_low=HPlat_croplat[0], y_high=HPlat_croplat[-1], N=100)


















######################################################
# Extents

# Extents in pixels
## Image
extent_eit_px_uncorrected_image = [-0.5, data_eit_crop.shape[1]-1+0.5, data_eit_crop.shape[0]-1+0.5, -0.5]
extent_eit_px_image = [-0.5, data_eit_crop_corrected.shape[1]-1+0.5, data_eit_crop_corrected.shape[0]-1+0.5, -0.5]
extent_sumer_px_image = [-0.5, intensity_map_croplat.shape[1]-1+0.5, intensity_map_croplat.shape[0]-1+0.5, -0.5]
## Contours
extent_eit_px_uncorrected_contours = [0., data_eit_crop.shape[1]-1, data_eit_crop.shape[0]-1, 0.]
extent_eit_px_contours = [0., data_eit_crop_corrected.shape[1]-1, data_eit_crop_corrected.shape[0]-1, 0.]
extent_sumer_px_contours = [0., intensity_map_croplat.shape[1]-1, intensity_map_croplat.shape[0]-1, 0.]

vmin_eit, vmax_eit = 4e1, 3e3


# Extents in arcsec
lat_half_bottom = abs((HPlat_croplat[1]-HPlat_croplat[0])/2.)
lat_half_top = abs((HPlat_croplat[-1]-HPlat_croplat[-2])/2.)
lon_half_left = abs((HPlon_rotcomp[1]-HPlon_rotcomp[0])/2.)
lon_half_right = abs((HPlon_rotcomp[-1]-HPlon_rotcomp[-2])/2.)
extent_eit_sumer_arcsec_image = [HPlon_rotcomp[0]-lon_half_left, HPlon_rotcomp[-1]+lon_half_right, HPlat_croplat[-1]-lat_half_bottom, HPlat_croplat[0]+lat_half_top] #arcsec
extent_eit_sumer_arcsec_contours = [HPlon_rotcomp[0], HPlon_rotcomp[-1], HPlat_croplat[-1], HPlat_croplat[0]] #arcsec


######################################################

# Define intensity bin
#lower_bound_eit, upper_bound_eit = get_bounds(intensitymap_croplat=data_eit_crop_corrected, range_percentage=range_percentage, threshold_value_type=threshold_value_type)
#print('lower_bound_eit, upper_bound_eit =', lower_bound_eit, ',', upper_bound_eit)
lower_bound_eit, upper_bound_eit = get_bounds(intensitymap_croplat=data_eit_crop, range_percentage=range_percentage, threshold_value_type=threshold_value_type)
print('lower_bound_eit, upper_bound_eit =', lower_bound_eit, ',', upper_bound_eit)

##############################################################
##############################################################
##############################################################
# Dopplermap and BR asymmetry map

# Load the intensity map and uncertainties
dopplermap_BRmap_loaded_dic = np.load('../outputs/dopplermap_BRmap.npz')
ddopplershift_map_binned_HRTSsub_lessmedian = dopplermap_BRmap_loaded_dic['ddopplershift_map_binned_HRTSsub_lessmedian']
BR_asymmetry_map_gaussian_binned_corrected_normalized = dopplermap_BRmap_loaded_dic['BR_map']


######################################################
# Add pixels addresses from the Ne VIII intensity map to the Dopplermap 

Z = intensity_map_croplat # Z is your 2D array


### Plot intensity map with contours, coordinates, and pixel scale
xlon1 = 0.
xlon2 = Z.shape[1]-1.
ylon1 = HPlon_rotcomp[0]
ylon2 = HPlon_rotcomp[-1]
mlon = (ylon2-ylon1)/(xlon2-xlon1)
blon = HPlon_rotcomp[0]
def pixels_to_HPlon(x): return mlon * x + blon #x=np.arange(0, Z.shape[1])
def HPlon_to_pixels(x): return (x-blon)/mlon #HPlon_rotcomp



xlat1 = 0
xlat2 = Z.shape[0]-1
ylat1 = HPlat_croplat[0]
ylat2 = HPlat_croplat[-1]
mlat = (ylat2-ylat1)/(xlat2-xlat1)
blat = HPlat_croplat[0]
def pixels_to_HPlat(x): return mlat * x + blat #x=np.arange(0, Z.shape[0])
def HPlat_to_pixels(x): return (x-blat)/mlat #HPlat_croplat



"""
### Dopplermap with contours of EIT and secondary axis with pixel addresses from intensity_map_croplat
vmin_vmax = [-12., 12.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
label_size = 18
img = ax.imshow(ddopplershift_map_binned_HRTSsub_lessmedian, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap='seismic', extent=extent_eit_sumer_arcsec_image)
#cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
#cbar = fig.colorbar(img, cax=cax)
#cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
ax.set_title(r'Doppler map, blends corrected', fontsize=20)
#ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
#plt.subplots_adjust(left=0.1, right=0.90, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_lower = ax.contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax.contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='black', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
# Secondary x-axis: arcsec -> pixels
secax_x = ax.secondary_xaxis(
    'top',
    functions=(HPlon_to_pixels, pixels_to_HPlon)
)
secax_x.set_xlabel('Pixel x', fontsize=16)

# Secondary y-axis: arcsec -> pixels
secax_y = ax.secondary_yaxis(
    'right',
    functions=(HPlat_to_pixels, pixels_to_HPlat)
)
secax_y.set_ylabel('Pixel y', fontsize=16)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'dopplermap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)
"""

ei = extent_eit_sumer_arcsec_image
extent_eit_sumer_pxNeVIII_image = [HPlon_to_pixels(x=ei[0]), HPlon_to_pixels(x=ei[1]), pixels_to_HPlat(x=ei[2]), pixels_to_HPlat(x=ei[3])]
ec = extent_eit_sumer_arcsec_contours
extent_eit_sumer_pxNeVIII_contours = [HPlon_to_pixels(x=ec[0]), HPlon_to_pixels(x=ec[1]), pixels_to_HPlat(x=ec[2]), pixels_to_HPlat(x=ec[3])]


### Dopplermap with contours of EIT and secondary axis with pixel addresses from intensity_map_croplat
vmin_vmax = [-12., 12.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
label_size = 18
img = ax.imshow(ddopplershift_map_binned_HRTSsub_lessmedian, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap='seismic', extent=extent_eit_sumer_pxNeVIII_image)
#cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
#cbar = fig.colorbar(img, cax=cax)
#cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
ax.set_title(r'Doppler map, blends corrected', fontsize=20)
#ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
#plt.subplots_adjust(left=0.1, right=0.90, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_lower = ax.contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_pxNeVIII_contours)
contour_upper = ax.contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='black', linewidths=2, extent=extent_eit_sumer_pxNeVIII_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
# Secondary x-axis: arcsec -> pixels
secax_x = ax.secondary_xaxis(
    'top',
    functions=(pixels_to_HPlon, HPlon_to_pixels)
)

# Secondary y-axis: arcsec -> pixels
secax_y = ax.secondary_yaxis(
    'right',
    functions=(pixels_to_HPlat, HPlat_to_pixels)
)
secax_x.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
secax_y.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
ax.set_xlabel('Pixel x', fontsize=16)
ax.set_ylabel('Pixel y', fontsize=16)
#if save_paper_images == 'yes':
#	fig_name = 'polygon'+region_id+'__dopplermap_NeVIII'
#	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



##############################################################
##############################################################
##############################################################


######################################################
# Create a polygon that enclose the left region of the CH and mark all pixels inside the polygon and the CH





fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
ax[0].imshow(data_eit_crop_corrected, norm=LogNorm(vmin=vmin_eit, vmax=vmax_eit), cmap='Greys_r', extent=extent_eit_sumer_arcsec_image)
ax[1].pcolormesh(HPlon_rotcomp, HPlat_croplat, intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
ax[1].axis('equal') # Ensures equal scaling of axis x and y
#ax[1].set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax[1].set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax[0].set_title('Contours of the CH in EIT overlaid in the Ne VIII intensity map', fontsize=18)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=16)
ax[0].text(1.02, 0.5, f'EIT-{header_eit["WAVELNTH"]}', fontsize=20,transform=ax[0].transAxes, va='center', ha='left', rotation=90)
ax[1].text(1.02, 0.5, f'SUMER-{line_center_label}', fontsize=20,transform=ax[1].transAxes, va='center', ha='left', rotation=90)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.95, wspace=0, hspace=0)
#ax[0].grid(color='white')
#ax[1].grid(color='white')
#ax[0].axvline(x=HPlon_rotcomp[closest_index], linestyle='-', linewidth=0.8, color='red', label='Slit position during\n EIT image')
#ax[1].axvline(x=HPlon_rotcomp[closest_index], linestyle='-', linewidth=0.8, color='red', label='Slit position during\n EIT image')
contour_lower = ax[0].contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax[0].contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='yellow', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='yellow', label=f'{range_percentage[1]} %')]
contour_lower = ax[1].contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax[1].contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='yellow', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
#ax[0].axvline(x=HPlon_slit_rotcomp_corrected, linewidth=1.5, color='red', label='slit position')
ax[0].set_aspect('auto')
ax[1].set_aspect('auto')
secax_x = ax[1].secondary_xaxis('top', functions=(HPlon_to_pixels, pixels_to_HPlon))
secax_x.set_xlabel('Pixel x')
secax_y = ax[1].secondary_yaxis('right', functions=(HPlat_to_pixels, pixels_to_HPlat))
secax_y.set_ylabel('Pixel y')
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__intensity_maps_SUMER_EIT_and_contours'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)


######################################################
### Create polygon and plot it

import numpy as np
from matplotlib.path import Path
from matplotlib.patches import Polygon
from skimage.measure import find_contours


# polygon vertices are given as (row, col)
poly_rc = np.array(poly_rc)
poly_xy = poly_rc[:, ::-1]   # convert (row, col) -> (x, y) = (col, row)

nrows_polygon, ncols_polygon = Z.shape
rr_polygon, cc_polygon = np.indices((nrows_polygon, ncols_polygon))

poly_path = Path(poly_rc[:, ::-1]) # If your polygon vertices are (row, col), Path expects (x, y), so pass them as (col, row)
inside_poly = poly_path.contains_points(np.c_[cc_polygon.ravel(), rr_polygon.ravel()]).reshape(Z.shape) # Build a mask of pixels inside the polygon
mask = inside_poly
rowscols_croplat = np.argwhere(mask) # (row, col) of all matching pixels
y_row_list_plot = rowscols_croplat[:,0] # convert the list of pairs [row, column] into 2 lists of rows and columns (for the scatterplot)
x_col_list_plot = rowscols_croplat[:,1]
print('Number of pixels in EIT:', len(rowscols_croplat))

print('##################################')
print('N PIXELS IN POLYGON (QS):', len(rowscols_croplat), '=', len(rowscols_croplat)*150./3600., 'hrs')
print('##################################')


contours_region = find_contours(mask.astype(float), 0.5)


"""
### Dopplermap with contours of EIT and secondary axis with pixel addresses from intensity_map_croplat
vmin_vmax_BR = [-1.,1.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
label_size = 18
img = ax.imshow(BR_asymmetry_map_gaussian_binned_corrected_normalized, vmin=vmin_vmax_BR[0], vmax=vmin_vmax_BR[1], cmap='seismic', extent=extent_eit_sumer_pxNeVIII_image)
#cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
#cbar = fig.colorbar(img, cax=cax)
#cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
ax.set_title(r'Doppler map, blends corrected', fontsize=20)
#ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
#plt.subplots_adjust(left=0.1, right=0.90, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_lower = ax.contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_pxNeVIII_contours)
contour_upper = ax.contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='black', linewidths=2, extent=extent_eit_sumer_pxNeVIII_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
# Secondary x-axis: arcsec -> pixels
secax_x = ax.secondary_xaxis(
    'top',
    functions=(pixels_to_HPlon, HPlon_to_pixels)
)

# Secondary y-axis: arcsec -> pixels
secax_y = ax.secondary_yaxis(
    'right',
    functions=(pixels_to_HPlat, HPlat_to_pixels)
)
secax_x.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
secax_y.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
ax.set_xlabel('Pixel x', fontsize=16)
ax.set_ylabel('Pixel y', fontsize=16)
contours = find_contours(mask.astype(float), 0.5)
for c in contours_region:
    ax.plot(c[:, 1], c[:, 0], color='cyan', linewidth=2)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__dopplermap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)




### BR asymmetry map with contours of EIT and secondary axis with pixel addresses from intensity_map_croplat
vmin_vmax = [-12., 12.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
label_size = 18
img = ax.imshow(ddopplershift_map_binned_HRTSsub_lessmedian, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap='seismic', extent=extent_eit_sumer_pxNeVIII_image)
#cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
#cbar = fig.colorbar(img, cax=cax)
#cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
ax.set_title(r'Doppler map, blends corrected', fontsize=20)
#ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
#plt.subplots_adjust(left=0.1, right=0.90, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_lower = ax.contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_pxNeVIII_contours)
contour_upper = ax.contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='black', linewidths=2, extent=extent_eit_sumer_pxNeVIII_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
# Secondary x-axis: arcsec -> pixels
secax_x = ax.secondary_xaxis(
    'top',
    functions=(pixels_to_HPlon, HPlon_to_pixels)
)

# Secondary y-axis: arcsec -> pixels
secax_y = ax.secondary_yaxis(
    'right',
    functions=(pixels_to_HPlat, HPlat_to_pixels)
)
secax_x.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
secax_y.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
ax.set_xlabel('Pixel x', fontsize=16)
ax.set_ylabel('Pixel y', fontsize=16)
contours = find_contours(mask.astype(float), 0.5)
for c in contours_region:
    ax.plot(c[:, 1], c[:, 0], color='cyan', linewidth=2)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__dopplermap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)
"""



# EIT and SUMER 
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,4))
ax.imshow(intensity_map_croplat, norm=LogNorm(), cmap='Greys_r')
ax.axis('auto') # Ensures equal scaling of axis x and y
ax.grid(color='white')
ax.scatter(x_col_list_plot, y_row_list_plot, marker='.', s=3, color='cyan')
for c in contours_region:
    ax.plot(c[:, 1], c[:, 0], color='red', linewidth=1)
plt.show(block=False)






from skimage.measure import find_contours
from matplotlib.path import Path
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# polygon vertices in pixel indices
poly_path = Path(poly_rc[:, ::-1])

fig, ax = plt.subplots(figsize=(9, 4))
HPlon_rotcomp = np.asarray(HPlon_rotcomp)
HPlat_croplat = np.asarray(HPlat_croplat)
extent = [HPlon_rotcomp.min(), HPlon_rotcomp.max(), HPlat_croplat.min(), HPlat_croplat.max()]
ax.imshow(
    data_eit_crop_corrected,
    norm=LogNorm(),
    cmap='Greys_r',
    extent=extent,
    origin='upper',
    aspect='equal'
)
# polygon border in coordinate space
#poly_x = HPlon_rotcomp[poly_rc[:, 1]]  # col -> x
#poly_y = HPlat_croplat[poly_rc[:, 0]]  # row -> y
#ax.add_patch(Polygon(np.c_[poly_x, poly_y], closed=True,fill=False, edgecolor='lime', linewidth=2))
# border of selected region
contours = find_contours(mask.astype(float), 0.5)
for c in contours:
    rr_c = c[:, 0]
    cc_c = c[:, 1]
    x = np.interp(cc_c, np.arange(len(HPlon_rotcomp)), HPlon_rotcomp)
    y = np.interp(rr_c, np.arange(len(HPlat_croplat)), HPlat_croplat)
    ax.plot(x, y, color='red', linewidth=1.5)
ax.set_xlabel('Helioprojective longitude (arcsec)')
ax.set_ylabel('Helioprojective latitude (arcsec)')
ax.grid(color='white', alpha=0.3)
plt.show(block=False)



######################################################

# Import SUMER data interpolated (wavelength calibrated)
data_interpolated_loaded = np.load('../data/data_modified/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)
# Average spectra of the pixels selected
lam_sumer_av, elam_sumer_av, rad_sumer_av, erad_sumer_av = average_profiles_from_pixels_selected_from_interpolated_data(wavelength_range_=wavelength_range_to_average, data_interpolated_loaded_=data_interpolated_loaded, rows_cols_of_spectroheliogram_croplat=rowscols_croplat)

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


if show_secondary_plots == 'yes':
    # EIT and SUMER with the grid (for alignment)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9,8))
    ax[0].imshow(data_eit_crop_corrected, norm=LogNorm(vmin=vmin_eit, vmax=vmax_eit), cmap='Greys_r', extent=extent_eit_sumer_arcsec_image)
    ax[1].pcolormesh(HPlon_rotcomp, HPlat_croplat, intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
    ax[1].axis('equal') # Ensures equal scaling of axis x and y
    #ax[1].set_title(f'SUMER', fontsize=22)
    #ax[0].set_title(f'EIT-{header_eit["WAVELNTH"]}', fontsize=22)
    #ax[1].set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
    ax[1].set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
    fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
    ax[0].text(1.02, 0.5, f'EIT-{header_eit["WAVELNTH"]}', fontsize=22,transform=ax[0].transAxes, va='center', ha='left', rotation=90)
    ax[1].text(1.02, 0.5, f'SUMER-{line_center_label}', fontsize=22,transform=ax[1].transAxes, va='center', ha='left', rotation=90)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)
    ax[0].grid(color='white')
    ax[1].grid(color='white')
    ax[0].axvline(x=HPlon_rotcomp[closest_index], linestyle='-', linewidth=0.8, color='red', label='Slit position during\n EIT image')
    ax[1].axvline(x=HPlon_rotcomp[closest_index], linestyle='-', linewidth=0.8, color='red')
    #ax[0].set_xlim([HPlon_rotcomp[0], HPlon_rotcomp[-1]])
    #ax[1].set_xlim([HPlon_rotcomp[0], HPlon_rotcomp[-1]])
    plt.show(block=False)



    # EIT and SUMER with the individual pixels marked
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,14))
    ax[0].imshow(data_eit_crop_corrected, norm=LogNorm(vmin=vmin_eit, vmax=vmax_eit), cmap='Greys_r', aspect='auto', extent=extent_eit_px_image)
    ax[1].imshow(intensity_map_croplat, norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer), cmap='Greys_r', aspect='auto', extent=extent_sumer_px_image)
    ax[1].set_aspect('auto')
    ax[1].set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    ax[0].text(1.02, 0.5, f'EIT-{header_eit["WAVELNTH"]}', fontsize=22,transform=ax[0].transAxes, va='center', ha='left', rotation=90)
    ax[1].text(1.02, 0.5, f'SUMER-{line_center_label}', fontsize=22,transform=ax[1].transAxes, va='center', ha='left', rotation=90)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    # EIT subplot (top) - contours from EIT data
    contour_lower_eit = ax[0].contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_px_contours)
    contour_upper_eit = ax[0].contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='blue', linewidths=2, extent=extent_eit_px_contours)
    # EIT subplot (bottom) - contours from EIT data
    contour_lower_eit = ax[1].contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_sumer_px_contours)
    contour_upper_eit = ax[1].contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='blue', linewidths=2, extent=extent_sumer_px_contours)
    # Plot scatter AFTER contours with zorder to ensure visibility
    ax[0].scatter(x_col_list_eit_plot, y_row_list_eit_plot, color='cyan', marker='s', s=1, zorder=10)
    ax[1].scatter(x_col_list_sumer_plot, y_row_list_sumer_plot, color='cyan', marker='o', s=0.7, zorder=10)
    plt.show(block=False)




### PAPER image: Contours EIT and region
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
ax[0].imshow(data_eit_crop_corrected, norm=LogNorm(vmin=vmin_eit, vmax=vmax_eit), cmap='Greys_r', extent=extent_eit_sumer_arcsec_image)
ax[1].pcolormesh(HPlon_rotcomp, HPlat_croplat, intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
ax[1].axis('equal') # Ensures equal scaling of axis x and y
#ax[1].set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax[1].set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax[0].set_title('Contours of the CH in EIT overlaid in the Ne VIII intensity map', fontsize=18)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=16)
ax[0].text(1.02, 0.5, f'EIT-{header_eit["WAVELNTH"]}', fontsize=20,transform=ax[0].transAxes, va='center', ha='left', rotation=90)
ax[1].text(1.02, 0.5, f'SUMER-{line_center_label}', fontsize=20,transform=ax[1].transAxes, va='center', ha='left', rotation=90)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.95, wspace=0, hspace=0)
#ax[0].grid(color='white')
#ax[1].grid(color='white')
#ax[0].axvline(x=HPlon_rotcomp[closest_index], linestyle='-', linewidth=0.8, color='red', label='Slit position during\n EIT image')
#ax[1].axvline(x=HPlon_rotcomp[closest_index], linestyle='-', linewidth=0.8, color='red', label='Slit position during\n EIT image')
contour_lower = ax[0].contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax[0].contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='yellow', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='yellow', label=f'{range_percentage[1]} %')]
contour_lower = ax[1].contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax[1].contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='yellow', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
#ax[0].axvline(x=HPlon_slit_rotcomp_corrected, linewidth=1.5, color='red', label='slit position')
ax[0].set_aspect('auto')
ax[1].set_aspect('auto')
contours = find_contours(mask.astype(float), 0.5)
for c in contours:
    rr_c = c[:, 0]
    cc_c = c[:, 1]
    x = np.interp(cc_c, np.arange(len(HPlon_rotcomp)), HPlon_rotcomp)
    y = np.interp(rr_c, np.arange(len(HPlat_croplat)), HPlat_croplat)
    ax[0].plot(x, y, color='red', linewidth=1.5)
    ax[1].plot(x, y, color='red', linewidth=1.5)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__intensity_maps_SUMER_EIT_and_contours'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



### PAPER image: Contours only region
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
ax[0].imshow(data_eit_crop_corrected, norm=LogNorm(vmin=vmin_eit, vmax=vmax_eit), cmap='Greys_r', extent=extent_eit_sumer_arcsec_image)
ax[1].pcolormesh(HPlon_rotcomp, HPlat_croplat, intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
ax[1].axis('equal') # Ensures equal scaling of axis x and y
#ax[1].set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax[1].set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax[0].set_title('Contours of the CH in EIT overlaid in the Ne VIII intensity map', fontsize=18)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=16)
ax[0].text(1.02, 0.5, f'EIT-{header_eit["WAVELNTH"]}', fontsize=20,transform=ax[0].transAxes, va='center', ha='left', rotation=90)
ax[1].text(1.02, 0.5, f'SUMER-{line_center_label}', fontsize=20,transform=ax[1].transAxes, va='center', ha='left', rotation=90)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.95, wspace=0, hspace=0)
#ax[0].grid(color='white')
#ax[1].grid(color='white')
#ax[0].axvline(x=HPlon_rotcomp[closest_index], linestyle='-', linewidth=0.8, color='red', label='Slit position during\n EIT image')
#ax[1].axvline(x=HPlon_rotcomp[closest_index], linestyle='-', linewidth=0.8, color='red', label='Slit position during\n EIT image')
#contour_lower = ax[0].contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
#contour_upper = ax[0].contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='yellow', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
#legend_elements = [
#    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
#    mlines.Line2D([],[],color='yellow', label=f'{range_percentage[1]} %')]
#contour_lower = ax[1].contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
#contour_upper = ax[1].contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='yellow', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
#ax[0].axvline(x=HPlon_slit_rotcomp_corrected, linewidth=1.5, color='red', label='slit position')
ax[0].set_aspect('auto')
ax[1].set_aspect('auto')
contours = find_contours(mask.astype(float), 0.5)
for c in contours:
    rr_c = c[:, 0]
    cc_c = c[:, 1]
    x = np.interp(cc_c, np.arange(len(HPlon_rotcomp)), HPlon_rotcomp)
    y = np.interp(rr_c, np.arange(len(HPlat_croplat)), HPlat_croplat)
    ax[0].plot(x, y, color='red', linewidth=1.5)
    ax[1].plot(x, y, color='red', linewidth=1.5)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__intensity_maps_SUMER_EIT_and_contours'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)





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
    filename_profile = 'average_profile__' + range_numbers_to_string + '__' + threshold_value_type + '_of_eit_' + str(eit_wavelength)
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
# Plot full Sun (EIT) with the FOV of the raster and the slit position during the image

if contour_intensity_eit_fullsun == 'upper_bound': contour_intensity_eit_fullsun = upper_bound_eit

##############################################################
# Physical units of x and y axis of the spectroheliogram

# Get HP latitude for the entire y axis (360 pixels), and for the cropped one
y_HPlat_fullLat = HPlat
y_HPlat_crop = HPlat[slit_top_px:slit_bottom_px+1]

# Closest SUMER spectrum (index) in time to EIT image
I = closest_index = closest_index_EIT_SUMER_dic[filename_eit]

# Times of EIT and SUMER: format the date in a more readable way
label_date_eit = time_EIT_dic[filename_eit].strftime("%d/%b/%Y %H:%M:%S")

##############################################################

# Extent to convert the axes to helioprojective units (arcseconds)
extent_eit_fullsun_HP = helioprojective_extent_EIT(header_eit=header_eit)

# Projected position and dimension of the slit over the EIT image (at the time of the EIT image)
x_slit = [HPlon_rotcomp[closest_index], HPlon_rotcomp[closest_index]]
y_slit = [y_HPlat_crop[0], y_HPlat_crop[-1]]

# Projected FOV of the raster over the EIT image with solar rotation compensated and not compensated
x_FOV_rotcomp, y_FOV_rotcomp = create_rectangle(x_left=HPlon_rotcomp[0], x_right=HPlon_rotcomp[-1], y_low=y_HPlat_crop[0], y_high=y_HPlat_crop[-1], N=100)
x_FOV_NOrotcomp, y_FOV_NOrotcomp = create_rectangle(x_left=HPlon[0], x_right=HPlon[-1], y_low=y_HPlat_crop[0], y_high=y_HPlat_crop[-1], N=100)





########## Full Sun 1: Show all contours INSIDE the solar disk

"""
### PAPER image: full Sun with contours and SUMER FOV
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 11))
img = ax.imshow(data_eit, cmap='Greys_r', norm=LogNorm(vmin=vmin_eit_fullsun, vmax=vmax_eit_fullsun), extent=extent_eit_fullsun_HP) # Plot the main solar image 
#cax = fig.add_axes([0.88, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
cax = fig.add_axes([0.92, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, ax=ax, cax=cax, pad=0.01)
cbar.set_label(f'Intensity (DN/s)', fontsize=16)
ax.set_title(f'SOHO/EIT - {header_eit["WAVELNTH"]} \u212B, {label_date_eit}', fontsize=20)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
if show_contours_fullsun=='yes':
    from skimage import measure
    contours = measure.find_contours(data_eit[::-1], level=contour_intensity_eit_fullsun) # Find contours at a given level
    solar_center = (0,0)  # Helioprojective (x, y) center of the Sun
    solar_radius = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius   
    largest_contour = None
    max_points = 0
    for contour in contours:
        # Convert image coordinates to helioprojective coordinates
        x_contour = np.interp(contour[:, 1], [0, data_eit.shape[1]], [extent_eit_fullsun_HP[0], extent_eit_fullsun_HP[1]])
        y_contour = np.interp(contour[:, 0], [0, data_eit.shape[0]], [extent_eit_fullsun_HP[2], extent_eit_fullsun_HP[3]])
        # Filter out contour points outside the solar disk
        distances = np.sqrt((x_contour - solar_center[0])**2 + (y_contour - solar_center[1])**2)
        inside_mask = distances < solar_radius
        # Only plot if contour has at least some points inside solar disk
        if np.any(inside_mask):
            x_inside = x_contour[inside_mask]
            y_inside = y_contour[inside_mask]
            ax.plot(x_inside, y_inside, color='yellow', linewidth=1.5)
    ax.plot([],[], color='yellow', linewidth=1.5, label=f'Contours {contour_intensity_eit_fullsun} DN/s')
ax.set_xlim([-1100,1100])
ax.set_ylim([-1100,1100])
ax.set_aspect('equal', adjustable='box')
if show_sumer_FOV=='yes':
    #ax.plot(x_FOV_rotcomp, y_FOV_rotcomp, linestyle='-', linewidth=1.1, color='cyan', label='Raster FOV, SR compensated')
    ax.plot(x_FOV_rotcomp, y_FOV_rotcomp, linestyle='-', linewidth=1.5, color='cyan', label='Raster FOV')
#ax.axis('equal')
ax.legend(fontsize=12)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__full_sun_and_SUMER_FOV'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



### PAPER image: full Sun with contours, SUMER FOV, and slit position
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 11))
img = ax.imshow(data_eit, cmap='Greys_r', norm=LogNorm(vmin=vmin_eit_fullsun, vmax=vmax_eit_fullsun), extent=extent_eit_fullsun_HP) # Plot the main solar image 
cax = fig.add_axes([0.92, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, ax=ax, cax=cax, pad=0.01)
cbar.set_label(f'Intensity (DN/s)', fontsize=16)
ax.set_title(f'SOHO/EIT - {header_eit["WAVELNTH"]} \u212B, {label_date_eit}', fontsize=20)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
if show_contours_fullsun=='yes':
    from skimage import measure
    contours = measure.find_contours(data_eit[::-1], level=contour_intensity_eit_fullsun) # Find contours at a given level
    solar_center = (0,0)  # Helioprojective (x, y) center of the Sun
    solar_radius = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius   
    largest_contour = None
    max_points = 0
    for contour in contours:
        # Convert image coordinates to helioprojective coordinates
        x_contour = np.interp(contour[:, 1], [0, data_eit.shape[1]], [extent_eit_fullsun_HP[0], extent_eit_fullsun_HP[1]])
        y_contour = np.interp(contour[:, 0], [0, data_eit.shape[0]], [extent_eit_fullsun_HP[2], extent_eit_fullsun_HP[3]])
        # Filter out contour points outside the solar disk
        distances = np.sqrt((x_contour - solar_center[0])**2 + (y_contour - solar_center[1])**2)
        inside_mask = distances < solar_radius
        # Only plot if contour has at least some points inside solar disk
        if np.any(inside_mask):
            x_inside = x_contour[inside_mask]
            y_inside = y_contour[inside_mask]
            ax.plot(x_inside, y_inside, color='yellow', linewidth=1.5)
    ax.plot([],[], color='yellow', linewidth=1.5, label=f'Contours {contour_intensity_eit_fullsun} DN/s')
ax.plot([HPlon_slit_rotcomp_corrected, HPlon_slit_rotcomp_corrected], HPlat_slit_croplat_corrected, linewidth=1.5, color='red', label='slit position')   
ax.set_xlim([-1100,1100])
ax.set_ylim([-1100,1100])
ax.set_aspect('equal', adjustable='box')
if show_sumer_FOV=='yes':
    #ax.plot(x_FOV_rotcomp, y_FOV_rotcomp, linestyle='-', linewidth=1.1, color='cyan', label='Raster FOV, SR compensated')
    ax.plot(x_FOV_rotcomp, y_FOV_rotcomp, linestyle='-', linewidth=1.5, color='cyan', label='Raster FOV')
#ax.axis('equal')
ax.legend(fontsize=12)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__full_sun_and_SUMER_FOV_and_slit'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



########## Full Sun 2: Plot only contours inside the solar disk, and also not small patches
# Isolate only the contours of the largest connected region and exclude small patches. We can filter the contours based on their area (or number of points). 

### PAPER image: full Sun with contours and SUMER FOV
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 11))
img = ax.imshow(data_eit, cmap='Greys_r', norm=LogNorm(vmin=vmin_eit_fullsun, vmax=vmax_eit_fullsun), extent=extent_eit_fullsun_HP) # Plot the main solar image 
cax = fig.add_axes([0.92, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, ax=ax, cax=cax, pad=0.01)
cbar.set_label(f'Intensity (DN/s)', fontsize=16)
ax.set_title(f'SOHO/EIT - {header_eit["WAVELNTH"]} \u212B, {label_date_eit}', fontsize=20)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
if show_sumer_FOV=='yes':
    #ax.plot(x_FOV_rotcomp, y_FOV_rotcomp, linestyle='-', linewidth=1.1, color='cyan', label='Raster FOV, SR compensated')
    ax.plot(x_FOV_rotcomp, y_FOV_rotcomp, linestyle='-', linewidth=1.5, color='cyan', label='Raster FOV')
if show_contours_fullsun=='yes':
    from skimage import measure
    contours = measure.find_contours(data_eit[::-1], level=contour_intensity_eit_fullsun) # Find contours at a given level
    solar_center = (0,0)  # Helioprojective (x, y) center of the Sun
    solar_radius = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius   
    largest_contour = None
    max_points = 0
    for contour in contours:
        # Convert image coordinates to helioprojective coordinates
        x_contour = np.interp(contour[:, 1], [0, data_eit.shape[1]], [extent_eit_fullsun_HP[0], extent_eit_fullsun_HP[1]])
        y_contour = np.interp(contour[:, 0], [0, data_eit.shape[0]], [extent_eit_fullsun_HP[2], extent_eit_fullsun_HP[3]])
        # Filter out contour points outside the solar disk
        distances = np.sqrt((x_contour - solar_center[0])**2 + (y_contour - solar_center[1])**2)
        inside_mask = distances < solar_radius
        x_inside = x_contour[inside_mask]
        y_inside = y_contour[inside_mask]
        # Keep only the largest contour (or contours above a size threshold)
        if len(x_inside) > max_points:
            largest_contour = (x_inside, y_inside)
            max_points = len(x_inside)
    # Plot the largest contour only
    if largest_contour is not None:
        ax.plot(largest_contour[0], largest_contour[1], color='yellow', linewidth=1.5)
    ax.plot([],[], color='yellow', linewidth=1.5, label=f'Contours {contour_intensity_eit_fullsun} DN/s')
ax.set_xlim([-1100,1100])
ax.set_ylim([-1100,1100])
ax.set_aspect('equal', adjustable='box')
#ax.axis('equal')
ax.legend(fontsize=12)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__full_sun_and_SUMER_FOV_bigcontour'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



### PAPER image: full Sun with contours, SUMER FOV, and slit position
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11,11))
img = ax.imshow(data_eit, cmap='Greys_r', norm=LogNorm(vmin=vmin_eit_fullsun, vmax=vmax_eit_fullsun), extent=extent_eit_fullsun_HP) # Plot the main solar image 
#cax = fig.add_axes([0.88, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
cax = fig.add_axes([0.92, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, ax=ax, cax=cax, pad=0.01)
cbar.set_label(f'Intensity (DN/s)', fontsize=16)
ax.set_title(f'SOHO/EIT - {header_eit["WAVELNTH"]} \u212B, {label_date_eit}', fontsize=20)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
if show_sumer_FOV=='yes':
    #ax.plot(x_FOV_rotcomp, y_FOV_rotcomp, linestyle='-', linewidth=1.1, color='cyan', label='Raster FOV, SR compensated')
    ax.plot(x_FOV_rotcomp, y_FOV_rotcomp, linestyle='-', linewidth=1.5, color='cyan', label='Raster FOV')
if show_contours_fullsun=='yes':
    from skimage import measure
    contours = measure.find_contours(data_eit[::-1], level=contour_intensity_eit_fullsun) # Find contours at a given level
    solar_center = (0,0)  # Helioprojective (x, y) center of the Sun
    solar_radius = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius   
    largest_contour = None
    max_points = 0
    for contour in contours:
        # Convert image coordinates to helioprojective coordinates
        x_contour = np.interp(contour[:, 1], [0, data_eit.shape[1]], [extent_eit_fullsun_HP[0], extent_eit_fullsun_HP[1]])
        y_contour = np.interp(contour[:, 0], [0, data_eit.shape[0]], [extent_eit_fullsun_HP[2], extent_eit_fullsun_HP[3]])
        # Filter out contour points outside the solar disk
        distances = np.sqrt((x_contour - solar_center[0])**2 + (y_contour - solar_center[1])**2)
        inside_mask = distances < solar_radius
        x_inside = x_contour[inside_mask]
        y_inside = y_contour[inside_mask]
        # Keep only the largest contour (or contours above a size threshold)
        if len(x_inside) > max_points:
            largest_contour = (x_inside, y_inside)
            max_points = len(x_inside)
    # Plot the largest contour only
    if largest_contour is not None:
        ax.plot(largest_contour[0], largest_contour[1], color='yellow', linewidth=1.5)
    ax.plot([],[], color='yellow', linewidth=1.5, label=f'Contours {contour_intensity_eit_fullsun} DN/s')
ax.plot([HPlon_slit_rotcomp_corrected, HPlon_slit_rotcomp_corrected], HPlat_slit_croplat_corrected, linewidth=1.5, color='red', label='slit position')
ax.set_xlim([-1100,1100])
ax.set_ylim([-1100,1100])
ax.set_aspect('equal', adjustable='box')
#ax.axis('equal')
ax.legend(fontsize=12)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__full_sun_and_SUMER_FOV_bigcontour_and_slit'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)
"""

##############################################################
##############################################################
##############################################################
# Dopplermap and BR asymmetry map


"""
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
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



### PAPER image: Dopplermap with contours of EIT
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
contour_lower = ax.contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax.contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='black', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__dopplermap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
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
#plt.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_lower = ax.contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax.contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='black', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
if save_paper_images == 'yes':
	fig_name = 'asymmetrymap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



### PAPER image: B-R asymmetry map with contours of EIT
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
contour_lower = ax.contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax.contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='black', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__asymmetrymap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)
"""

############################################################################################################
############################################################################################################
############################################################################################################
# Profile of SUMER uncorrected and corrected (using the 3 spectra of HRTS)

range_numbers_to_string = '__'.join(f"{x:.2f}".replace('.', '_').rstrip('0') if f"{x:.2f}"[-1] != '0' else f"{x:.1f}".replace('.', '_') for x in range_percentage) 
filename_averaged_spectrum = 'average_profile__' + range_numbers_to_string + '__' + threshold_value_type + '_of_'+instrument_line+'.npz'


## Import dictionary
profiles_loaded_dic = np.load('../outputs/'+filename_averaged_spectrum)

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
"""
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
## HRTS scaled and convolved
ax.errorbar(x=v_hrtsa_cropNeVIII, y=rad_hrtsa_conv_scaled_cropNeVIII, linestyle='--', linewidth=1.2, color=color_hrts_qra, label='HRTS QS-A')
ax.errorbar(x=v_hrtsb_cropNeVIII, y=rad_hrtsb_conv_scaled_cropNeVIII, linestyle='--', linewidth=1.2, color=color_hrts_qrb, label='HRTS QS-B')
## SUMER uncorrected
#ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, linestyle='-', marker='.', markersize=10, linewidth=1.5, color=color_sumer_uncorrected, label='SUMER not corrected')
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, linestyle='-', linewidth=1.5, color=color_sumer_uncorrected, label='SUMER not corrected')
## SUMER corrected
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII_corrected_qra, yerr=erad_sumer_cropNeVIII_corrected_qra, linestyle='-', linewidth=1.2, color=color_hrts_qra, label='SUMER corrected, QS-A')
ax.errorbar(x=v_sumer_cropNeVIII, y=rad_sumer_cropNeVIII_corrected_qrb, yerr=erad_sumer_cropNeVIII_corrected_qrb, linestyle='-', linewidth=1.2, color=color_hrts_qrb, label='SUMER corrected, QS-B')
## 
ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, label=rest_wavelength_label_figures)
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_title(f'SUMER spectrum of the CH uncorrected and corrected with HRTS', fontsize=title_size)
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
# legend in desired order:
handles, labels = ax.get_legend_handles_labels()
order = [
    labels.index('SUMER not corrected'),
    labels.index('SUMER corrected, QS-A'),
    labels.index('SUMER corrected, QS-B'),
    labels.index(rest_wavelength_label_figures),]
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=legend_size)
ax.set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)
ax.set_xlim(x_lims)
plt.tight_layout()
plt.show(block=False)
"""

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
#0.5-bckg_fit_corrected_qra, -30., 20.,
3.-bckg_fit_corrected_qra, 0.0, 50.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
#0.5-bckg_fit_corrected_qrb, -30., 40.,
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
#if save_paper_images == 'yes':
#    fig_name = 'polygon'+region_id+'__spectra_sumer_and_hrts_together'
#    plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



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
#ax.set_title(f'QS-1', fontsize=title_size)
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
    'QS-1',    # your text (multiline if you want)
    transform=ax.transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=35,             # “big” text; adjust as needed
    fontweight='bold', color='cyan')#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
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

### PAPER image: same as above but in wavelength instead of velocity
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
#ax.set_title(f'QS-1', fontsize=title_size)
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
    'QS-1',    # your text (multiline if you want)
    transform=ax.transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color='cyan')#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
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
    fig_name = 'polygon'+region_id+'__spectra_sumer_and_hrts_together'
    plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)




y_lims_fit = [-0.02, 1.7]


######################################################
######################################################
######################################################
######################################################
######################################################

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
    fig_name = 'polygon'+region_id+'__spectra_sumer_and_hrts_together_and_line_identification'
    plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)

######################################################
# Fit uncorrected Ne VIII line

x_uncorrected = lam_sumer_cropNeVIII
y_uncorrected = rad_sumer_cropNeVIII
y_unc_uncorrected = erad_sumer_cropNeVIII

"""
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
ax[0].set_title(f'QS-1 uncorrected', fontsize=title_size)
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
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__spectrum_multigaussian_fit_uncorrected'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)
"""


######################################################
# Single fit, QS-A

bckg_fit1_corrected_qra = -0.3
init_parameters1_corrected_qra = [bckg_fit1_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
3.-bckg_fit1_corrected_qra, 0.0, 50.
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
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0) ,y=y_corrected_qra, yerr=y_unc_corrected_qra, color=color_hrts_qra, marker='o', linewidth=0, elinewidth=1., label='SUMER data')
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
ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII - 770.428 \u212B')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-A, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'QS-1 corrected with QS-A', fontsize=title_size)
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
    'QS-1',    # your text (multiline if you want)
    transform=ax[0].transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color='cyan')#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__spectrum_multigaussian_fit_corrected_qra'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)


           
           
######################################################
# Double fit, QS-A

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
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0) ,y=y_corrected_qra, yerr=y_unc_corrected_qra, color=color_hrts_qra, marker='o', linewidth=0, elinewidth=1., label='SUMER data')
ax[0].plot(x_fit_corrected_qra, y_fit_corrected_qra, color=color_hrts_qra, linestyle='-', label='Double Gaussian fit', zorder=1) 
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
ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII - 770.428 \u212B')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-A, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'QS-1 corrected with QS-A', fontsize=title_size)
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
    'QS-1',    # your text (multiline if you want)
    transform=ax[0].transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color='cyan')#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
# Reduced chi-square text on the left
ax[0].text(0.02, 0.98,  # x, y in axes coordinates (0–1); top-left corner
    r'Double Gaussian fit: $\chi^2_{\rm red} = ' + f'{chi2red_qra_double:.1f}' + r'$' + '\n'
    r'Single Gaussian fit: $\chi^2_{\rm red} = ' + f'{chi2red_qra_single:.1f}' + r'$', transform=ax[0].transAxes, ha='left', va='top', fontsize=chi2red_fontsize, fontweight='normal', color='black')
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__spectrum_multigaussian_fit_corrected_qra'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)


######################################################
# Single fit, QS-B

bckg_fit1_corrected_qrb = -0.3
init_parameters1_corrected_qrb = [bckg_fit1_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
3.-bckg_fit1_corrected_qrb, 0.0, 50.
]

x_corrected_qrb = lam_sumer_cropNeVIII
y_corrected_qrb = rad_sumer_cropNeVIII_corrected_qrb
y_unc_corrected_qrb = erad_sumer_cropNeVIII_corrected_qrb


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
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0) ,y=y_corrected_qrb, yerr=y_unc_corrected_qrb, color=color_hrts_qrb, marker='o', linewidth=0, elinewidth=1., label='SUMER data')
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
ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII - 770.428 \u212B')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-B, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'QS-1 corrected with QS-B', fontsize=title_size)
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
    'QS-1',    # your text (multiline if you want)
    transform=ax[0].transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color='cyan')#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__spectrum_multigaussian_fit_corrected_qrb'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)

######################################################
# Double fit, QS-B


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
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0) ,y=y_corrected_qrb, yerr=y_unc_corrected_qrb, color=color_hrts_qrb, marker='o', linewidth=0, elinewidth=1., label='SUMER data')
ax[0].plot(x_fit_corrected_qrb, y_fit_corrected_qrb, color=color_hrts_qrb, linestyle='-', label='Double Gaussian fit', zorder=1) 
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
ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII - 770.428 \u212B')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-B, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'QS-1 corrected with QS-B', fontsize=title_size)
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
    'QS-1',    # your text (multiline if you want)
    transform=ax[0].transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color='cyan')#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
# Reduced chi-square text on the left
ax[0].text(0.02, 0.98,  # x, y in axes coordinates (0–1); top-left corner
    r'Double Gaussian fit: $\chi^2_{\rm red} = ' + f'{chi2red_qrb_double:.1f}' + r'$' + '\n'
    r'Single Gaussian fit: $\chi^2_{\rm red} = ' + f'{chi2red_qrb_single:.1f}' + r'$', transform=ax[0].transAxes, ha='left', va='top', fontsize=chi2red_fontsize, fontweight='normal', color='black')
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__spectrum_multigaussian_fit_corrected_qrb'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)


######################################################
# Fit corrected Ne VIII line, QS-L
"""
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
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected_qrl, lamb_0=lam_0) ,y=y_corrected_qrl, yerr=y_unc_corrected_qrl, color=color_hrts_qrl, marker='o', linewidth=0, elinewidth=1., label='SUMER data')
ax[0].plot(x_fit_corrected_qrl, y_fit_corrected_qrl, color=color_hrts_qrl, linestyle='-', label='Double Gaussian fit', zorder=1) 
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
ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII - 770.428 \u212B')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-L, multigaussian fit', fontsize=title_size)
ax[0].set_title(f'QS-1 corrected with QS-L', fontsize=title_size)
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
# Big text at the right inside the panel
ax[0].text(
    0.8, 0.5,               # x, y in axes coordinates (0–1); >1 is outside data area but still in axes
    'QS-1',    # your text (multiline if you want)
    transform=ax[0].transAxes,  # interpret x,y in axes coordinates
    ha='left', va='top',  # align left, centered vertically
    fontsize=40,             # “big” text; adjust as needed
    fontweight='bold', color='cyan')#, color=color_contours_eit_Imap)#color=(0.83, 0.70, 0.00)
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
if save_paper_images == 'yes':
	fig_name = 'polygon'+region_id+'__spectrum_multigaussian_fit_corrected_qrl'
	plt.savefig(folder_name+'/'+fig_name+'.pdf', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)


#############################################


#xb_uncorrected, yb_uncorrected = find_bisector(x_data=x_uncorrected[3:-7], y_data=y_uncorrected[3:-7], y_unc_data=y_unc_uncorrected[3:-7], y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')

#xb_corrected, yb_corrected = find_bisector(x_data=x_corrected, y_data=y_corrected, y_unc_data=y_unc_corrected, y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')
"""

#############################################

print('chi2red_qra_single: ', chi2red_qra_single)
print('chi2red_qra_double: ', chi2red_qra_double)
print('chi2red_qrb_single: ', chi2red_qrb_single)
print('chi2red_qrb_double: ', chi2red_qrb_double)


