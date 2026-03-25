
#  Inputs
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'

eit_wavelength = 195 #171, 195, 284, or 304 [Angstrom]
eit_time = 'late' #'early' or 'late' (early: around 1 or 4 am; late: around 6 or 7 am)

# Threshold value: label (type) and range of percentageRange percentage of the threshold value
threshold_value_type = 'mean' #'max', 'min', 'mean', 'median'
range_percentage = [0., 3.4208379]
range_percentage = [0., 4.]
range_percentage = [0., 60.]


fwhm_conv = 1.95*0.04215 #Angstrom

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
save_average_profile = 'no' 

show_plots_correction = 'no'

save_average_profile_map = 'yes'


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

# Rest wavelength used
rest_wavelength_label = 'Peter_and_Judge_1999' #'SUMER_atlas', 'Peter_1998', 'Dammasch_1999', 'Peter_and_Judge_1999', 'Kelly_database'
lamb_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][0] #Angstrom
lamb_unc_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][1] #Angstrom
print('Rest wavelength Ne VIII (2nd order):', lamb_0, r'$\pm$', lamb_unc_0, '\u212B')

# uncertainty of the rest wavelength in km/s
v_unc_0 = vkms_doppler_unc(lamb=lamb_0, lamb_unc=lamb_unc_0, lamb_0=lamb_0, lamb_0_unc=lamb_unc_0) 


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

# Corrected alignment
dx_px = 0
dy_px = -6
data_eit_crop_corrected = data_eit[y_px_crop_top+dy_px : y_px_crop_bottom+dy_px, x_px_crop_left+dx_px : x_px_crop_right+dx_px]

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
lower_bound_eit, upper_bound_eit = get_bounds(intensitymap_croplat=data_eit_crop_corrected, range_percentage=range_percentage, threshold_value_type=threshold_value_type)
print('lower_bound_eit, upper_bound_eit =', lower_bound_eit, ',', upper_bound_eit)

# rows and columns inside the intensity bin  in EIT ,map
rowscols_croplat_eit = np.argwhere((data_eit_crop_corrected>=lower_bound_eit) & (data_eit_crop_corrected<=upper_bound_eit))
y_row_list_eit_plot = rowscols_croplat_eit[:,0] # convert the list of pairs [row, column] into 2 lists of rows and columns (for the scatterplot)
x_col_list_eit_plot = rowscols_croplat_eit[:,1]
print('Number of pixels detected:', len(rowscols_croplat_eit))

# rows and columns from EIT in SUMER
rowscols_croplat_sumer_from_eit = map_pixels_array1_to_array2(arr1=data_eit_crop_corrected, arr2=intensity_map_croplat, pixel_address_list_1=rowscols_croplat_eit)
rowscols_croplat_sumer_from_eit = np.array(rowscols_croplat_sumer_from_eit)
y_row_list_sumer_plot = rowscols_croplat_sumer_from_eit[:,0] # convert the list of pairs [row, column] into 2 lists of rows and columns (for the scatterplot)
x_col_list_sumer_plot = rowscols_croplat_sumer_from_eit[:,1]
print('Number of pixels in SUMER:', len(rowscols_croplat_sumer_from_eit))

######################################################

# Import SUMER data interpolated (wavelength calibrated)
data_interpolated_loaded = np.load('../data/data_modified/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)
# Average spectra of the pixels selected
lam_sumer_av, elam_sumer_av, rad_sumer_av, erad_sumer_av = average_profiles_from_pixels_selected_from_interpolated_data(wavelength_range_=wavelength_range_to_average, data_interpolated_loaded_=data_interpolated_loaded, rows_cols_of_spectroheliogram_croplat=rowscols_croplat_sumer_from_eit)

############################################################################################################
############################################################################################################
############################################################################################################
# Substract HRTS

#subtract HRTS QR-A
fsh_qra = fun_scale_hrts(hrts_qr='a', lamb_0=lamb_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_conv, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction)
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
fsh_qrb = fun_scale_hrts(hrts_qr='b', lamb_0=lamb_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_conv, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction)
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
fsh_qrl = fun_scale_hrts(hrts_qr='l', lamb_0=lamb_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_conv, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction)
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

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9,8))
label_size = 18
ax[0].imshow(data_eit_crop_corrected, norm=LogNorm(vmin=vmin_eit, vmax=vmax_eit), cmap='Greys_r', extent=extent_eit_sumer_arcsec_image)
ax[1].pcolormesh(HPlon_rotcomp, HPlat_croplat, intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
ax[1].axis('equal') # Ensures equal scaling of axis x and y
#ax[1].set_title(f'SUMER', fontsize=22)
#ax[0].set_title(f'EIT-{header_eit["WAVELNTH"]}', fontsize=22)
ax[1].set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
ax[0].text(1.02, 0.5, f'EIT-{header_eit["WAVELNTH"]}', fontsize=22,transform=ax[0].transAxes, va='center', ha='left', rotation=90)
ax[1].text(1.02, 0.5, f'SUMER-{line_center_label}', fontsize=22,transform=ax[1].transAxes, va='center', ha='left', rotation=90)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)
ax[0].grid(color='white')
ax[1].grid(color='white')
ax[0].axvline(x=HPlon_rotcomp[closest_index], linestyle='-', linewidth=0.8, color='red', label='Slit position during\n EIT image')
ax[1].axvline(x=HPlon_rotcomp[closest_index], linestyle='-', linewidth=0.8, color='red', label='Slit position during\n EIT image')
#ax[0].set_xlim([HPlon_rotcomp[0], HPlon_rotcomp[-1]])
#ax[1].set_xlim([HPlon_rotcomp[0], HPlon_rotcomp[-1]])
plt.show(block=False)




fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,14))
label_size = 18
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




fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,10))
label_size = 18
ax[0].imshow(data_eit_crop_corrected, norm=LogNorm(vmin=vmin_eit, vmax=vmax_eit), cmap='Greys_r', extent=extent_eit_sumer_arcsec_image)
ax[1].pcolormesh(HPlon_rotcomp, HPlat_croplat, intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
ax[1].axis('equal') # Ensures equal scaling of axis x and y
ax[1].set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
ax[0].set_title('Contours of the CH in EIT overlaid in the Ne VIII intensity map', fontsize=20)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
ax[0].text(1.02, 0.5, f'EIT-{header_eit["WAVELNTH"]}', fontsize=22,transform=ax[0].transAxes, va='center', ha='left', rotation=90)
ax[1].text(1.02, 0.5, f'SUMER-{line_center_label}', fontsize=22,transform=ax[1].transAxes, va='center', ha='left', rotation=90)
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
ax[0].set_aspect('auto')
ax[1].set_aspect('auto')
plt.show(block=False)




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
ax.errorbar(x=lam_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color='blue', linewidth=1., label='SUMER data')
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
#ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER lowest {range_percentage}%, not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII_corrected_qra, yerr=erad_sumer_cropNeVIII_corrected_qra, color=color_sumer_corrected, linestyle='-', linewidth=2., label=f'SUMER corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_hrtsa_cropNeVIII, lamb_0=lamb_0), y=rad_hrtsa_conv_scaled_cropNeVIII, yerr=erad_hrtsa_conv_scaled_cropNeVIII, color='black', linestyle='--', linewidth=2., label='HRST - QS A') #Real spectrum (SUMER)
ax.axvline(x=0, color='black', linestyle=':', linewidth=2., label='Rest wavelength of Ne VIII/2')
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_title(f'Comparison SUMER before and after correction with HRTS QS-A', fontsize=18)
ax.set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
ax.set_xlim([vkms_doppler(lamb=min(lam_hrtsa_cropNeVIII), lamb_0=lamb_0), vkms_doppler(lamb=max(lam_hrtsa_cropNeVIII), lamb_0=lamb_0)])
ax.legend(fontsize=12)
ax.set_yscale('linear')
plt.show(block=False)


# Plot: Comparison SUMER corrected and uncorrected
fig, ax = plt.subplots(figsize=(12, 5))
#ax.errorbar(x=vkms_doppler(lamb=lam_crop, lamb_0=lamb_0), y=rad_crop, yerr=erad_crop, color='black', linewidth=0.6, label='SUMER box') #Real spectrum (SUMER) 
#ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER lowest {range_percentage}%, not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII_corrected_qrb, yerr=erad_sumer_cropNeVIII_corrected_qrb, color=color_sumer_corrected, linestyle='-', linewidth=2., label=f'SUMER corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_hrtsb_cropNeVIII, lamb_0=lamb_0), y=rad_hrtsb_conv_scaled_cropNeVIII, yerr=erad_hrtsb_conv_scaled_cropNeVIII, color='black', linestyle='--', linewidth=2., label='HRST - QS B') #Real spectrum (SUMER)
ax.axvline(x=0, color='black', linestyle=':', linewidth=2., label='Rest wavelength of Ne VIII/2')
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_title(f'Comparison SUMER before and after correction with HRTS QS-B', fontsize=18)
ax.set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
ax.set_xlim([vkms_doppler(lamb=min(lam_hrtsb_cropNeVIII), lamb_0=lamb_0), vkms_doppler(lamb=max(lam_hrtsb_cropNeVIII), lamb_0=lamb_0)])
ax.legend(fontsize=12)
ax.set_yscale('linear')
plt.show(block=False)


# Plot: Comparison SUMER corrected and uncorrected
fig, ax = plt.subplots(figsize=(12, 5))
#ax.errorbar(x=vkms_doppler(lamb=lam_crop, lamb_0=lamb_0), y=rad_crop, yerr=erad_crop, color='black', linewidth=0.6, label='SUMER box') #Real spectrum (SUMER) 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER lowest {range_percentage}%, not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII_corrected_qrl, yerr=erad_sumer_cropNeVIII_corrected_qrl, color=color_sumer_corrected, linestyle='-', linewidth=2., label=f'SUMER {range_percentage} of the maximum%, corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_hrtsl_cropNeVIII, lamb_0=lamb_0), y=rad_hrtsl_conv_scaled_cropNeVIII, yerr=erad_hrtsl_conv_scaled_cropNeVIII, color='black', linestyle='--', linewidth=2., label='HRST - QS L') #Real spectrum (SUMER)
ax.axvline(x=0, color='black', linestyle=':', linewidth=2., label='Rest wavelength of Ne VIII/2')
ax.axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_title(f'Comparison SUMER before and after correction with HRTS QS-L', fontsize=18)
ax.set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
ax.set_xlim([vkms_doppler(lamb=min(lam_hrtsl_cropNeVIII), lamb_0=lamb_0), vkms_doppler(lamb=max(lam_hrtsl_cropNeVIII), lamb_0=lamb_0)])
ax.legend(fontsize=12)
ax.set_yscale('linear')
plt.show(block=False)




# save average profile as .npy
if save_average_profile_map == 'yes':
    range_numbers_to_string = '__'.join(f"{x:.2f}".replace('.', '_').rstrip('0') if f"{x:.2f}"[-1] != '0' else f"{x:.1f}".replace('.', '_') for x in range_percentage) 
    filename_profile = 'average_profile__' + range_numbers_to_string + '__' + threshold_value_type + '_of_EIT_' + str(eit_wavelength)
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





#######################################
# EIT contours

# Load the intensity map and uncertainties
intensitymap_loaded_dic = np.load('../data/data_modified/dopplermap_BRmap.npz')
ddopplershift_map_binned_HRTSsub_lessmedian = intensitymap_loaded_dic['ddopplershift_map_binned_HRTSsub_lessmedian']
BR_asymmetry_map_gaussian_binned_corrected_normalized = intensitymap_loaded_dic['BR_map']




vmin_vmax = [-12., 12.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
label_size = 18
img = ax.imshow(ddopplershift_map_binned_HRTSsub_lessmedian, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap='seismic', extent=extent_eit_sumer_arcsec_image)
cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cax)
cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
ax.set_title(r'Doppler map, blends corrected', fontsize=20)
ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
#plt.subplots_adjust(left=0.1, right=0.90, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_lower = ax.contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax.contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='black', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
plt.show(block=False)




vmin_vmax_BR = [-1.,1.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
img = ax.imshow(BR_asymmetry_map_gaussian_binned_corrected_normalized, vmin=vmin_vmax_BR[0], vmax=vmin_vmax_BR[1], cmap='seismic', extent=extent_eit_sumer_arcsec_image)
cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cax)
cbar.set_label('Red-blue asymmetry normalized', fontsize=16)
ax.set_title('R-B normalized, blends corrected', fontsize=20)
ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
#plt.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_lower = ax.contour(data_eit_crop_corrected[::-1], levels=[lower_bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax.contour(data_eit_crop_corrected[::-1], levels=[upper_bound_eit], colors='black', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{range_percentage[0]} %'),
    mlines.Line2D([],[],color='black', label=f'{range_percentage[1]} %')]
ax.set_aspect('auto')
plt.show(block=False)



