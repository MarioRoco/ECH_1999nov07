
#  Inputs
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'

# Threshold value: label (type) and range of percentageRange percentage of the threshold value
threshold_value_type = 'max' #'max', 'min', 'mean', 'median'
range_percentage = [0., 4.]

# Wavelength ranges to crop spectra
wavelength_range_to_average = [1531.1147, 1551.7688]
wavelength_range_to_analyze_NeVIII = [1540.2, 1541.4]

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
from utils.data_path import path_data_soho 
from utils.SOHO_aux_functions import *
from utils.calibration_parameters__output import *
from utils.spectroheliogram_functions import *
from autils.solar_rotation_variables import *
from utils.aux_functions import *
from utils.general_variables import *
from utils.NeVIII_rest_wavelength import *

######################################################
######################################################
######################################################

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
lam_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][0] #Angstrom
lam_unc_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][1] #Angstrom
print('Rest wavelength Ne VIII (2nd order):', lam_0, r'$\pm$', lam_unc_0, '\u212B')

######################################################

# Import SUMER data interpolated (wavelength calibrated)
data_interpolated_loaded = np.load('../data/data_modified/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)
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
ax.errorbar(x=lam_sumer_avNeVIII, y=rad_sumer_avNeVIII, yerr=erad_sumer_avNeVIII, color='blue', linewidth=1., label='SUMER data')
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

erad_integrated_sumer_mean_list = [(r2-r1)/2. for r1, r2 in rad_integrated_sumer_range_list]
rad_percentage_range_sumer = range_percentage_sumer_list
rad_percentage_mean_sumer = [(r2+r1)/2. for r1, r2 in rad_percentage_range_sumer]
erad_percentage_mean_sumer = [(r2-r1)/2. for r1, r2 in rad_percentage_range_sumer]

color_list = ['blue', 'red', 'green', 'orange', 'magenta', 'purple', 'lime', 'grey', 'cyan', 'brown', 'blue', 'red', 'green', 'orange', 'magenta', 'purple', 'lime', 'grey', 'cyan', 'brown']

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
ax.errorbar(x=rad_peak_uncorrected_list, xerr=erad_peak_uncorrected_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='.', label='SUMER uncorrected')
ax.errorbar(x=rad_peak_uncorrected_list, xerr=erad_peak_uncorrected_list, y=v_peak_corrected_list, yerr=ev_peak_corrected_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Spectral radiance peak (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.legend()
plt.show(block=False)



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




