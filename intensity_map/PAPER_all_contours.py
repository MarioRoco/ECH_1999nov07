#  Inputs

save_paper_images = 'no'
folder_name = '../outputs/paper_figures/maps_with_contours/v1' #name of the folder where you save the images
save_dpi = 100 #resolution: number of pixels per inch. ChatGPT gave me 300 by default. 


# polygon vertices are given as (row, col)
poly_rc, region_id = [[12,104], [12,199], [83,180], [83,127]], 'QSa6' #QS a6


line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'
eit_wavelength = 195 #171, 195, 284, or 304 [Angstrom]
eit_time = 'late' #'early' or 'late' (early: around 1 or 4 am; late: around 6 or 7 am)

# EIT's threshold value: percentage of the threshold value and label of the threshold value
percentage_eit, threshold_value_type_eit = 4., 'max' #'max', 'min', 'mean', 'median'

# SUMER's's threshold value: percentage of the threshold value and label of the threshold value
percentage_sumer, threshold_value_type_sumer = 6.5, 'max' #'max', 'min', 'mean', 'median'



# Full Sun
show_sumer_FOV = 'yes'
show_contours_fullsun = 'yes'
vmin_eit_fullsun, vmax_eit_fullsun = 5.5e1, 5e3 
contour_intensity_eit_fullsun = 135. #110.
#contour_intensity_eit_fullsun = 'upper_bound'

legend_size = 13

color_contours_eit_Imap = 'yellow'
color_contours_sumer_Imap = 'magenta'

color_contours_eit_dopplermap, linestyle_eit_dopplermap = 'springgreen', '-'
color_contours_sumer_dopplermap, linestyle_sumer_dopplermap = 'black', '-' #'springgreen', 'green', 'lime', 'olive', 'forestgreen', 'seagreen', 'springgreen', 'darkgreen', 'lightgreen', 'mediumseagreen', 'palegreen', 'yellowgreen', 'chartreuse', 'lawngreen'; 'teal', 'black', 'gold', 'yellow', 'orange', 'coral', 'cyan', 'darkgreen'


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
"""
HPlat_slit_croplat_corrected = HPlat_croplat[[0,-1]]


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
######################################################
######################################################

# Bound EIT
_, bound_eit = get_bounds(intensitymap_croplat=data_eit_crop, range_percentage=[0., percentage_eit], threshold_value_type=threshold_value_type_eit)
print('bound_eit =', bound_eit)

#Bound SUMER
# Define intensity bin
_, bound_sumer = get_bounds(intensitymap_croplat=intensity_map_croplat, range_percentage=[0., percentage_sumer], threshold_value_type=threshold_value_type_sumer)
print('bound_sumer =', bound_sumer)


######################################################
######################################################
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

contours_region = find_contours(mask.astype(float), 0.5)

######################################################

ZZ = data_eit_crop_corrected # Z is your 2D array

delta_lon_left = np.abs(HPlon_rotcomp[1] - HPlon_rotcomp[0])
delta_lon_right = np.abs(HPlon_rotcomp[-2] - HPlon_rotcomp[-1])
xlon1 = -0.5
xlon2 = ZZ.shape[1]-1.+0.5
ylon1 = HPlon_rotcomp[0] - delta_lon_left/2.
ylon2 = HPlon_rotcomp[-1] + delta_lon_right/2.
mlon = (ylon2-ylon1)/(xlon2-xlon1)
blon = HPlon_rotcomp[0]
HPlon_eit = mlon * np.arange(0, ZZ.shape[1]) + blon
print('left eit: ', ylon1)
print('left sumer: ', HPlon_rotcomp[0]-np.abs(HPlon_rotcomp[1]-HPlon_rotcomp[0])/2.)
print('right eit: ', ylon2)
print('right sumer: ', HPlon_rotcomp[-1]+np.abs(HPlon_rotcomp[-1]-HPlon_rotcomp[-2])/2.)




delta_lat_high = np.abs(HPlat_croplat[1] - HPlat_croplat[0])
delta_lat_low =  np.abs(HPlat_croplat[-1] - HPlat_croplat[-2])
xlat1 = -0.5
xlat2 = ZZ.shape[0]-1.+0.5
ylat1 = HPlat_croplat[0] + delta_lat_high/2.
ylat2 = HPlat_croplat[-1] - delta_lat_low/2.
mlat = (ylat2-ylat1)/(xlat2-xlat1)
blat = HPlat_croplat[0]
HPlat_eit = mlat * np.arange(0, ZZ.shape[0]) + blat
print('low eit: ', ylat1)
print('low sumer: ', HPlat_croplat[0]+np.abs(HPlat_croplat[1]-HPlat_croplat[0])/2.)
print('high eit: ', ylat2)
print('high sumer: ', HPlat_croplat[-1]-np.abs(HPlat_croplat[-1]-HPlat_croplat[-2])/2.)


HPlon_eit = np.asarray(HPlon_eit)
HPlat_eit = np.asarray(HPlat_eit)

######################################################
######################################################
######################################################

# polygon vertices are given as (row, col)
poly_rc_left_region = [[27,21], [10,58], [17,82], [37,71], [39,47], [44,28], [35,17]] #left part
poly_rc_left_region = np.array(poly_rc_left_region)


######################################################
# Create a polygon that enclose the left region of the CH and mark all pixels inside the polygon and the CH

from skimage.measure import find_contours
from matplotlib.path import Path
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# polygon vertices in pixel indices
poly_path_left_region = Path(poly_rc_left_region[:, ::-1])

nrows, ncols = data_eit_crop_corrected.shape
rr, cc = np.indices((nrows, ncols))
inside_poly_left_region = poly_path_left_region.contains_points(np.c_[cc.ravel(), rr.ravel()]).reshape(data_eit_crop_corrected.shape)
inside_level = data_eit_crop_corrected <= bound_eit
mask_left_region = inside_poly_left_region & inside_level

######################################################
######################################################
######################################################


######################################################
# Show image of the intensity map with the contours and the pixels inside the contours

"""
### EIT and SUMER maps with contours
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
contour_lower = ax[0].contour(data_eit_crop_corrected[::-1], levels=[bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax[0].contour(intensity_map_croplat[::-1], levels=[bound_sumer], colors='yellow', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color='red', label=f'{bound_eit} %'),
    mlines.Line2D([],[],color='yellow', label=f'{bound_sumer} %')]
contour_lower = ax[1].contour(data_eit_crop_corrected[::-1], levels=[bound_eit], colors='red', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_upper = ax[1].contour(intensity_map_croplat[::-1], levels=[bound_sumer], colors='yellow', linewidths=2, extent=extent_eit_sumer_arcsec_contours)
#ax[0].axvline(x=HPlon_slit_rotcomp_corrected, linewidth=1.5, color='red', label='slit position')
ax[0].set_aspect('auto')
ax[1].set_aspect('auto')
if save_paper_images == 'yes':
    fig_name = 'intensity_maps_SUMER_EIT_and_contours'
    plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)
"""




### EIT and SUMER maps with contours
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9.8,10.5), sharex=True)
vmin_eit, vmax_eit = 4e1, 3e3
#ax[0].imshow(data_eit_crop_corrected, norm=LogNorm(vmin=vmin_eit, vmax=vmax_eit), cmap='Greys_r', extent=extent_eit_sumer_arcsec_image)
ax[0].imshow(data_eit_crop_corrected, norm=LogNorm(vmin=vmin_eit, vmax=vmax_eit), cmap='Greys_r', extent=extent_eit_sumer_arcsec_image)
#ax[0].imshow(data_eit_crop_corrected, norm=LogNorm(), cmap='Greys_r', extent=extent_eit_sumer_arcsec_image)
#ax[1].pcolormesh(HPlon_rotcomp, HPlat_croplat, intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
ax[1].imshow(intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer), extent=extent_eit_sumer_arcsec_image)
ax[1].axis('equal') # Ensures equal scaling of axis x and y
ax[1].set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
ax[0].text(1.02, 0.5, f'EIT-{header_eit["WAVELNTH"]}', fontsize=22,transform=ax[0].transAxes, va='center', ha='left', rotation=90)
ax[1].text(1.02, 0.5, f'SUMER-{line_center_label}', fontsize=22,transform=ax[1].transAxes, va='center', ha='left', rotation=90)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.95, wspace=0, hspace=0)
#ax[0].axvline(x=HPlon_slit_rotcomp_corrected, linestyle='-', linewidth=0.8, color='red', label='Slit position during\n EIT image')
#ax[1].axvline(x=HPlon_slit_rotcomp_corrected, linestyle='-', linewidth=0.8, color='red')
#contour_sumer = ax[0].contour(intensity_map_croplat[::-1], levels=[bound_sumer], colors='orange', linewidths=1, extent=extent_eit_sumer_arcsec_contours)
#contour_sumer = ax[1].contour(intensity_map_croplat[::-1], levels=[bound_sumer], colors='orange', linewidths=1, extent=extent_eit_sumer_arcsec_contours)
contour_eit0 = ax[0].contour(data_eit_crop_corrected[::-1], levels=[bound_eit], colors=color_contours_eit_Imap, linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_sumer0 = ax[0].contour(intensity_map_croplat[::-1], levels=[bound_sumer], colors=color_contours_sumer_Imap, linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color=color_contours_eit_Imap, label=f'{bound_eit} %'),
    mlines.Line2D([],[],color=color_contours_sumer_Imap, label=f'{bound_sumer} %')]
contour_eit1 = ax[1].contour(data_eit_crop_corrected[::-1], levels=[bound_eit], colors=color_contours_eit_Imap, linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_sumer1 = ax[1].contour(intensity_map_croplat[::-1], levels=[bound_sumer], colors=color_contours_sumer_Imap, linewidths=2, extent=extent_eit_sumer_arcsec_contours)
#ax[0].set_xlim([HPlon_rotcomp[0], HPlon_rotcomp[-1]])
#ax[1].set_xlim([HPlon_rotcomp[0], HPlon_rotcomp[-1]])
contours = find_contours(mask.astype(float), 0.5)
for c in contours:
    rr_c = c[:, 0]
    cc_c = c[:, 1]
    x = np.interp(cc_c, np.arange(len(HPlon_rotcomp)), HPlon_rotcomp)
    y = np.interp(rr_c, np.arange(len(HPlat_croplat)), HPlat_croplat)
    ax[0].plot(x, y, color='cyan', linewidth=1.5)
    ax[1].plot(x, y, color='cyan', linewidth=1.5)
"""
contours_left_region = find_contours(mask_left_region.astype(float), 0.5)
for c in contours_left_region:
    rr_c = c[:, 0]
    cc_c = c[:, 1]
    x = np.interp(cc_c, np.arange(len(HPlon_eit)), HPlon_eit)
    y = np.interp(rr_c, np.arange(len(HPlat_eit)), HPlat_eit)
    ax[0].plot(x, y, color='red', linewidth=1.5)
    ax[1].plot(x, y, color='red', linewidth=1.5)
"""
if save_paper_images == 'yes':
    fig_name = 'intensity_maps_SUMER_EIT_and_contours'
    plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)


##############################################################
##############################################################
##############################################################
# Dopplermap and BR asymmetry map

# Load the intensity map and uncertainties
dopplermap_BRmap_loaded_dic = np.load('../outputs/dopplermap_BRmap.npz')
ddopplershift_map_binned_HRTSsub_lessmedian = dopplermap_BRmap_loaded_dic['ddopplershift_map_binned_HRTSsub_lessmedian']
BR_asymmetry_map_gaussian_binned_corrected_normalized = dopplermap_BRmap_loaded_dic['BR_map']



### PAPER image: Dopplermap (corrected from blends) with contours of EIT and SUMER
vmin_vmax = [-12., 12.]
vmin_vmax = [-17., 17.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
label_size = 18
img = ax.imshow(ddopplershift_map_binned_HRTSsub_lessmedian, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap='seismic', extent=extent_eit_sumer_arcsec_image)
cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cax)
cbar.set_label(f'Doppler shift (km/s)', fontsize=16)
ax.set_title(r'Dopplergram', fontsize=20)
#ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
#plt.subplots_adjust(left=0.1, right=0.90, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_sumer = ax.contour(intensity_map_croplat[::-1], levels=[bound_sumer], colors=color_contours_sumer_dopplermap, linestyles=linestyle_sumer_dopplermap, linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_eit = ax.contour(data_eit_crop_corrected[::-1], levels=[bound_eit], colors=color_contours_eit_dopplermap, linestyles=linestyle_eit_dopplermap, linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color=color_contours_eit_dopplermap, linestyle=linestyle_eit_dopplermap, label=f'{bound_eit}'),
    mlines.Line2D([],[],color=color_contours_sumer_dopplermap, linestyle=linestyle_sumer_dopplermap, label=f'{bound_sumer}')]
ax.set_aspect('auto')
if save_paper_images == 'yes':
	fig_name = 'dopplermap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)



### PAPER image: B-R asymmetry map (normalized and corrected from blends) with contours of EIT and SUMER
vmin_vmax_BR = [-1.,1.]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
img = ax.imshow(BR_asymmetry_map_gaussian_binned_corrected_normalized, vmin=vmin_vmax_BR[0], vmax=vmin_vmax_BR[1], cmap='seismic', extent=extent_eit_sumer_arcsec_image)
cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cax)
cbar.set_label('Red-blue asymmetry normalized', fontsize=16)
ax.set_title('R-B asymmetry map', fontsize=20)
#ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=16)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=16)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
#plt.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.95, wspace=0, hspace=0)
contour_sumer = ax.contour(intensity_map_croplat[::-1], levels=[bound_sumer], colors=color_contours_sumer_dopplermap, linestyles=linestyle_sumer_dopplermap, linewidths=2, extent=extent_eit_sumer_arcsec_contours)
contour_eit = ax.contour(data_eit_crop_corrected[::-1], levels=[bound_eit], colors=color_contours_eit_dopplermap, linestyles=linestyle_eit_dopplermap, linewidths=2, extent=extent_eit_sumer_arcsec_contours)
legend_elements = [
    mlines.Line2D([],[],color=color_contours_eit_dopplermap, linestyle=linestyle_eit_dopplermap, label=f'{bound_eit}'),
    mlines.Line2D([],[],color=color_contours_sumer_dopplermap, linestyle=linestyle_sumer_dopplermap, label=f'{bound_sumer}')]
ax.set_aspect('auto')
if save_paper_images == 'yes':
	fig_name = 'asymmetrymap_NeVIII'
	plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
plt.show(block=False)




##############################################################
##############################################################
##############################################################
# 




