# Binning
bin_lat = 4
bin_lon = 1

BR_distance_centroid = 50. #[km/s] Distance of the center of the range from the centroid of the fitted gaussian
BR_width = 50. #[km/s]

line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', 'cold_line'

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
# Import NOT binned data

## Spectral images
data_interpolated_loaded = np.load('../data/data_modified/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)


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
dopplershift_map_loaded_dic = np.load('../outputs/dopplershift_map_'+line_label+f'_binned_lon{bin_lon}_lat{bin_lat}.npz', allow_pickle=True)
dopplershift_map_binned = dopplershift_map_loaded_dic['dopplershift_map_binned'] #2D-array
dopplershift_map_binned_HRTSsub_qra = dopplershift_map_loaded_dic['dopplershift_map_binned_HRTSsub_qra'] #2D-array
dopplershift_map_binned_HRTSsub_qrb = dopplershift_map_loaded_dic['dopplershift_map_binned_HRTSsub_qrb'] #2D-array
dopplershift_map_binned_HRTSsub_qrl = dopplershift_map_loaded_dic['dopplershift_map_binned_HRTSsub_qrl'] #2D-array


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


############################################################
#  Extent in pixels 
#extent_sumer_px_contours = [0., intensitymap_croplat.shape[1]-1, intensitymap_croplat.shape[0]-1, 0.]
#extent_sumer_px_image = [-0.5, intensitymap_croplat.shape[1]-1+0.5, intensitymap_croplat.shape[0]-1+0.5, -0.5]
#extent_sumer_binned_px_contours = [0., intensitymap_croplat_binned.shape[1]-1, intensitymap_croplat_binned.shape[0]-1, 0.]
#extent_sumer_binned_px_image = [-0.5, intensitymap_croplat_binned.shape[1]-1+0.5, intensitymap_croplat_binned.shape[0]-1+0.5, -0.5]

extent_binned_px_image = [-0.5, dopplershift_map_binned.shape[1]-1+0.5, dopplershift_map_binned.shape[0]-1+0.5, -0.5]


# Extent in pixels 
extent_sumer_px_contours = [0., intensity_map_croplat.shape[1]-1, intensity_map_croplat.shape[0]-1, 0.]
extent_sumer_px_image = [-0.5, intensity_map_croplat.shape[1]-1+0.5, intensity_map_croplat.shape[0]-1+0.5, -0.5]


############################################################
# 

vmin_vmax__dopplermap = [-12, 12]
label_size = 18

dopplershift_map_binned_HRTSsub_qra_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qra)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
ax.imshow(dopplershift_map_binned_HRTSsub_qra_lessmedian, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
ax.set_title('Dopplershift map, HRTS: QR-A, median subtracted', fontsize=18)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
plt.show(block=False)


############################################################
# 





############################################################
# 





############################################################
# 









