#  Inputs
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'
instrument_line = 'sumer_NeVIII'

#range_percentage_list = [[0.,10.], [10.,20.], [20.,30.], [30.,40.], [40.,50.], [50.,60.], [60.,70.], [70.,80.], [80.,90.], [90.,100.]]
#range_percentage_list = [[0., 5.], [5., 10.], [10., 15.], [15., 20.],[20., 25.], [25., 30.], [30., 35.], [35., 40.],[40., 45.], [45., 50.], [50., 55.], [55., 60.],[60., 65.], [65., 70.], [70., 75.], [75., 80.],[80., 85.], [85., 90.], [90., 95.], [95., 100.]]
#range_percentage_list = [[0., 2.], [2., 4.], [4., 6.], [6., 8.], [8., 10.],[10., 12.], [12., 14.], [14., 16.], [16., 18.], [18., 20.],[20., 22.], [22., 24.], [24., 26.], [26., 28.], [28., 30.],[30., 32.], [32., 34.], [34., 36.], [36., 38.], [38., 40.],[40., 42.], [42., 44.], [44., 46.], [46., 48.], [48., 50.],[50., 52.], [52., 54.], [54., 56.], [56., 58.], [58., 60.],[60., 62.], [62., 64.], [64., 66.], [66., 68.], [68., 70.],[70., 72.], [72., 74.], [74., 76.], [76., 78.], [78., 80.],[80., 82.], [82., 84.], [84., 86.], [86., 88.], [88., 90.],[90., 92.], [92., 94.], [94., 96.], [96., 98.], [98., 100.]]
#range_percentage_list = [[0.,3.], [3.,6.], [6.,10.], [10.,20.], [20.,30.], [30.,40.], [40.,60.], [60.,100.]]

#range_percentage_list = [[0.,5.], [5.,10.], [10.,20.], [20.,30.], [30.,40.], [40.,50.], [50.,60.], [60.,80.], [80.,100.]]
#range_percentage_list = [[0.,5.], [5.,10.], [10.,20.], [20.,30.], [30.,40.], [40.,50.], [50.,70.], [70.,100.]]
#range_percentage_list = [[0.,3.], [3.,4.], [4.,6.],[6.,8.], [8.,10.], [10.,20.], [20.,30.], [30.,40.], [40.,50.], [50.,70.], [70.,100.]]



#range_percentage_list = [[0.,4.], [4.,6.], [6.,8.], [8.,10.], [10.,15.], [15.,25.], [25.,40.], [40.,100.]]





range_percentage_list = [[0.,3.], [3.,4.], [4.,6.],[6.,8.], [8.,10.], [10.,15.], [15.,25.], [25.,40.], [40.,100.]]

#range_percentage_list = [[0., 5.], [5., 10.], [10., 15.], [15., 20.],[20., 25.], [25., 30.], [30., 35.], [35., 40.],[40., 45.], [45., 50.], [50., 55.], [55., 60.],[60., 65.], [65., 70.], [70., 75.], [75., 80.],[80., 85.], [85., 90.], [90., 95.], [95., 100.]]



# Wavelength ranges to crop spectra
wavelength_range_to_average = [1531.1147, 1551.7688]
wavelength_range_to_analyze_NeVIII = [1540.2, 1541.4]

threshold_value_type_sumer = 'max' #'max', 'min', 'mean', 'median'


show_plots_correction = 'yes'
show_plots_iterations = 'yes'

# save average profile as .npy?
save_average_profile_map = 'no' 


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

show_secondary_plots = 'yes'

save_paper_images = 'no'
folder_name = '../outputs' #name of the folder where you save the images
save_dpi = 100 #resolution: number of pixels per inch. ChatGPT gave me 300 by default. 


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

######################################################
######################################################
######################################################

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
######################################################
######################################################
# Fitting

######################################################
# Initial parameters of the fitting


#average_profile__0_0__60_0__mean_of_eit_195.npz

wavelength_range_NeVIII = [1540.32, 1541.43]


bckg_fit_uncorrected = 0.2 #HRST not subtracted
init_parameters_uncorrected = [bckg_fit_uncorrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.3-bckg_fit_uncorrected, -54., 20.,
1.07-bckg_fit_uncorrected, -10, 45.,
0.5-bckg_fit_uncorrected, 15., 45.,
0.25-bckg_fit_uncorrected, 97., 30.
]

"""
bckg_fit_corrected_qra = -0.3
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.1-bckg_fit_corrected_qra, -60., 20.,
0.5-bckg_fit_corrected_qra, -30., 20.,
3.-bckg_fit_corrected_qra, 0.0, 50.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
0.5-bckg_fit_corrected_qrb, -30., 40.,
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
"""

init_parameters_corrected_qra_list, init_parameters_corrected_qrb_list = [],[]



#[0.,3.]

bckg_fit_corrected_qra = 0.0
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.13-bckg_fit_corrected_qra, -50., 30.,
0.23-bckg_fit_corrected_qra, 10., 30.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
#0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
0.1-bckg_fit_corrected_qrb, -40., 40.,
0.24-bckg_fit_corrected_qrb, 0.0, 40.,
#1.5-bckg_fit_corrected_qrb, 33., 40.
]

init_parameters_corrected_qra_list.append(init_parameters_corrected_qra)
init_parameters_corrected_qrb_list.append(init_parameters_corrected_qrb)




#[3.,4.]

bckg_fit_corrected_qra = 0.0
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.13-bckg_fit_corrected_qra, -50., 30.,
0.23-bckg_fit_corrected_qra, 10., 30.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
#0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
0.1-bckg_fit_corrected_qrb, -40., 40.,
0.24-bckg_fit_corrected_qrb, 0.0, 40.,
#1.5-bckg_fit_corrected_qrb, 33., 40.
]
init_parameters_corrected_qra_list.append(init_parameters_corrected_qra)
init_parameters_corrected_qrb_list.append(init_parameters_corrected_qrb)



#[4.,6.]

bckg_fit_corrected_qra = 0.0
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.13-bckg_fit_corrected_qra, -50., 30.,
0.23-bckg_fit_corrected_qra, 10., 30.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
#0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
0.1-bckg_fit_corrected_qrb, -40., 40.,
0.24-bckg_fit_corrected_qrb, 0.0, 40.,
#1.5-bckg_fit_corrected_qrb, 33., 40.
]
init_parameters_corrected_qra_list.append(init_parameters_corrected_qra)
init_parameters_corrected_qrb_list.append(init_parameters_corrected_qrb)




#[6.,8.]

bckg_fit_corrected_qra = 0.0
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.13-bckg_fit_corrected_qra, -50., 30.,
0.7-bckg_fit_corrected_qra, 0., 30.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
#0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
0.15-bckg_fit_corrected_qrb, -50., 40.,
0.7-bckg_fit_corrected_qrb, 0.0, 40.,
#1.5-bckg_fit_corrected_qrb, 33., 40.
]
init_parameters_corrected_qra_list.append(init_parameters_corrected_qra)
init_parameters_corrected_qrb_list.append(init_parameters_corrected_qrb)



#[8.,10.]

bckg_fit_corrected_qra = 0.0
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.13-bckg_fit_corrected_qra, -50., 30.,
1.-bckg_fit_corrected_qra, 0., 30.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
#0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
0.1-bckg_fit_corrected_qrb, -50., 40.,
1.-bckg_fit_corrected_qrb, 0.0, 40.,
#1.5-bckg_fit_corrected_qrb, 33., 40.
]
init_parameters_corrected_qra_list.append(init_parameters_corrected_qra)
init_parameters_corrected_qrb_list.append(init_parameters_corrected_qrb)


#[10.,15.]

bckg_fit_corrected_qra = 0.0
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.13-bckg_fit_corrected_qra, -50., 30.,
1.-bckg_fit_corrected_qra, 0., 30.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
#0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
0.1-bckg_fit_corrected_qrb, -50., 40.,
1.-bckg_fit_corrected_qrb, 0.0, 40.,
#1.5-bckg_fit_corrected_qrb, 33., 40.
]
init_parameters_corrected_qra_list.append(init_parameters_corrected_qra)
init_parameters_corrected_qrb_list.append(init_parameters_corrected_qrb)


#[15.,25.]

bckg_fit_corrected_qra = 0.0
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.13-bckg_fit_corrected_qra, -50., 30.,
2.-bckg_fit_corrected_qra, 0., 40.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
#0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
0.1-bckg_fit_corrected_qrb, -50., 40.,
2.-bckg_fit_corrected_qrb, 0.0, 40.,
#1.5-bckg_fit_corrected_qrb, 33., 40.
]
init_parameters_corrected_qra_list.append(init_parameters_corrected_qra)
init_parameters_corrected_qrb_list.append(init_parameters_corrected_qrb)



#[25.,40.]

bckg_fit_corrected_qra = 0.0
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.13-bckg_fit_corrected_qra, -50., 40.,
3.4-bckg_fit_corrected_qra, 0., 50.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
#0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
0.13-bckg_fit_corrected_qrb, -50., 40.,
3.3-bckg_fit_corrected_qrb, 0.0, 50.,
#1.5-bckg_fit_corrected_qrb, 33., 40.
]
init_parameters_corrected_qra_list.append(init_parameters_corrected_qra)
init_parameters_corrected_qrb_list.append(init_parameters_corrected_qrb)



#[40.,100.]

bckg_fit_corrected_qra = 0.0
init_parameters_corrected_qra = [bckg_fit_corrected_qra, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
0.13-bckg_fit_corrected_qra, -50., 40.,
6.-bckg_fit_corrected_qra, 0., 50.,
#1.5-bckg_fit_corrected_qra, 25., 30.
]

bckg_fit_corrected_qrb = 0.0
init_parameters_corrected_qrb = [bckg_fit_corrected_qrb, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
#0.04-bckg_fit_corrected_qrb, -78., 30.,
#0.3-bckg_fit_corrected_qrb, -50., 30.,
0.13-bckg_fit_corrected_qrb, -50., 40.,
6.-bckg_fit_corrected_qrb, 0.0, 50.,
#1.5-bckg_fit_corrected_qrb, 33., 40.
]
init_parameters_corrected_qra_list.append(init_parameters_corrected_qra)
init_parameters_corrected_qrb_list.append(init_parameters_corrected_qrb)

######################################################
######################################################
######################################################
# 


# rows and columns inside the intensity bin
def pixels_higher_intensity_than_a_value(arr_2d, bound_, N_pixels):
    mask_ = arr_2d > bound_                    # keep only pixels brighter than the value "bound_"
    values_ = arr_2d[mask_] #values (intensity) of the pixels brighter than the value "bound_"
    coords_ = np.argwhere(mask_) #coordinates (row, col) of the pixels brighter than the value "bound_"
    idx_sorted = np.argsort(values_)   # indices that sort values in ascending order 
    coords_sorted = coords_[idx_sorted]      # coordinates ordered by brightness
    rowscols_inside_range__ = coords_sorted[:N_pixels] # Take first N_pixels (brightest N_pixels pixels)
    return rowscols_inside_range__


# Extent in pixels 
extent_sumer_px_contours = [0., intensity_map_croplat.shape[1]-1, intensity_map_croplat.shape[0]-1, 0.]
extent_sumer_px_image = [-0.5, intensity_map_croplat.shape[1]-1+0.5, intensity_map_croplat.shape[0]-1+0.5, -0.5]



########################### uncorrected
lam_peak_uncorrected_list = []
elam_peak_uncorrected_list = []
v_peak_uncorrected_list = []
ev_peak_uncorrected_list = []
########################### corrected QR A
lam_peak_corrected_qra_list = []
elam_peak_corrected_qra_list = []
v_peak_corrected_qra_list = []
ev_peak_corrected_qra_list = []
rad_peak_corrected_qra_list = []
########################### corrected QR B
lam_peak_corrected_qrb_list = []
elam_peak_corrected_qrb_list = []
v_peak_corrected_qrb_list = []
ev_peak_corrected_qrb_list = []
rad_peak_corrected_qrb_list = []
########################### corrected QR L
"""
lam_peak_corrected_qrl_list = []
elam_peak_corrected_qrl_list = []
v_peak_corrected_qrl_list = []
ev_peak_corrected_qrl_list = []
rad_peak_corrected_qrl_list = []
"""
###########################
bound_mean_list, bound_unc_list = [],[]

percentage_list, percentage_unc_list = [],[]
centroid_weak_component_qra, intensity_weak_component_qra, centroid_main_component_qra, intensity_main_component_qra, fwhm_weak_component_qra, fwhm_main_component_qra = [],[],[],[],[],[]
unc_centroid_weak_component_qra, unc_centroid_main_component_qra = [],[]
unc_intensity_weak_component_qra, unc_intensity_main_component_qra = [],[]
centroid_weak_component_qrb, intensity_weak_component_qrb, centroid_main_component_qrb, intensity_main_component_qrb, fwhm_weak_component_qrb, fwhm_main_component_qrb = [],[],[],[],[],[]
unc_centroid_weak_component_qrb, unc_centroid_main_component_qrb = [],[]
unc_intensity_weak_component_qrb, unc_intensity_main_component_qrb = [],[]
area_weak_component_qra, area_main_component_qra, area_weak_component_qrb, area_main_component_qrb = [],[],[],[]
unc_area_weak_component_qra, unc_area_main_component_qra, unc_area_weak_component_qrb, unc_area_main_component_qrb = [],[],[],[]
scaling_factor_list_qsa, scaling_factor_unc_list_qsa, chi2_red_list_qsa = [],[],[]
scaling_factor_list_qsb, scaling_factor_unc_list_qsb, chi2_red_list_qsb = [],[],[]
for ii, range_percentage_i in enumerate(range_percentage_list):
	"""
	lower_bound, upper_bound = get_bounds(intensitymap_croplat=intensity_map_croplat, range_percentage=range_percentage_i, threshold_value_type=threshold_value_type_sumer)

	# rows and columns inside the intensity bin  in EIT ,map
	rowscols_inside_range_i = np.argwhere((intensity_map_croplat>=lower_bound) & (intensity_map_croplat<=upper_bound))
	y_row_list_plot = rowscols_inside_range_i[:,0] # convert the list of pairs [row, column] into 2 lists of rows and columns (for the scatterplot)
	x_col_list_plot = rowscols_inside_range_i[:,1]
	"""
	
	init_parameters_corrected_qra = init_parameters_corrected_qra_list[ii]
	init_parameters_corrected_qrb = init_parameters_corrected_qrb_list[ii]

	lower_bound, upper_bound = get_bounds(intensitymap_croplat=intensity_map_croplat, range_percentage=range_percentage_i, threshold_value_type=threshold_value_type_sumer)
	#lower_bound, upper_bound = get_bounds(intensitymap_croplat=np.log(intensity_map_croplat), range_percentage=range_percentage_i, threshold_value_type=threshold_value_type_sumer)

	# rows and columns inside the intensity bin  in EIT ,map
	rowscols_inside_range_i = np.argwhere((intensity_map_croplat>=lower_bound) & (intensity_map_croplat<=upper_bound))
	y_row_list_plot = rowscols_inside_range_i[:,0] # convert the list of pairs [row, column] into 2 lists of rows and columns (for the scatterplot)
	x_col_list_plot = rowscols_inside_range_i[:,1]

	bound_mean_list.append((lower_bound+upper_bound)/2.)
	bound_unc_list.append((upper_bound-lower_bound)/2.)


	# Import SUMER data interpolated (wavelength calibrated)
	data_interpolated_loaded = np.load('../data/data_modified/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)
	# Average spectra of the pixels selected
	lam_sumer_av, elam_sumer_av, rad_sumer_av, erad_sumer_av = average_profiles_from_pixels_selected_from_interpolated_data(wavelength_range_=wavelength_range_to_average, data_interpolated_loaded_=data_interpolated_loaded, rows_cols_of_spectroheliogram_croplat=rowscols_inside_range_i)
	
	
	
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
	

	#################################################
	#################################################
	#################################################
	

	# crop near Ne VIII
	lam_sumer_cropNeVIII, idx_sumer_crop_ = crop_range(list_to_crop=lam_sumer_av, range_values=wavelength_range_to_analyze_NeVIII)
	elam_sumer_cropNeVIII = elam_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
	rad_sumer_cropNeVIII_uncorrected = rad_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
	erad_sumer_cropNeVIII_uncorrected = erad_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]




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
	
	"""
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
	"""
	

	#########################################################
	# Profile of SUMER uncorrected and corrected (using the 3 spectra of HRTS)

	## HRTS wavelength and Doppler velicity
	v_hrtsa_cropNeVIII = vkms_doppler(lamb=lam_hrtsa_cropNeVIII, lamb_0=lam_0)
	v_hrtsb_cropNeVIII = vkms_doppler(lamb=lam_hrtsb_cropNeVIII, lamb_0=lam_0)
	#v_hrtsl_cropNeVIII = vkms_doppler(lamb=lam_hrtsl_cropNeVIII, lamb_0=lam_0)


	## SUMER wavelength and Doppler velocity (respect to the rest wavelength of Ne VIII 770 in 2nd order)
	v_sumer_cropNeVIII = vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lam_0)

	x_lims_fits = [min(v_sumer_cropNeVIII), max(v_sumer_cropNeVIII)]

	#########################################################


	######################################################
	# 6) calculate peak

	# uncorrected data
	mpi_uncorrected = find_maximum_by_parabolic_interpolation_adapted(wavelength=lam_sumer_cropNeVIII, radiance=rad_sumer_cropNeVIII_uncorrected, radiance_unc=erad_sumer_cropNeVIII_uncorrected, show_figure='no')
	mpi_uncorrected["v_vertex"] = vkms_doppler(lamb=mpi_uncorrected["x_vertex"], lamb_0=lam_0) #convert wavelength to speed
	mpi_uncorrected["v_unc_vertex"] = vkms_doppler_unc(lamb=mpi_uncorrected["x_vertex"], lamb_unc=mpi_uncorrected["x_unc_vertex"], lamb_0=lam_0, lamb_0_unc=lam_unc_0) 

	# corrected data QR-A
	mpi_corrected_qra = find_maximum_by_parabolic_interpolation_adapted(wavelength=lam_sumer_cropNeVIII, radiance=rad_sumer_cropNeVIII_corrected_qra, radiance_unc=erad_sumer_cropNeVIII_corrected_qra, show_figure='no')
	mpi_corrected_qra["v_vertex"] = vkms_doppler(lamb=mpi_corrected_qra["x_vertex"], lamb_0=lam_0) #convert wavelength to speed
	mpi_corrected_qra["v_unc_vertex"] = vkms_doppler_unc(lamb=mpi_corrected_qra["x_vertex"], lamb_unc=mpi_corrected_qra["x_unc_vertex"], lamb_0=lam_0, lamb_0_unc=lam_unc_0) 

	# corrected data QR-B
	mpi_corrected_qrb = find_maximum_by_parabolic_interpolation_adapted(wavelength=lam_sumer_cropNeVIII, radiance=rad_sumer_cropNeVIII_corrected_qrb, radiance_unc=erad_sumer_cropNeVIII_corrected_qrb, show_figure='no')
	mpi_corrected_qrb["v_vertex"] = vkms_doppler(lamb=mpi_corrected_qrb["x_vertex"], lamb_0=lam_0) #convert wavelength to speed
	mpi_corrected_qrb["v_unc_vertex"] = vkms_doppler_unc(lamb=mpi_corrected_qrb["x_vertex"], lamb_unc=mpi_corrected_qrb["x_unc_vertex"], lamb_0=lam_0, lamb_0_unc=lam_unc_0) 
	
	"""
	# corrected data QR-L
	mpi_corrected_qrl = find_maximum_by_parabolic_interpolation_adapted(wavelength=lam_sumer_cropNeVIII, radiance=rad_sumer_cropNeVIII_corrected_qrl, radiance_unc=erad_sumer_cropNeVIII_corrected_qrl, show_figure='no')
	mpi_corrected_qrl["v_vertex"] = vkms_doppler(lamb=mpi_corrected_qrl["x_vertex"], lamb_0=lam_0) #convert wavelength to speed
	mpi_corrected_qrl["v_unc_vertex"] = vkms_doppler_unc(lamb=mpi_corrected_qrl["x_vertex"], lamb_unc=mpi_corrected_qrl["x_unc_vertex"], lamb_0=lam_0, lamb_0_unc=lam_unc_0) 
	"""

	######################################################
	######################################################
	######################################################


	########################### uncorrected')
	lam_peak_uncorrected_list.append(mpi_uncorrected["x_vertex"])
	elam_peak_uncorrected_list.append(mpi_uncorrected["x_unc_vertex"])
	v_peak_uncorrected_list.append(mpi_uncorrected["v_vertex"])
	ev_peak_uncorrected_list.append(mpi_uncorrected["v_unc_vertex"])
	########################### corrected QR-A')
	lam_peak_corrected_qra_list.append(mpi_corrected_qra["x_vertex"])
	elam_peak_corrected_qra_list.append(mpi_corrected_qra["x_unc_vertex"])
	v_peak_corrected_qra_list.append(mpi_corrected_qra["v_vertex"])
	ev_peak_corrected_qra_list.append(mpi_corrected_qra["v_unc_vertex"])
	########################### corrected QR-B')
	lam_peak_corrected_qrb_list.append(mpi_corrected_qrb["x_vertex"])
	elam_peak_corrected_qrb_list.append(mpi_corrected_qrb["x_unc_vertex"])
	v_peak_corrected_qrb_list.append(mpi_corrected_qrb["v_vertex"])
	ev_peak_corrected_qrb_list.append(mpi_corrected_qrb["v_unc_vertex"])
	########################### corrected QR-L')
	"""
	lam_peak_corrected_qrl_list.append(mpi_corrected_qrl["x_vertex"])
	elam_peak_corrected_qrl_list.append(mpi_corrected_qrl["x_unc_vertex"])
	v_peak_corrected_qrl_list.append(mpi_corrected_qrl["v_vertex"])
	ev_peak_corrected_qrl_list.append(mpi_corrected_qrl["v_unc_vertex"])
	"""
	###########################
	percentage_list.append(np.mean(range_percentage_i))
	percentage_unc_list.append(np.mean(range_percentage_i) - range_percentage_i[0])


	######################################################
	######################################################
	######################################################


	######################################################
	# Show image of the intensity map with the contours and the pixels inside the contours
	
	if show_plots_iterations=='yes':
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5.7))
		img = ax.imshow(intensity_map_croplat, cmap='Greys_r', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer), extent=extent_sumer_px_image)
		cbar = fig.colorbar(img, ax=ax, pad=0.03)
		ax.set_title(f'{range_percentage_i[0]} - {range_percentage_i[1]} %', fontsize=title_size)
		#ax.set_title(f'SOHO/SUMER intensity map {line_center_label}, solar rotation NOT compensated')
		ax.set_xlabel('Helioprojective longitude (arcsec), rotation compensated')
		ax.set_ylabel('Helioprojective latitude (arcsec)')
		ax.axis('auto') # Ensures equal scaling of axis x and y
		ax.scatter(x=x_col_list_plot, y=y_row_list_plot, s=1, color='yellow')
		contour_lower = ax.contour(intensity_map_croplat[::-1], levels=[lower_bound], colors='red', linewidths=1, extent=extent_sumer_px_contours)
		contour_upper = ax.contour(intensity_map_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=1, extent=extent_sumer_px_contours)
		legend_elements = [
		mlines.Line2D([],[],color='red', label=f'{range_percentage_i[0]} %'),
		mlines.Line2D([],[],color='blue', label=f'{range_percentage_i[1]} %')]
		plt.show(block=False)


	######################################################
	######################################################
	######################################################


	### PAPER image: spectra sumer and hrts together, and identified lines

	x_lims = [max(min(v_hrtsa_cropNeVIII), min(v_sumer_cropNeVIII)),      min(max(v_hrtsa_cropNeVIII), max(v_sumer_cropNeVIII))]
	if show_plots_iterations=='yes':
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
		ax.set_title(f'{range_percentage_i[0]} - {range_percentage_i[1]} %', fontsize=title_size)
		#ax.set_title(f'SUMER spectrum of the CH uncorrected and corrected with HRTS', fontsize=title_size)
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
		if save_paper_images == 'yes':
			fig_name = 'contours_EIT__spectra_sumer_and_hrts_together_and_line_identification'
			plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
		plt.show(block=False)

	######################################################
	# Fit uncorrected Ne VIII line
	"""
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
	ax[0].set_title(f'{range_percentage_i[0]} - {range_percentage_i[1]} %', fontsize=title_size)
	#ax[0].set_title(f'SUMER spectrum uncorrected, multigaussian fit', fontsize=title_size)
	#ax[0].set_title(f'SUMER {range_percentage_i}%, corrected', fontsize=title_size)
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
		fig_name = 'contours_EIT__spectrum_multigaussian_fit_uncorrected'
		plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
	plt.show(block=False)
	"""
	
	"""
	######################################################
	# Fit corrected Ne VIII line, QS-A

	x_corrected_qra = lam_sumer_cropNeVIII
	y_corrected_qra = rad_sumer_cropNeVIII_corrected_qra
	y_unc_corrected_qra = erad_sumer_cropNeVIII_corrected_qra


	# Perform the fit
	popt_qra, pcov_qra = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), y_corrected_qra, p0=init_parameters_corrected_qra, sigma=y_unc_corrected_qra, absolute_sigma=True) #popt_qra are the optimized parameters. pcov_qra is the covariance matrix of the parameters. 
	perr_qra = np.sqrt(np.diag(pcov_qra)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


	# fitted curve
	x_fit_corrected_qra = np.linspace(min(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0)), max(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0)), 300)
	y_fit_corrected_qra = multigaussian_function_for_curvefit(x_fit_corrected_qra, *popt_qra)


	# Residuals
	y_residuals_qra = y_corrected_qra - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), *popt_qra)
	y_unc_fit_length_corrected_qra = multi_gaussian_function_uncertainties(B=popt_qra, B_unc=perr_qra, x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0))))
	y_unc_residuals_qra = np.sqrt(y_unc_corrected_qra**2 + y_unc_fit_length_corrected_qra**2)
	
	
	if show_plots_iterations=='yes':
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
		ax[0].set_title(f'{range_percentage_i[0]} - {range_percentage_i[1]} %', fontsize=title_size)
		#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-A, multigaussian fit', fontsize=title_size)
		#ax[0].set_title(f'SUMER {range_percentage_i}%, corrected', fontsize=title_size)
		ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=axislabel_size)
		ax[0].legend(fontsize=legend_size)
		ax[0].set_yscale('linear')
		ax[1].errorbar(x=vkms_doppler(lamb=x_corrected_qra, lamb_0=lam_0), y=y_residuals_qra, yerr=y_unc_residuals_qra, color='black', marker='.')
		ax[1].set_xlabel('Doppler shift (km/s)', fontsize=axislabel_size)
		ax[1].set_ylabel('Residuals', fontsize=axislabel_size)
		ax[0].set_xlim(x_lims_fits)
		ax[1].set_xlim(x_lims_fits)
		#plt.tight_layout()
		plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.0)
		if save_paper_images == 'yes':
			fig_name = 'contours_EIT__spectrum_multigaussian_fit_corrected_qra'
			plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
		plt.show(block=False)
	
	##########################
	# 
	
	# Weak component
	idx = 0
	centroid = popt_qra[3*idx+2]
	unc_centroid = perr_qra[3*idx+2]
	intensity = popt_qra[3*idx+1]
	unc_intensity = perr_qra[3*idx+1]
	fwhm = popt_qra[3*idx+3]
	unc_fwhm = perr_qra[3*idx+3]
	
	area = gaussian_area(peak_=intensity, fwhm_=fwhm)
	unc_area = unc_gaussian_area(peak_=intensity, fwhm_=fwhm, peak_unc_=unc_intensity, fwhm_unc_=unc_fwhm)
	
	centroid_weak_component_qra.append(centroid)
	unc_centroid_weak_component_qra.append(unc_centroid)
	intensity_weak_component_qra.append(intensity)
	unc_intensity_weak_component_qra.append(unc_intensity)
	fwhm_weak_component_qra.append(fwhm)
	area_weak_component_qra.append(area)
	unc_area_weak_component_qra.append(unc_area)
	
	
	
	# Main component
	idx = 1
	centroid = popt_qra[3*idx+2]
	unc_centroid = perr_qra[3*idx+2]
	intensity = popt_qra[3*idx+1]
	unc_intensity = perr_qra[3*idx+1]
	fwhm = popt_qra[3*idx+3]
	unc_fwhm = perr_qra[3*idx+3]
	
	area = gaussian_area(peak_=intensity, fwhm_=fwhm)
	unc_area = unc_gaussian_area(peak_=intensity, fwhm_=fwhm, peak_unc_=unc_intensity, fwhm_unc_=unc_fwhm)
	
	centroid_main_component_qra.append(centroid)
	unc_centroid_main_component_qra.append(unc_centroid)
	intensity_main_component_qra.append(intensity)
	unc_intensity_main_component_qra.append(unc_intensity)
	fwhm_main_component_qra.append(fwhm)
	area_main_component_qra.append(area)
	unc_area_main_component_qra.append(unc_area)

	######################################################
	# Fit corrected Ne VIII line, QS-B

	x_corrected_qrb = lam_sumer_cropNeVIII
	y_corrected_qrb = rad_sumer_cropNeVIII_corrected_qrb
	y_unc_corrected_qrb = erad_sumer_cropNeVIII_corrected_qrb


	# Perform the fit
	popt_qrb, pcov_qrb = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), y_corrected_qrb, p0=init_parameters_corrected_qrb, sigma=y_unc_corrected_qrb, absolute_sigma=True) #popt_qrb are the optimized parameters. pcov_qrb is the covariance matrix of the parameters. 
	perr_qrb = np.sqrt(np.diag(pcov_qrb)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


	# fitted curve
	x_fit_corrected_qrb = np.linspace(min(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0)), max(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0)), 300)
	y_fit_corrected_qrb = multigaussian_function_for_curvefit(x_fit_corrected_qrb, *popt_qrb)


	# Residuals
	y_residuals_qrb = y_corrected_qrb - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), *popt_qrb)
	y_unc_fit_length_corrected_qrb = multi_gaussian_function_uncertainties(B=popt_qrb, B_unc=perr_qrb, x=vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected_qrb, lamb_0=lam_0))))
	y_unc_residuals_qrb = np.sqrt(y_unc_corrected_qrb**2 + y_unc_fit_length_corrected_qrb**2)
	
	
	if show_plots_iterations=='yes':
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
		ax[0].set_title(f'{range_percentage_i[0]} - {range_percentage_i[1]} %', fontsize=title_size)
		#ax[0].set_title(f'SUMER spectrum corrected with HRTS QS-B, multigaussian fit', fontsize=title_size)
		#ax[0].set_title(f'SUMER {range_percentage_i}%, corrected', fontsize=title_size)
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
		if save_paper_images == 'yes':
			fig_name = 'contours_EIT__spectrum_multigaussian_fit_corrected_qrb'
			plt.savefig(folder_name+'/'+fig_name+'.png', dpi=save_dpi, bbox_inches='tight')  # Save as PNG (high-res)
		plt.show(block=False)
	
	##########################
	# 
	
	# Weak component
	idx = 0
	centroid = popt_qrb[3*idx+2]
	unc_centroid = perr_qrb[3*idx+2]
	intensity = popt_qrb[3*idx+1]
	unc_intensity = perr_qrb[3*idx+1]
	fwhm = popt_qrb[3*idx+3]
	unc_fwhm = perr_qrb[3*idx+3]
	
	area = gaussian_area(peak_=intensity, fwhm_=fwhm)
	unc_area = unc_gaussian_area(peak_=intensity, fwhm_=fwhm, peak_unc_=unc_intensity, fwhm_unc_=unc_fwhm)
	
	centroid_weak_component_qrb.append(centroid)
	unc_centroid_weak_component_qrb.append(unc_centroid)
	intensity_weak_component_qrb.append(intensity)
	unc_intensity_weak_component_qrb.append(unc_intensity)
	fwhm_weak_component_qrb.append(fwhm)
	area_weak_component_qrb.append(area)
	unc_area_weak_component_qrb.append(unc_area)
	
	
	# Main component
	idx = 1
	centroid = popt_qrb[3*idx+2]
	unc_centroid = perr_qrb[3*idx+2]
	intensity = popt_qrb[3*idx+1]
	unc_intensity = perr_qrb[3*idx+1]
	fwhm = popt_qrb[3*idx+3]
	unc_fwhm = perr_qrb[3*idx+3]
	
	area = gaussian_area(peak_=intensity, fwhm_=fwhm)
	unc_area = unc_gaussian_area(peak_=intensity, fwhm_=fwhm, peak_unc_=unc_intensity, fwhm_unc_=unc_fwhm)
	
	centroid_main_component_qrb.append(centroid)
	unc_centroid_main_component_qrb.append(unc_centroid)
	intensity_main_component_qrb.append(intensity)
	unc_intensity_main_component_qrb.append(unc_intensity)
	fwhm_main_component_qrb.append(fwhm)
	area_main_component_qrb.append(area)
	unc_area_main_component_qrb.append(unc_area)
	"""
    
	######################################################
	
	# Scaling factor and Chi2red
	## QS-A
	scaling_factor_list_qsa.append(fsh_qra['scaling_factor'])
	scaling_factor_unc_list_qsa.append(fsh_qra['scaling_factor_unc'])
	chi2_red_list_qsa.append(fsh_qra['chi2_red_sf'])
	## QS-B
	scaling_factor_list_qsb.append(fsh_qrb['scaling_factor'])
	scaling_factor_unc_list_qsb.append(fsh_qrb['scaling_factor_unc'])
	chi2_red_list_qsb.append(fsh_qrb['chi2_red_sf'])
	
	######################################################
	######################################################
	######################################################
	



##############################
# Dopplershift vs intensity

"""
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.set_title('Dopplershift of Ne VIII as a function of the radiance', fontsize=18)
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
#ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Radiance (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
#ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
#ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax.axvline(x=bound_CH, color='brown', linewidth=1.2, linestyle=':', label='CH boundary') 
ax.legend()
plt.show(block=False)


##############################
# Weak components intensity vs intensity

#Ratio: intensity weak component / main component
Num = np.array(intensity_weak_component_qra)
Den = np.array(intensity_main_component_qra)
Num_unc = np.array(unc_intensity_weak_component_qra)
Den_unc = np.array(unc_intensity_main_component_qra)
ratio_intensity_weak_main_qra = Num / Den
unc_ratio_intensity_weak_main_qra = division_unc(Num=Num, Den=Den, Num_unc=Num_unc, Den_unc=Den_unc)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.set_title('Intensity of the weak and main components. Corrected with QS-A.', fontsize=18)
#ax.set_title(f'', fontsize=18) 
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=intensity_weak_component_qra, yerr=unc_intensity_weak_component_qra, color='red', linewidth=0., elinewidth=1.0, marker='^', label='Weak component')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=intensity_main_component_qra, yerr=unc_intensity_main_component_qra, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='Main component')
ax2 = ax.twinx() 
ax2.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=ratio_intensity_weak_main_qra, yerr=unc_ratio_intensity_weak_main_qra, color='black', linewidth=0., elinewidth=1.0, marker='.', label='Ratio weak/main')
ax2.set_ylabel('Ratio radiances weak/main', color='black', fontsize=16)
ax.set_xlabel(r'Integrated radiance (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel(r'Radiance (W/sr/m$^2$''/\u212B)', color='black', fontsize=16)
#ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
#ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax.axvline(x=bound_CH, color='brown', linewidth=1.2, linestyle=':', label='CH boundary') 
ax.legend()
ax2.legend()
plt.show(block=False)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.set_title('Centroid of the weak and main components. Corrected with QS-A.', fontsize=18)
#ax.set_title(f'', fontsize=18) 
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=centroid_weak_component_qra, yerr=unc_centroid_weak_component_qra, color='red', linewidth=0., elinewidth=1.0, marker='^', label='Weak component')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=centroid_main_component_qra, yerr=unc_centroid_main_component_qra, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='Main component')
ax.set_xlabel(r'Integrated radiance (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel(r'Doppler shift (km/s)', color='black', fontsize=16)
#ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
#ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax.axvline(x=bound_CH, color='brown', linewidth=1.2, linestyle=':', label='CH boundary') 
ax.legend()
ax2.legend()
plt.show(block=False)

##############################
# Area


#Ratio: area weak component / main component
Num = np.array(area_weak_component_qra)
Den = np.array(area_main_component_qra)
Num_unc = np.array(unc_area_weak_component_qra)
Den_unc = np.array(unc_area_main_component_qra)
ratio_area_weak_main_qra = Num / Den
unc_ratio_area_weak_main_qra = division_unc(Num=Num, Den=Den, Num_unc=Num_unc, Den_unc=Den_unc)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.set_title('Area of the weak and main components. Corrected with QS-A.', fontsize=18)
#ax.set_title(f'', fontsize=18) 
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=area_weak_component_qra, yerr=unc_area_weak_component_qra, color='red', linewidth=0., elinewidth=1.0, marker='^', label='Weak component')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=area_main_component_qra, yerr=unc_area_main_component_qra, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='Main component')
ax2 = ax.twinx() 
ax2.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=ratio_area_weak_main_qra, yerr=unc_ratio_area_weak_main_qra, color='black', linewidth=0., elinewidth=1.0, marker='.', label='Ratio weak/main')
ax2.set_ylabel('Ratio integrated radiances weak/main', color='black', fontsize=16)
ax.set_xlabel(r'Integrated radiance (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel(r'Integrated radiance (W/sr/m$^2$)', color='black', fontsize=16)
#ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
#ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax.axvline(x=bound_CH, color='brown', linewidth=1.2, linestyle=':', label='CH boundary') 
ax.legend()
ax2.legend()
plt.show(block=False)
"""

##############################
# 

# Scaling factor
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.set_title(r'Scaling factor', fontsize=18)
#ax.set_title(f'', fontsize=18) 
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=scaling_factor_list_qsa, yerr=scaling_factor_unc_list_qsa, color='blue', linewidth=0.5, elinewidth=1.0, marker='.', label='Scaling factor QS-A')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=scaling_factor_list_qsb, yerr=scaling_factor_unc_list_qsb, color='red', linewidth=0.5, elinewidth=1.0, marker='.', label='Scaling factor QS-A')
ax.set_xlabel(r'Integrated radiance (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel(r'Scaling factor SUMER/HRTS', color='black', fontsize=16)
#ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
#ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
#ax.axvline(x=bound_CH, color='brown', linewidth=1.2, linestyle=':', label='CH boundary') 
ax.legend()
plt.show(block=False)



# Chi2red of the scaling factor
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.set_title(r'$\chi^2_{\rm red}$ of the scaling factor', fontsize=18)
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=chi2_red_list_qsa, color='blue', linewidth=0.5, elinewidth=1.0, marker='^', label=r'$\chi^2_{\rm red}$ QS-A')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=chi2_red_list_qsb, color='red', linewidth=0.5, elinewidth=1.0, marker='^', label=r'$\chi^2_{\rm red}$ QS-B')
ax.set_ylabel(r'$\chi^2_{\rm red}$', color='black', fontsize=16)
ax.set_xlabel(r'Integrated radiance (W/sr/m$^2$)', color='black', fontsize=16)
ax.axhline(y=1., linestyle=':', linewidth=1., color='gray')
ax.legend()
plt.show(block=False)



"""

    
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
    ax.errorbar(x=lam_sumer_cropNeVIII, y=rad_sumer_cropNeVIII_uncorrected, yerr=erad_sumer_cropNeVIII_uncorrected, color='blue', linewidth=1., label='SUMER data')
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

"""
percentage_CH = 10.
_, bound_CH = get_bounds(intensitymap_croplat=intensity_map_croplat, range_percentage=[0., percentage_CH], threshold_value_type='max')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.set_title('Dopplershift of Ne VIII as a function of the radiance', fontsize=18)
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
#ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Radiance (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.axvline(x=bound_CH, color='brown', linewidth=1.2, linestyle=':', label='CH boundary') 
ax.legend()
plt.show(block=False)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.set_title('Dopplershift of Ne VIII as a function of the radiance', fontsize=18)
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
#ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
#ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Radiance (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.axvline(x=bound_CH, color='brown', linewidth=1.2, linestyle=':', label='CH boundary') 
ax.legend()
plt.show(block=False)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.set_title('Dopplershift of Ne VIII as a function of the percentage of the maximum radiance', fontsize=18)
ax.errorbar(x=percentage_list, xerr=percentage_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax.errorbar(x=percentage_list, xerr=percentage_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax.errorbar(x=percentage_list, xerr=percentage_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
#ax.errorbar(x=percentage_list, xerr=percentage_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Percentage of the brightest pixel', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.axvline(x=percentage_CH, color='brown', linewidth=1.2, linestyle=':', label='CH boundary')
ax.legend()
plt.show(block=False)

"""



"""

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




