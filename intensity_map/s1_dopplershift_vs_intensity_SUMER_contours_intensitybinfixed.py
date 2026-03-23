#  Inputs
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'
fwhm_conv = 1.95*0.04215 #Angstrom
#range_percentage_list = [[0.,10.], [10.,20.], [20.,30.], [30.,40.], [40.,50.], [50.,60.], [60.,70.], [70.,80.], [80.,90.], [90.,100.]]
#range_percentage_list = [[0., 5.], [5., 10.], [10., 15.], [15., 20.],[20., 25.], [25., 30.], [30., 35.], [35., 40.],[40., 45.], [45., 50.], [50., 55.], [55., 60.],[60., 65.], [65., 70.], [70., 75.], [75., 80.],[80., 85.], [85., 90.], [90., 95.], [95., 100.]]
#range_percentage_list = [[0., 2.], [2., 4.], [4., 6.], [6., 8.], [8., 10.],[10., 12.], [12., 14.], [14., 16.], [16., 18.], [18., 20.],[20., 22.], [22., 24.], [24., 26.], [26., 28.], [28., 30.],[30., 32.], [32., 34.], [34., 36.], [36., 38.], [38., 40.],[40., 42.], [42., 44.], [44., 46.], [46., 48.], [48., 50.],[50., 52.], [52., 54.], [54., 56.], [56., 58.], [58., 60.],[60., 62.], [62., 64.], [64., 66.], [66., 68.], [68., 70.],[70., 72.], [72., 74.], [74., 76.], [76., 78.], [78., 80.],[80., 82.], [82., 84.], [84., 86.], [86., 88.], [88., 90.],[90., 92.], [92., 94.], [94., 96.], [96., 98.], [98., 100.]]
#range_percentage_list = [[0.,3.], [3.,6.], [6.,10.], [10.,20.], [20.,30.], [30.,40.], [40.,60.], [60.,100.]]

range_percentage_list = [[0.,5.], [5.,10.], [10.,20.], [20.,30.], [30.,40.], [40.,50.], [50.,60.], [60.,80.], [80.,100.]]
range_percentage_list = [[0.,5.], [5.,10.], [10.,20.], [20.,30.], [30.,40.], [40.,50.], [50.,70.], [70.,100.]]
range_percentage_list = [[2.,4.], [4.,6.],[6.,8.], [8.,10.], [10.,20.], [20.,30.], [30.,40.], [40.,50.], [50.,70.], [70.,100.]]


# Wavelength ranges to crop spectra
wavelength_range_to_average = [1531.1147, 1551.7688]
wavelength_range_to_analyze_NeVIII = [1540.2, 1541.4]

## Ranges of wavelength
wavelength_range_scalefactor_left = [1537.7, 1539.5] #nm
wavelength_range_scalefactor_right = [1542., 1544.] #nm

threshold_value_type_sumer = 'max' #'max', 'min', 'mean', 'median'

show_plots_correction = 'no'
show_plots = 'yes'

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
from utils.solar_rotation_variables import *
from utils.aux_functions import *
from utils.general_variables import *
from utils.NeVIII_rest_wavelength import *
from scale_hrts import *

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

# uncertainty of the rest wavelength in km/s
v_unc_0 = vkms_doppler_unc(lamb=lam_0, lamb_unc=lam_unc_0, lamb_0=lam_0, lamb_0_unc=lam_unc_0) 

######################################################
######################################################
######################################################

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
lam_peak_corrected_qrl_list = []
elam_peak_corrected_qrl_list = []
v_peak_corrected_qrl_list = []
ev_peak_corrected_qrl_list = []
rad_peak_corrected_qrl_list = []
###########################
bound_mean_list, bound_unc_list = [],[]

percentage_list, percentage_unc_list = [],[]
for range_percentage_i in range_percentage_list:
    """
    lower_bound, upper_bound = get_bounds(intensitymap_croplat=intensity_map_croplat, range_percentage=range_percentage_i, threshold_value_type=threshold_value_type_sumer)
    
    # rows and columns inside the intensity bin  in EIT ,map
    rowscols_inside_range_i = np.argwhere((intensity_map_croplat>=lower_bound) & (intensity_map_croplat<=upper_bound))
    y_row_list_plot = rowscols_inside_range_i[:,0] # convert the list of pairs [row, column] into 2 lists of rows and columns (for the scatterplot)
    x_col_list_plot = rowscols_inside_range_i[:,1]
    """
    
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

    # crop near Ne VIII
    lam_sumer_cropNeVIII, idx_sumer_crop_ = crop_range(list_to_crop=lam_sumer_av, range_values=wavelength_range_to_analyze_NeVIII)
    elam_sumer_cropNeVIII = elam_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    rad_sumer_cropNeVIII_uncorrected = rad_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    erad_sumer_cropNeVIII_uncorrected = erad_sumer_av[idx_sumer_crop_[0]:idx_sumer_crop_[1]+1]
    
    
    hrts_qr='a'
    fsha = fun_scale_hrts(hrts_qr=hrts_qr, lamb_0=lam_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_conv, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction)
    lam_sumer_cropNeVIII, rad_sumer_cropNeVIII, erad_sumer_cropNeVIII, rad_sumer_cropNeVIII_corrected_qra, erad_sumer_cropNeVIII_corrected_qra = fsha['lam_sumer_cropNeVIII'], fsha['rad_sumer_cropNeVIII'], fsha['erad_sumer_cropNeVIII'], fsha['rad_sumer_cropNeVIII_corrected'], fsha['erad_sumer_cropNeVIII_corrected']
    
    hrts_qr='b'
    fshb = fun_scale_hrts(hrts_qr=hrts_qr, lamb_0=lam_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_conv, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction)
    rad_sumer_cropNeVIII_corrected_qrb, erad_sumer_cropNeVIII_corrected_qrb = fshb['rad_sumer_cropNeVIII_corrected'], fshb['erad_sumer_cropNeVIII_corrected']
    
    hrts_qr='l'
    fshl = fun_scale_hrts(hrts_qr=hrts_qr, lamb_0=lam_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_conv, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction)    
    rad_sumer_cropNeVIII_corrected_qrl, erad_sumer_cropNeVIII_corrected_qrl = fshl['rad_sumer_cropNeVIII_corrected'], fshl['erad_sumer_cropNeVIII_corrected']


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
    
    # corrected data QR-L
    mpi_corrected_qrl = find_maximum_by_parabolic_interpolation_adapted(wavelength=lam_sumer_cropNeVIII, radiance=rad_sumer_cropNeVIII_corrected_qrl, radiance_unc=erad_sumer_cropNeVIII_corrected_qrl, show_figure='no')
    mpi_corrected_qrl["v_vertex"] = vkms_doppler(lamb=mpi_corrected_qrl["x_vertex"], lamb_0=lam_0) #convert wavelength to speed
    mpi_corrected_qrl["v_unc_vertex"] = vkms_doppler_unc(lamb=mpi_corrected_qrl["x_vertex"], lamb_unc=mpi_corrected_qrl["x_unc_vertex"], lamb_0=lam_0, lamb_0_unc=lam_unc_0) 


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
    ########################### corrected QR-B')
    lam_peak_corrected_qrl_list.append(mpi_corrected_qrl["x_vertex"])
    elam_peak_corrected_qrl_list.append(mpi_corrected_qrl["x_unc_vertex"])
    v_peak_corrected_qrl_list.append(mpi_corrected_qrl["v_vertex"])
    ev_peak_corrected_qrl_list.append(mpi_corrected_qrl["v_unc_vertex"])
    ###########################
    percentage_list.append(np.mean(range_percentage_i))
    percentage_unc_list.append(np.mean(range_percentage_i) - range_percentage_i[0])
    

    ######################################################
    ######################################################
    ######################################################


    ######################################################
    # Show image of the intensity map with the contours and the pixels inside the contours
    
    if show_plots=='yes':
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
        mlines.Line2D([],[],color='red', label=f'{range_percentage_i[0]} %'),
        mlines.Line2D([],[],color='blue', label=f'{range_percentage_i[1]} %')]
    plt.show(block=False)


    ######################################################
    ######################################################
    ######################################################
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

percentage_CH = 10.
_, bound_CH = get_bounds(intensitymap_croplat=intensity_map_croplat, range_percentage=[0., percentage_CH], threshold_value_type='max')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.set_title('Dopplershift of Ne VIII as a function of the radiance', fontsize=18)
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
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
ax.errorbar(x=percentage_list, xerr=percentage_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Percentage of the brightest pixel', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.axvline(x=percentage_CH, color='brown', linewidth=1.2, linestyle=':', label='CH boundary')
ax.legend()
plt.show(block=False)


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




