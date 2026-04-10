#  Inputs
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', or 'cold_line'
fwhm_conv = 1.95*0.04215 #Angstrom
N_pixels_bin = 8000

intensity_CH_boundary = 132.58588 #132.58588 is the 4% of the maximum intensity of EIT array
percentage_CH_boundary = 4.


# Wavelength ranges to crop spectra
wavelength_range_to_average = [1531.1147, 1551.7688]
wavelength_range_to_analyze_NeVIII = [1540.2, 1541.4]

## Ranges of wavelength
wavelength_range_scalefactor_left = [1537.7, 1539.5] #nm
wavelength_range_scalefactor_right = [1542., 1544.] #nm

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


# Import SUMER data interpolated (wavelength calibrated)
data_interpolated_loaded = np.load('../data/data_modified/wcal4__spectral_image_list_intepolated_and_wavelength.npz', allow_pickle=True)

# wavelength from the interpolated data
lam_sumer_full = data_interpolated_loaded['reference_wavelength'] 
elam_sumer_full = data_interpolated_loaded['unc_reference_wavelength'] 
row_reference = int(data_interpolated_loaded['row_reference'])  


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

bound_i = np.min(intensity_map_croplat)-0.01
y_intensity_correction_qra, y_intensity_correction_qrb, y_intensity_correction_qrl = [],[],[]
yerr_intensity_correction_qra, yerr_intensity_correction_qrb, yerr_intensity_correction_qrl = [],[],[]
while bound_i<np.max(intensity_map_croplat):
    lower_bound_i = bound_i
    print(bound_i, ',', np.max(intensity_map_croplat))
    rowscols_inside_range_i = pixels_higher_intensity_than_a_value(arr_2d=intensity_map_croplat, bound_=bound_i, N_pixels=N_pixels_bin)
    print(len(rowscols_inside_range_i))
    row_high, col_high = rowscols_inside_range_i[-1] #address of the pixel with maximum intensity of this bin
    bound_i = intensity_map_croplat[row_high, col_high] #maximum intensity of this bin
    upper_bound_i = bound_i
    bound_mean_list.append((upper_bound_i+lower_bound_i)/2.)
    bound_unc_list.append((upper_bound_i-lower_bound_i)/2.)
    
    
    
    # convert the list of pairs [row, column] into 2 lists of rows and columns (for the scatterplot)
    y_row_list_plot, x_col_list_plot = convert_list_of_pairs_to_2_lists(list_of_pairs=rowscols_inside_range_i)
    
    
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
    # 6) Calculate the correction of intensity made by HRTS
    
    wavelength_range_intensity_map = [1540.45, 1541.2] #Angstroem
    wavelength_range_intensity_map_bckg = [1539.8, 1540.2] #Angstroem
    
    # pixel scale
    w = pixelscale_list[row_reference]
    w_unc = pixelscale_unc_list[row_reference]
    
    
    ## Spectral radiance of the line uncorrected
    lam_sumer_cropint, idx_int = crop_range(list_to_crop=fsha['lam_sumer_cropNeVIII'], range_values=wavelength_range_intensity_map) #crop in the integration range
    Iu_i = fsha['rad_sumer_cropNeVIII'][idx_int[0]:idx_int[1]+1] #1d-array
    Iu_unc_i = fsha['erad_sumer_cropNeVIII'][idx_int[0]:idx_int[1]+1] #1d-array (uncertainties)
    ## Integrate the line
    Iu = w * np.sum(Iu_i) #continuum average
    Iu_unc_T1 = w_unc * np.sum(Iu_i)
    Iu_unc_T2 = np.sum((w*Iu_unc_i)**2)
    Iu_unc = np.sqrt( Iu_unc_T1**2 + Iu_unc_T2**2)
    
    
    ## Spectral radiance of the line corrected with HRTS
    lam_sumer_cropint, idx_int = crop_range(list_to_crop=fsha['lam_sumer_cropNeVIII'], range_values=wavelength_range_intensity_map) #crop in the integration range
    Ia_i = fsha['rad_sumer_cropNeVIII_corrected'][idx_int[0]:idx_int[1]+1] #1d-array
    Ia_unc_i = fsha['erad_sumer_cropNeVIII_corrected'][idx_int[0]:idx_int[1]+1] #1d-array (uncertainties)
    ## Integrate the line
    Ia = w * np.sum(Ia_i) #continuum average
    Ia_unc_T1 = w_unc * np.sum(Ia_i)
    Ia_unc_T2 = np.sum((w*Ia_unc_i)**2)
    Ia_unc = np.sqrt( Ia_unc_T1**2 + Ia_unc_T2**2)
    
    
    ## Spectral radiance of the line
    lam_sumer_cropint, idx_int = crop_range(list_to_crop=fshb['lam_sumer_cropNeVIII'], range_values=wavelength_range_intensity_map) #crop in the integration range
    Ib_i = fshb['rad_sumer_cropNeVIII_corrected'][idx_int[0]:idx_int[1]+1] #1d-array
    Ib_unc_i = fshb['erad_sumer_cropNeVIII_corrected'][idx_int[0]:idx_int[1]+1] #1d-array (uncertainties)
    ## Integrate the line
    Ib = w * np.sum(Ib_i) #continuum average
    Ib_unc_T1 = w_unc * np.sum(Ib_i)
    Ib_unc_T2 = np.sum((w*Ib_unc_i)**2)
    Ib_unc = np.sqrt( Ib_unc_T1**2 + Ib_unc_T2**2)
    
    
    ## Spectral radiance of the line
    lam_sumer_cropint, idx_int = crop_range(list_to_crop=fshl['lam_sumer_cropNeVIII'], range_values=wavelength_range_intensity_map) #crop in the integration range
    Il_i = fshl['rad_sumer_cropNeVIII_corrected'][idx_int[0]:idx_int[1]+1] #1d-array
    Il_unc_i = fshl['erad_sumer_cropNeVIII_corrected'][idx_int[0]:idx_int[1]+1] #1d-array (uncertainties)
    ## Integrate the line
    Il = w * np.sum(Il_i) #continuum average
    Il_unc_T1 = w_unc * np.sum(Il_i)
    Il_unc_T2 = np.sum((w*Il_unc_i)**2)
    Il_unc = np.sqrt( Il_unc_T1**2 + Il_unc_T2**2)
    
    
    # Calculate the percentage of intensity removed by the blends
    ## Percentage
    percentage_corrected_qra = (1-Ia/Iu)*100.
    percentage_corrected_qrb = (1-Ib/Iu)*100.
    percentage_corrected_qrl = (1-Il/Iu)*100.
    y_intensity_correction_qra.append(percentage_corrected_qra) 
    y_intensity_correction_qrb.append(percentage_corrected_qrb) 
    y_intensity_correction_qrl.append(percentage_corrected_qrl) 
    ## Uncertainty of the percentage
    unc_percentage_corrected_qra = 100.*np.sqrt((-Ia_unc/Iu)**2 + (Ia*Iu_unc/(Iu**2))**2)
    unc_percentage_corrected_qrb = 100.*np.sqrt((-Ib_unc/Iu)**2 + (Ib*Iu_unc/(Iu**2))**2)
    unc_percentage_corrected_qrl = 100.*np.sqrt((-Il_unc/Iu)**2 + (Il*Iu_unc/(Iu**2))**2)
    yerr_intensity_correction_qra.append(unc_percentage_corrected_qra) 
    yerr_intensity_correction_qrb.append(unc_percentage_corrected_qrb) 
    yerr_intensity_correction_qrl.append(unc_percentage_corrected_qrl) 

    ######################################################
    # 7) calculate peak
    
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
        contour_upper = ax.contour(intensity_map_croplat[::-1], levels=[bound_i], colors='blue', linewidths=1, extent=extent_sumer_px_contours)
        legend_elements = [mlines.Line2D([],[],color='blue', label=f'{bound_i}')]
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


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Spectral radiance peak (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.legend()
plt.show(block=False)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
ax.errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax.set_title(f'', fontsize=18) 
ax.set_xlabel(r'Spectral radiance peak (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}''\u212B')#, label=label_i) 
ax.axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax.set_xscale('log')
ax.legend()
plt.show(block=False)




#Teriaca's comment for the paper: Percentage of intensity that is taken by HRTS. This is a general discussion about why blends are important bla bla bla. It is a general plot. And then I go to the details (maps and single things). 

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1.5]}, sharex=True)
ax[0].errorbar(x=bound_mean_list, xerr=bound_unc_list, y=y_intensity_correction_qra, yerr=yerr_intensity_correction_qra, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax[0].errorbar(x=bound_mean_list, xerr=bound_unc_list, y=y_intensity_correction_qrb, yerr=yerr_intensity_correction_qrb, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
#ax[0].errorbar(x=bound_mean_list, xerr=bound_unc_list, y=y_intensity_correction_qrl, yerr=yerr_intensity_correction_qrl, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax[0].set_ylabel('Intensity correction\n HRTS/SUMER (%)', color='black', fontsize=16)
ax[1].errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax[1].errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax[1].errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
#ax[1].errorbar(x=bound_mean_list, xerr=bound_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax[1].set_title(f'', fontsize=18) 
ax[1].set_xlabel(r'Spectral radiance peak (W/sr/m$^2$)', color='black', fontsize=16)
ax[1].set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax[1].axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}'' S\u212B')#, label=label_i) 
ax[1].axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].axvline(x=intensity_CH_boundary, color='black', linestyle=':', label='CH boundary')
ax[1].axvline(x=intensity_CH_boundary, color='black', linestyle=':')
ax[0].legend()
ax[1].legend()
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.02)
plt.show(block=False)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1.5]}, sharex=True)
ax[0].errorbar(x=percentage_list, xerr=percentage_unc_list, y=y_intensity_correction_qra, yerr=yerr_intensity_correction_qra, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax[0].errorbar(x=percentage_list, xerr=percentage_unc_list, y=y_intensity_correction_qrb, yerr=yerr_intensity_correction_qrb, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
#ax[0].errorbar(x=percentage_list, xerr=percentage_unc_list, y=y_intensity_correction_qrl, yerr=yerr_intensity_correction_qrl, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax[0].set_ylabel('Intensity correction\n HRTS/SUMER (%)', color='black', fontsize=16)
ax[1].errorbar(x=percentage_list, xerr=percentage_unc_list, y=v_peak_uncorrected_list, yerr=ev_peak_uncorrected_list, color='red', linewidth=0., elinewidth=1.0, marker='^', label='SUMER uncorrected')
ax[1].errorbar(x=percentage_list, xerr=percentage_unc_list, y=v_peak_corrected_qra_list, yerr=ev_peak_corrected_qra_list, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-A')
ax[1].errorbar(x=percentage_list, xerr=percentage_unc_list, y=v_peak_corrected_qrb_list, yerr=ev_peak_corrected_qrb_list, color='green', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-B')
#ax[1].errorbar(x=percentage_list, xerr=percentage_unc_list, y=v_peak_corrected_qrl_list, yerr=ev_peak_corrected_qrl_list, color='cyan', linewidth=0., elinewidth=1.0, marker='.', label='SUMER corrected, QR-L')
ax[1].set_title(f'', fontsize=18) 
ax[1].set_xlabel(r'Percentage of maximum intensity', color='black', fontsize=16)
ax[1].set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax[1].axhline(y=0., color='black', linewidth=1.2, linestyle='--', label=f'Rest wavelength {lam_0}'' \u212B')#, label=label_i) 
ax[1].axhspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].axvline(x=percentage_CH_boundary, color='black', linestyle=':', label='CH boundary')
ax[1].axvline(x=percentage_CH_boundary, color='black', linestyle=':')
ax[0].legend()
ax[1].legend()
#plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.9, wspace=0., hspace=0.02)
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




