

#filename_sumer_spectrum = 'average_profile__0_0__50_0__mean_of_intensitymap_NeVIII.npz'
#filename_sumer_spectrum = 'average_profile__0_0__37_6__mean_of_intensitymap_NeVIII.npz'
filename_sumer_spectrum = 'average_profile__0_0__40_0__mean_of_intensitymap_NeVIII.npz'
percentage_label = 40

fwhm_conv = 1.95*0.04215 #nm

sun_region = 'qqr_l' #HRTS spectrum
#wavelength_range_NeVIII = [1539., 1542.5]
wavelength_range_NeVIII = [1540., 1541.5]
wavelength_range_NeVIII = [1540.1, 1541.46]
wavelength_range_NeVIII = [1540.32, 1541.43]
n_gaussian_to_analyze_corrected = 1 
n_gaussian_to_analyze = 1

## Rest wavelength used
lamb_0 = 1540.856
#lamb_0 = 1540.85
lamb_unc_0 = 0.014 #Angstrom


color_sumer = 'blue'
color_hrts = 'green'
color_sumer_uncorrected = 'red'
color_sumer_corrected = 'blue'

## Ranges of wavelength
wavelength_range_scalefactor_left = [1537.7, 1539.5] #nm
wavelength_range_scalefactor_right = [1542., 1544.] #nm

#wavelength_range_scalefactor_left = [1537.7, 1540.4] #nm
#wavelength_range_scalefactor_right = [1541.22, 1544.] #nm

######################################################
# Initial parameters of the fitting

bckg_fit_uncorrected = 1.5 #HRST not subtracted
init_parameters_uncorrected = [bckg_fit_uncorrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
           2.-bckg_fit_uncorrected, -54., 20.,
           5.-bckg_fit_uncorrected, -10, 45.,
           4.-bckg_fit_uncorrected, 15., 45.,
           2.15-bckg_fit_uncorrected, 97., 30.
           ]

if sun_region=='qqr_a':
    wavelength_range_NeVIII = [1540.32, 1541.43]
    bckg_fit_corrected = -0.3
    init_parameters_corrected = [bckg_fit_corrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
                   0.1-bckg_fit_corrected, -60., 20.,
                   3.-bckg_fit_corrected, 0.0, 50.,
                   1.5-bckg_fit_corrected, 25., 30.
                   ]
               
elif sun_region=='qqr_b':
    wavelength_range_NeVIII = [1540.3, 1541.43]
    bckg_fit_corrected = -0.3
    init_parameters_corrected = [bckg_fit_corrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
                   1.-bckg_fit_corrected, -40., 30.,
                   3.25-bckg_fit_corrected, 0.0, 50.,
                   1.5-bckg_fit_corrected, 33., 40.
                   ]
                   
elif sun_region=='qqr_l':
    bckg_fit_corrected = -0.3
    init_parameters_corrected = [bckg_fit_corrected, #[background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]
                   0.39-bckg_fit_corrected, -75., 20.,
                   0.93-bckg_fit_corrected, -35., 30.,
                   3.-bckg_fit_corrected, 7., 35.,
                   1.-bckg_fit_corrected, 37., 50.
                   ]

######################################################
######################################################
######################################################
# Import packages, functions...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

import sys
import os
sys.path.append(os.path.abspath('..'))
from auxiliar_functions.aux_functions import *
######################################################
######################################################
######################################################
# Import data

######################################################
# SUMER

foldername_sumer_spectrum = '../auxiliar_functions/'
profiles_loaded_dic = np.load(foldername_sumer_spectrum + filename_sumer_spectrum)
lam_sumer = profiles_loaded_dic['lam_sumer_av'] #Angstrom
rad_sumer = profiles_loaded_dic['rad_sumer_av'] #[W/sr/m^2/Angstroem]
erad_sumer = profiles_loaded_dic['erad_sumer_av'] #[W/sr/m^2/Angstroem]
#lam_sumer_avNeVIII = profiles_loaded_dic['lam_sumer_avNeVIII'] #Angstrom
#rad_sumer_avNeVIII = profiles_loaded_dic['rad_sumer_avNeVIII'] #[W/sr/m^2/Angstroem]
#erad_sumer_avNeVIII = profiles_loaded_dic['erad_sumer_avNeVIII'] #[W/sr/m^2/Angstroem]

lam_sumer_idx = np.arange(0, len(lam_sumer))


######################################################
# HRTS

if sun_region=='qqr_a':
	from hrts_spectra.data__qqr_a_xdr import lambda__qqr_a_xdr, radiance__qqr_a_xdr, unc_radiance__qqr_a_xdr
	lam_hrts, rad_hrts = 10.*lambda__qqr_a_xdr, 0.1*radiance__qqr_a_xdr #multiply by 10. and 0.1 to convert nm to Angstrom
elif sun_region=='qqr_b':
	from hrts_spectra.data__qqr_b_xdr import lambda__qqr_b_xdr, radiance__qqr_b_xdr, unc_radiance__qqr_b_xdr
	lam_hrts, rad_hrts = 10.*lambda__qqr_b_xdr, 0.1*radiance__qqr_b_xdr
elif sun_region=='qqr_l':
	from hrts_spectra.data__qqr_l_xdr import lambda__qqr_l_xdr, radiance__qqr_l_xdr, unc_radiance__qqr_l_xdr
	lam_hrts, rad_hrts = 10.*lambda__qqr_l_xdr, 0.1*radiance__qqr_l_xdr
elif sun_region=='qr':
	from hrts_spectra.data__qr_xdr import lambda__qr_xdr, radiance__qr_xdr, unc_radiance__qr_xdr
	lam_hrts, rad_hrts = 10.*lambda__qr_xdr, 0.1*radiance__qr_xdr
else: print("sun_region should be 'qqr_a', 'qqr_b', 'qqr_l', or 'qr'")

lam_hrts
rad_hrts
erad_hrts = np.zeros(len(rad_hrts)) #TODO: we don't know the uncertainties of HRTS


######################################################
######################################################
######################################################
# uncertainty of the rest wavelength in km/s

v_unc_0 = vkms_doppler_unc(lamb=lamb_0, lamb_unc=lamb_unc_0, lamb_0=lamb_0, lamb_0_unc=lamb_unc_0) 
    
######################################################
######################################################
######################################################
# 

def indices_closer_to_range(arr_1d, range_):
    rg0, rg1 = range_

    # Find indices of closest values
    idx0 = (np.abs(arr_1d - rg0)).argmin()
    idx1 = (np.abs(arr_1d - rg1)).argmin()
    return [idx0, idx1]
    
def convolution_FWHM_in_pixels(wavelength, fwhm_wavelength):
    """
    FWHM of the convolution Gaussian profile. This function calculates it in pixels of the "wavelength" array. 
    """
    w = wavelength
    pixel_scale = (w[-1] - w[0]) / (len(w)-1)
    fwhm_px = fwhm_wavelength / pixel_scale
    return fwhm_px


######################################################
# 1) Convolve HRTS spectrum with SUMER instrumental profile (gaussian profile of FWHM fwhm_conv)

from scipy.ndimage import convolve1d

inst_fwhm_HRTSpx = convolution_FWHM_in_pixels(wavelength=lam_hrts, fwhm_wavelength=fwhm_conv) #FWHM in pixels
inst_sigma_HRTSpx = inst_fwhm_HRTSpx/(2*np.sqrt(2*np.log(2))) # convert FWHM to sigma (pixels)

# build kernel
half_width_ = int(5*inst_sigma_HRTSpx) #Using ±5σ captures essentially all the Gaussian
xk = np.arange(-half_width_, half_width_+1)
kernel_ = np.exp(-0.5*(xk/inst_sigma_HRTSpx)**2)
kernel_ /= kernel_.sum() #Properly normalized kernel

# convolve flux
rad_hrts_conv = convolve1d(rad_hrts, kernel_, mode='constant', cval=0)

# propagate variance properly
varrad_hrts_conv = convolve1d(erad_hrts**2, kernel_**2, mode='constant', cval=0)
erad_hrts_conv = np.sqrt(varrad_hrts_conv)

"""
rad_hrts_conv = gaussian_filter1d(rad_hrts, sigma=inst_sigma_HRTSpx) 
varrad_hrts_conv = gaussian_filter1d(erad_hrts**2, sigma=inst_sigma_HRTSpx) # Propagate variance properly
erad_hrts_conv =  = np.sqrt(varrad_hrts_conv) # This is NOT yet correct uncertainty
"""


######################################################
# 2) Create the interpolation function of the entire HRTS spectrum (after convolvolution)

from scipy.interpolate import interp1d

# Radiances
rad_hrts_conv = np.ma.filled(rad_hrts_conv, np.nan) # Convert masked values to NaNs
interp_func_hrts = interp1d(lam_hrts, rad_hrts_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
rad_hrts_conv_SUMERgrid = interp_func_hrts(lam_sumer)

# Uncertainties of the radiance
erad_hrts_conv = np.ma.filled(erad_hrts_conv, np.nan) # Convert masked values to NaNs
interp_func_hrts_err = interp1d(lam_hrts, erad_hrts_conv, kind='linear', bounds_error=False, fill_value=np.nan) 
erad_hrts_conv_SUMERgrid = interp_func_hrts_err(lam_sumer)


######################################################
# 3) Scaling factor SUMER/HRTS. Crop regions at left and right of Ne VIII line to calculate the scaling factor of HRTS


################
# 3.1) Select range(s) of wavelength

# Left
idx_left_sumer_0, idx_left_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_scalefactor_left)
lam_sumer_cropleft = lam_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
rad_sumer_cropleft = rad_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
erad_sumer_cropleft = erad_sumer[idx_left_sumer_0:idx_left_sumer_1+1]
rad_hrts_conv_SUMERgrid_cropleft = rad_hrts_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]
erad_hrts_conv_SUMERgrid_cropleft = erad_hrts_conv_SUMERgrid[idx_left_sumer_0:idx_left_sumer_1+1]

# Right
idx_right_sumer_0, idx_right_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_scalefactor_right)
lam_sumer_cropright = lam_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
rad_sumer_cropright = rad_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
erad_sumer_cropright = erad_sumer[idx_right_sumer_0:idx_right_sumer_1+1]
rad_hrts_conv_SUMERgrid_cropright = rad_hrts_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]
erad_hrts_conv_SUMERgrid_cropright = erad_hrts_conv_SUMERgrid[idx_right_sumer_0:idx_right_sumer_1+1]


# Concatenate left and right
lam_sumer_cropscale  = np.concatenate([lam_sumer_cropleft,  lam_sumer_cropright])
rad_sumer_cropscale  = np.concatenate([rad_sumer_cropleft,  rad_sumer_cropright])
erad_sumer_cropscale = np.concatenate([erad_sumer_cropleft, erad_sumer_cropright])
rad_hrts_conv_SUMERgrid_cropscale  = np.concatenate([rad_hrts_conv_SUMERgrid_cropleft, rad_hrts_conv_SUMERgrid_cropright])
erad_hrts_conv_SUMERgrid_cropscale  = np.concatenate([erad_hrts_conv_SUMERgrid_cropleft, erad_hrts_conv_SUMERgrid_cropright])


# Plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.errorbar(x=lam_sumer, y=rad_sumer, yerr=erad_sumer, color=color_sumer, linewidth=0.6, label='SUMER full spectrum')
ax.errorbar(x=lam_sumer_cropscale, y=rad_sumer_cropscale, yerr=erad_sumer_cropscale, color=color_sumer, linewidth=0, elinewidth=1., marker='.', markersize=3, label='SUMER regions for scaling factor')
ax.errorbar(x=lam_hrts, y=rad_hrts_conv, color=color_hrts, linewidth=0.6, label='HRST convolved') 
ax.errorbar(x=lam_sumer_cropscale, y=rad_hrts_conv_SUMERgrid_cropscale, yerr=erad_hrts_conv_SUMERgrid_cropscale, color=color_hrts, linewidth=0, elinewidth=1., marker='.', markersize=3, label='HRTS convolved, regions for scaling factor') 
ax.set_title('Comparison SUMER and HRTS with wavelength ranges used\n for the calculation of the scaling factor', fontsize=18)
ax.axvspan(wavelength_range_scalefactor_left[0], wavelength_range_scalefactor_left[1], color='grey', alpha=0.15, label='Wavelength ranges')
ax.axvspan(wavelength_range_scalefactor_right[0], wavelength_range_scalefactor_right[1], color='grey', alpha=0.15)
ax.set_xlabel(r'1$^{\rm st}$ order wavelength (''\u212B)', fontsize=15)#'Wavelength (nm)'
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
ax.legend(fontsize=12)
ax.set_yscale('log')
plt.show(block=False)

################
# 3.2) Calculate the factor with a linear fit (and zero intercept). Fit straight line to the radiance of HRTS vs SUMER (y = m*x + 0 (intercept = 0))

from scipy.optimize import curve_fit

def linear_func(x_, m_):
    return m_ * x_   # zero intercept

y_hrts = rad_hrts_conv_SUMERgrid_cropscale
yerr_hrts = erad_hrts_conv_SUMERgrid_cropscale
y_sumer = rad_sumer_cropscale
yerr_sumer = erad_sumer_cropscale
popt_sf, pcov_sf = curve_fit(linear_func, y_hrts, y_sumer, sigma=yerr_sumer, absolute_sigma=True)

scaling_factor_ = popt_sf[0]
scaling_factor_err = np.sqrt(pcov_sf[0, 0])

# Compute reduced chi^2
y_model = linear_func(y_hrts, *popt_sf) # Model prediction 
chi2_sf = np.sum(((y_sumer - y_model) / yerr_sumer) ** 2) # Chi-square
dof_sf = len(y_sumer) - len(popt_sf) # Degrees of freedom
chi2_red_sf = chi2_sf / dof_sf # Reduced chi-square

print("Scaling factor SUMER/HRTS:", scaling_factor_)
print("Scaling factor error     :", scaling_factor_err)
print("Reduced chi-square       :", chi2_red_sf)


# Generate fitted line for plotting
xfit_sf = np.linspace(min(y_hrts), max(y_hrts), 1000)
yfit_sf = scaling_factor_ * xfit_sf #or linear_func(x_=xfit_sf, m_=scaling_factor_), they're the same



# Show radiances of SUMER and HRTS
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
ax.errorbar(x=y_hrts, y=y_sumer, yerr=yerr_sumer, color='black', linewidth=0, elinewidth=1., marker='.', markersize=5, label='Data')
ax.plot(xfit_sf, yfit_sf, color='brown', label=f'Linear fit. Slope (scaling factor): {np.round(scaling_factor_,4)}')
ax.set_title(f'Spectral radiances of SUMER vs HRTS, and fitting', fontsize=18) 
ax.set_xlabel(r'Spectral radiance HRTS (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', color=color_hrts, fontsize=16)
ax.set_ylabel(r'Spectral radiance SUMER (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', color=color_sumer, fontsize=16)
# Change tick label colors
ax.tick_params(axis='x', colors=color_hrts)
ax.tick_params(axis='y', colors=color_sumer)
# ---- Make axes square with same limits ----
min_val = min(y_sumer.min(), y_hrts.min())
max_val = max(y_sumer.max(), y_hrts.max())
delta_extremes = 0.05 * (max_val - min_val)
ax.set_xlim(min_val-delta_extremes, max_val+delta_extremes)
ax.set_ylim(min_val-delta_extremes, max_val+delta_extremes)
ax.set_aspect('equal', adjustable='box')
ax.legend(fontsize=10)
plt.show(block=False)


def error_propagation_product(a_, a_err, b_, b_err):
    """
    Error propagartion of c_=a_*b_
    """
    T1 = a_*b_err
    T2 = b_*a_err
    return np.sqrt(T1**2 + T2**2)


# Scale HRTS and crop in range of analysis
## Radiance
rad_hrts_conv_scaled = scaling_factor_ * rad_hrts_conv
rad_hrts_conv_scaled_SUMERgrid = scaling_factor_ * rad_hrts_conv_SUMERgrid
## Uncertainties
erad_hrts_conv_scaled = error_propagation_product(a_=scaling_factor_, a_err=scaling_factor_err, b_=rad_hrts_conv, b_err=erad_hrts_conv)
erad_hrts_conv_scaled_SUMERgrid = error_propagation_product(a_=scaling_factor_, a_err=scaling_factor_err, b_=rad_hrts_conv_SUMERgrid, b_err=erad_hrts_conv_SUMERgrid)

######################################################
# 4) Crop in the range of analysis

# Crop SUMER and HRTS (interpolated to SUMER grid)
idx_NeVIII_sumer_0, idx_NeVIII_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_NeVIII)
lam_sumer_cropNeVIII = lam_sumer[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
rad_sumer_cropNeVIII = rad_sumer[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
erad_sumer_cropNeVIII = erad_sumer[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
rad_hrts_conv_SUMERgrid_cropNeVIII = rad_hrts_conv_SUMERgrid[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
erad_hrts_conv_SUMERgrid_cropNeVIII = erad_hrts_conv_SUMERgrid[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
rad_hrts_conv_scaled_SUMERgrid_cropNeVIII = rad_hrts_conv_scaled_SUMERgrid[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
erad_hrts_conv_scaled_SUMERgrid_cropNeVIII = erad_hrts_conv_scaled_SUMERgrid[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]

# Crop HRTS original (nor interpolated to SUMER grid)
idx_NeVIII_hrts_0, idx_NeVIII_hrts_1 = indices_closer_to_range(arr_1d=lam_hrts, range_=wavelength_range_NeVIII)
lam_hrts_cropNeVIII = lam_hrts[idx_NeVIII_hrts_0:idx_NeVIII_hrts_1+1]
rad_hrts_conv_scaled_cropNeVIII = rad_hrts_conv_scaled[idx_NeVIII_hrts_0:idx_NeVIII_hrts_1+1]
erad_hrts_conv_scaled_cropNeVIII = erad_hrts_conv_scaled[idx_NeVIII_hrts_0:idx_NeVIII_hrts_1+1]


# Plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.errorbar(x=lam_sumer, y=rad_sumer, yerr=erad_sumer, color=color_sumer_uncorrected, linewidth=0.6, label='SUMER full spectrum')
ax.errorbar(x=lam_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linewidth=0, marker='.', markersize=3, label='SUMER regions for analysis (Ne VIII)')
ax.errorbar(x=lam_hrts, y=rad_hrts_conv_scaled, yerr=erad_hrts_conv_scaled, color=color_hrts, linewidth=0.6, label=r'HRST convolved and scaled ($\times$'f' {np.round(scaling_factor_,4)})') 
ax.errorbar(x=lam_sumer_cropNeVIII, y=rad_hrts_conv_scaled_SUMERgrid_cropNeVIII, yerr=erad_hrts_conv_scaled_SUMERgrid_cropNeVIII, color=color_hrts, linewidth=0, elinewidth=1., marker='.', markersize=3, label='HRTS convolved, range of analysis (Ne VIII)') 
ax.set_title(f'HRTS scaled', fontsize=18)
ax.axvspan(wavelength_range_NeVIII[0], wavelength_range_NeVIII[1], color='grey', alpha=0.15, label='Wavelength range around Ne VIII')
ax.set_xlabel(r'1$^{\rm st}$ order wavelength (nm)', fontsize=15)#'Wavelength (nm)'
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
ax.legend(fontsize=12)
ax.set_yscale('log')
plt.show(block=False)


######################################################
# 5) Subtract HRTS

rad_sumer_cropNeVIII_corrected = rad_sumer_cropNeVIII - rad_hrts_conv_scaled_SUMERgrid_cropNeVIII
erad_sumer_cropNeVIII_corrected = np.sqrt(erad_sumer_cropNeVIII**2 + erad_hrts_conv_scaled_SUMERgrid_cropNeVIII**2) #TODO: we don't have uncertainties in the HRTS spectrum


x_corrected = lam_sumer_cropNeVIII
y_corrected = rad_sumer_cropNeVIII_corrected
y_unc_corrected = erad_sumer_cropNeVIII_corrected


# Plot: Comparison SUMER corrected and uncorrected
fig, ax = plt.subplots(figsize=(12, 5))
#ax.errorbar(x=vkms_doppler(lamb=lam_crop, lamb_0=lamb_0), y=rad_crop, yerr=erad_crop, color='black', linewidth=0.6, label='SUMER box') #Real spectrum (SUMER) 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER lowest {percentage_label}%, not corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII_corrected, yerr=erad_sumer_cropNeVIII_corrected, color=color_sumer_corrected, linestyle='-', linewidth=2., label=f'SUMER lowest {percentage_label}%, corrected') 
ax.errorbar(x=vkms_doppler(lamb=lam_hrts_cropNeVIII, lamb_0=lamb_0), y=rad_hrts_conv_scaled_cropNeVIII, yerr=erad_hrts_conv_scaled_cropNeVIII, color='black', linestyle='--', linewidth=2., label='HRST - QR A') #Real spectrum (SUMER)
ax.axvline(x=0, color='brown', linestyle=':', linewidth=2., label='Rest wavelength of Ne VIII/2')
ax.axvspan(-v_unc_0, v_unc_0, color='brown', alpha=0.15)
ax.set_title(f'Comparison SUMER before and after correction with HRTS', fontsize=18)
ax.set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
ax.set_xlim([vkms_doppler(lamb=min(lam_hrts_cropNeVIII), lamb_0=lamb_0), vkms_doppler(lamb=max(lam_hrts_cropNeVIII), lamb_0=lamb_0)])
ax.legend(fontsize=12)
ax.set_yscale('linear')
plt.show(block=False)


######################################################
######################################################
######################################################
# Fitting

######################################################
# Inputs


               

           
######################################################
# Fit uncorrected Ne VIII line

x_uncorrected = lam_sumer_cropNeVIII
y_uncorrected = rad_sumer_cropNeVIII
y_unc_uncorrected = erad_sumer_cropNeVIII


# Perform the fit
popt, pcov = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0), y_uncorrected, p0=init_parameters_uncorrected, sigma=y_unc_uncorrected, absolute_sigma=True) #popt are the optimized parameters. pcov is the covariance matrix of the parameters. 
perr = np.sqrt(np.diag(pcov)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_uncorrected = np.linspace(min(vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0)), max(vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0)), 300)
y_fit_uncorrected = multigaussian_function_for_curvefit(x_fit_uncorrected, *popt)


# Residuals
y_residuals = y_uncorrected - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0), *popt)
y_unc_fit_length_uncorrected = multi_gaussian_function_uncertainties(B=popt, B_unc=perr, x=vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0))))
y_unc_residuals = np.sqrt(y_unc_uncorrected**2 + y_unc_fit_length_uncorrected**2)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].errorbar(x=vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0) ,y=y_uncorrected, yerr=y_unc_uncorrected, color=color_sumer_uncorrected, marker='o', linewidth=0, elinewidth=1., label='SUMER uncorrected')
ax[0].plot(x_fit_uncorrected, y_fit_uncorrected, color='orange', linestyle='-', label='Fit', zorder=1) 
#ax[0].plot(vkms_doppler(lamb=x_fit_uncorrected_singlegauss, lamb_0=lamb_0), y_fit_uncorrected_singlegauss, color='magenta', linestyle='-', label='Individual gaussian', zorder=1)

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
    ax[0].plot(x_fit_corrected_singlegauss, y_fit_corrected_singlegauss, color=color_i, linestyle=':')#, label='Individual gaussians')
ax[0].plot([], [], color='grey', linestyle=':', label='Individual gaussians')

ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII/2')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].set_title(f'SUMER {percentage_label}% of the average of the spectroheliogram, uncorrected', fontsize=18)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=15)
ax[0].legend(fontsize=12)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_uncorrected, lamb_0=lamb_0), y=y_residuals, yerr=y_unc_residuals, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
ax[1].set_ylabel('Residuals', fontsize=15)
plt.tight_layout()
plt.show(block=False)

           
           
######################################################
# Fit corrected Ne VIII line

x_corrected = lam_sumer_cropNeVIII
y_corrected = rad_sumer_cropNeVIII_corrected
y_unc_corrected = erad_sumer_cropNeVIII_corrected


# Perform the fit
popt, pcov = curve_fit(multigaussian_function_for_curvefit, vkms_doppler(lamb=x_corrected, lamb_0=lamb_0), y_corrected, p0=init_parameters_corrected, sigma=y_unc_corrected, absolute_sigma=True) #popt are the optimized parameters. pcov is the covariance matrix of the parameters. 
perr = np.sqrt(np.diag(pcov)) #You can extract the standard deviation (1-sigma uncertainty) of the fitted parameters


# fitted curve
x_fit_corrected = np.linspace(min(vkms_doppler(lamb=x_corrected, lamb_0=lamb_0)), max(vkms_doppler(lamb=x_corrected, lamb_0=lamb_0)), 300)
y_fit_corrected = multigaussian_function_for_curvefit(x_fit_corrected, *popt)


# Residuals
y_residuals = y_corrected - multigaussian_function_for_curvefit(vkms_doppler(lamb=x_corrected, lamb_0=lamb_0), *popt)
y_unc_fit_length_corrected = multi_gaussian_function_uncertainties(B=popt, B_unc=perr, x=vkms_doppler(lamb=x_corrected, lamb_0=lamb_0), x_unc=np.zeros(len(vkms_doppler(lamb=x_corrected, lamb_0=lamb_0))))
y_unc_residuals = np.sqrt(y_unc_corrected**2 + y_unc_fit_length_corrected**2)


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].errorbar(x=vkms_doppler(lamb=x_corrected, lamb_0=lamb_0) ,y=y_corrected, yerr=y_unc_corrected, color=color_sumer_corrected, marker='o', linewidth=0, elinewidth=1., label='SUMER corrected')
ax[0].plot(x_fit_corrected, y_fit_corrected, color='green', linestyle='-', label='Fit', zorder=1) 
#ax[0].plot(vkms_doppler(lamb=x_fit_corrected_singlegauss, lamb_0=lamb_0), y_fit_corrected_singlegauss, color='magenta', linestyle='-', label='Individual gaussian', zorder=1)

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
    ax[0].plot(x_fit_corrected_singlegauss, y_fit_corrected_singlegauss, color=color_i, linestyle=':')#, label='Individual gaussians')
ax[0].plot([], [], color='grey', linestyle=':', label='Individual gaussians')

ax[0].axvline(x=0, color='black', linestyle='--', label='Ne VIII/2')
ax[0].axvspan(-v_unc_0, v_unc_0, color='grey', alpha=0.15)
ax[0].set_title(f'SUMER {percentage_label}% of the average of the spectroheliogram, corrected', fontsize=18)
ax[0].set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fontsize=15)
ax[0].legend(fontsize=12)
ax[0].set_yscale('linear')
ax[1].errorbar(x=vkms_doppler(lamb=x_corrected, lamb_0=lamb_0), y=y_residuals, yerr=y_unc_residuals, color='black', marker='.')
ax[1].set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
ax[1].set_ylabel('Residuals', fontsize=15)
plt.tight_layout()
plt.show(block=False)



#############################################


x_corrected = lam_sumer_cropNeVIII
x_uncorrected = x_corrected
y_corrected = rad_sumer_cropNeVIII_corrected
y_unc_corrected = erad_sumer_cropNeVIII_corrected
y_uncorrected = rad_sumer_cropNeVIII
y_unc_uncorrected = erad_sumer_cropNeVIII



xb_uncorrected, yb_uncorrected = find_bisector(x_data=x_uncorrected[3:-7], y_data=y_uncorrected[3:-7], y_unc_data=y_unc_uncorrected[3:-7], y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')

xb_corrected, yb_corrected = find_bisector(x_data=x_corrected, y_data=y_corrected, y_unc_data=y_unc_corrected, y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes')




