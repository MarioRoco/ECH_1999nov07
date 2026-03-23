

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



def fun_scale_hrts(hrts_qr, lamb_0, lam_sumer, rad_sumer, erad_sumer, fwhm_conv, wavelength_range_to_average, wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left, wavelength_range_scalefactor_right, show_plot='yes'):

    color_sumer = 'blue'
    color_hrts = 'green'
    color_sumer_uncorrected = 'red'
    color_sumer_corrected = 'blue'


    ######################################################
    # HRTS

    if hrts_qr=='a':
	    from hrts_spectra.data__qqr_a_xdr import lambda__qqr_a_xdr, radiance__qqr_a_xdr, unc_radiance__qqr_a_xdr
	    lam_hrts, rad_hrts = 10.*lambda__qqr_a_xdr, 0.1*radiance__qqr_a_xdr #multiply by 10. and 0.1 to convert nm to Angstrom
    elif hrts_qr=='b':
	    from hrts_spectra.data__qqr_b_xdr import lambda__qqr_b_xdr, radiance__qqr_b_xdr, unc_radiance__qqr_b_xdr
	    lam_hrts, rad_hrts = 10.*lambda__qqr_b_xdr, 0.1*radiance__qqr_b_xdr
    elif hrts_qr=='l':
	    from hrts_spectra.data__qqr_l_xdr import lambda__qqr_l_xdr, radiance__qqr_l_xdr, unc_radiance__qqr_l_xdr
	    lam_hrts, rad_hrts = 10.*lambda__qqr_l_xdr, 0.1*radiance__qqr_l_xdr

    erad_hrts = np.zeros(len(rad_hrts)) #TODO: we don't know the uncertainties of HRTS

        
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
    if show_plot=='yes':
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
    if show_plot=='yes':
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
    idx_NeVIII_sumer_0, idx_NeVIII_sumer_1 = indices_closer_to_range(arr_1d=lam_sumer, range_=wavelength_range_to_analyze_NeVIII)
    lam_sumer_cropNeVIII = lam_sumer[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
    rad_sumer_cropNeVIII = rad_sumer[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
    erad_sumer_cropNeVIII = erad_sumer[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
    rad_hrts_conv_SUMERgrid_cropNeVIII = rad_hrts_conv_SUMERgrid[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
    erad_hrts_conv_SUMERgrid_cropNeVIII = erad_hrts_conv_SUMERgrid[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
    rad_hrts_conv_scaled_SUMERgrid_cropNeVIII = rad_hrts_conv_scaled_SUMERgrid[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]
    erad_hrts_conv_scaled_SUMERgrid_cropNeVIII = erad_hrts_conv_scaled_SUMERgrid[idx_NeVIII_sumer_0:idx_NeVIII_sumer_1+1]

    # Crop HRTS original (nor interpolated to SUMER grid)
    idx_NeVIII_hrts_0, idx_NeVIII_hrts_1 = indices_closer_to_range(arr_1d=lam_hrts, range_=wavelength_range_to_analyze_NeVIII)
    lam_hrts_cropNeVIII = lam_hrts[idx_NeVIII_hrts_0:idx_NeVIII_hrts_1+1]
    rad_hrts_cropNeVIII = rad_hrts[idx_NeVIII_hrts_0:idx_NeVIII_hrts_1+1]
    erad_hrts_cropNeVIII = erad_hrts[idx_NeVIII_hrts_0:idx_NeVIII_hrts_1+1]
    rad_hrts_conv_scaled_cropNeVIII = rad_hrts_conv_scaled[idx_NeVIII_hrts_0:idx_NeVIII_hrts_1+1]
    erad_hrts_conv_scaled_cropNeVIII = erad_hrts_conv_scaled[idx_NeVIII_hrts_0:idx_NeVIII_hrts_1+1]


    # Plot
    if show_plot=='yes':
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.errorbar(x=lam_sumer, y=rad_sumer, yerr=erad_sumer, color=color_sumer_uncorrected, linewidth=0.6, label='SUMER full spectrum')
        ax.errorbar(x=lam_sumer_cropNeVIII, y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linewidth=0, marker='.', markersize=3, label='SUMER regions for analysis (Ne VIII)')
        ax.errorbar(x=lam_hrts, y=rad_hrts_conv_scaled, yerr=erad_hrts_conv_scaled, color=color_hrts, linewidth=0.6, label=r'HRST convolved and scaled ($\times$'f' {np.round(scaling_factor_,4)})') 
        ax.errorbar(x=lam_sumer_cropNeVIII, y=rad_hrts_conv_scaled_SUMERgrid_cropNeVIII, yerr=erad_hrts_conv_scaled_SUMERgrid_cropNeVIII, color=color_hrts, linewidth=0, elinewidth=1., marker='.', markersize=3, label='HRTS convolved, range of analysis (Ne VIII)') 
        ax.set_title(f'HRTS scaled', fontsize=18)
        ax.axvspan(wavelength_range_to_analyze_NeVIII[0], wavelength_range_to_analyze_NeVIII[1], color='grey', alpha=0.15, label='Wavelength range around Ne VIII')
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
    if show_plot=='yes':
        fig, ax = plt.subplots(figsize=(12, 5))
        #ax.errorbar(x=vkms_doppler(lamb=lam_crop, lamb_0=lamb_0), y=rad_crop, yerr=erad_crop, color='black', linewidth=0.6, label='SUMER box') #Real spectrum (SUMER) 
        ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII, yerr=erad_sumer_cropNeVIII, color=color_sumer_uncorrected, linestyle='-', linewidth=2., label=f'SUMER not corrected') 
        ax.errorbar(x=vkms_doppler(lamb=lam_sumer_cropNeVIII, lamb_0=lamb_0), y=rad_sumer_cropNeVIII_corrected, yerr=erad_sumer_cropNeVIII_corrected, color=color_sumer_corrected, linestyle='-', linewidth=2., label=f'SUMER corrected') 
        ax.errorbar(x=vkms_doppler(lamb=lam_hrts_cropNeVIII, lamb_0=lamb_0), y=rad_hrts_conv_scaled_cropNeVIII, yerr=erad_hrts_conv_scaled_cropNeVIII, color='black', linestyle='--', linewidth=2., label='HRST - QR A') #Real spectrum (SUMER)
        ax.axvline(x=0, color='brown', linestyle=':', linewidth=2., label='Rest wavelength of Ne VIII/2')
        #ax.axvspan(-v_unc_0, v_unc_0, color='brown', alpha=0.15)
        ax.set_title(f'Comparison SUMER before and after correction with HRTS', fontsize=18)
        ax.set_xlabel('Doppler shift (km/s)', fontsize=15)#'Wavelength (nm)'
        ax.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ ''\u212B'r'$^{-1}$)', fontsize=15)
        ax.set_xlim([vkms_doppler(lamb=min(lam_hrts_cropNeVIII), lamb_0=lamb_0), vkms_doppler(lamb=max(lam_hrts_cropNeVIII), lamb_0=lamb_0)])
        ax.legend(fontsize=12)
        ax.set_yscale('linear')
        plt.show(block=False)
        
    #return [lam_sumer_cropNeVIII, rad_sumer_cropNeVIII, erad_sumer_cropNeVIII, rad_sumer_cropNeVIII_corrected, erad_sumer_cropNeVIII_corrected]
    return {'lam_sumer_cropNeVIII': lam_sumer_cropNeVIII, 'rad_sumer_cropNeVIII': rad_sumer_cropNeVIII, 'erad_sumer_cropNeVIII': erad_sumer_cropNeVIII, 'rad_sumer_cropNeVIII_corrected': rad_sumer_cropNeVIII_corrected, 'erad_sumer_cropNeVIII_corrected': erad_sumer_cropNeVIII_corrected, 'lam_hrts': lam_hrts, 'rad_hrts': rad_hrts, 'erad_hrts': erad_hrts, 'rad_hrts_conv': rad_hrts_conv, 'erad_hrts_conv': erad_hrts_conv, 'rad_hrts_conv_scaled': rad_hrts_conv_scaled, 'erad_hrts_conv_scaled': erad_hrts_conv_scaled, 'lam_hrts_cropNeVIII': lam_hrts_cropNeVIII, 'rad_hrts_cropNeVIII': rad_hrts_cropNeVIII, 'erad_hrts_cropNeVIII': erad_hrts_cropNeVIII, 'rad_hrts_conv_scaled_cropNeVIII': rad_hrts_conv_scaled_cropNeVIII, 'erad_hrts_conv_scaled_cropNeVIII': erad_hrts_conv_scaled_cropNeVIII}


#rad_hrts_conv_SUMERgrid, erad_hrts_conv_SUMERgrid
#rad_hrts_conv_scaled_SUMERgrid, erad_hrts_conv_scaled_SUMERgrid
#rad_hrts_conv_SUMERgrid_cropNeVIII, erad_hrts_conv_SUMERgrid_cropNeVIII
#rad_hrts_conv_scaled_SUMERgrid_cropNeVIII, erad_hrts_conv_scaled_SUMERgrid_cropNeVIII

"""
fsh = fun_scale_hrts(hrts_qr=hrts_qr, lamb_0=lamb_0, lam_sumer=lam_sumer_av, rad_sumer=rad_sumer_av, erad_sumer=erad_sumer_av, fwhm_conv=fwhm_conv, wavelength_range_to_average=wavelength_range_to_average, wavelength_range_to_analyze_NeVIII=wavelength_range_to_analyze_NeVIII, wavelength_range_scalefactor_left=wavelength_range_scalefactor_left, wavelength_range_scalefactor_right=wavelength_range_scalefactor_right, show_plot=show_plots_correction)
lam_sumer_cropNeVIII = fsh['lam_sumer_cropNeVIII']
rad_sumer_cropNeVIII = fsh['rad_sumer_cropNeVIII']
erad_sumer_cropNeVIII = fsh['erad_sumer_cropNeVIII']
rad_sumer_cropNeVIII_corrected = fsh['rad_sumer_cropNeVIII_corrected']
erad_sumer_cropNeVIII_corrected = fsh['erad_sumer_cropNeVIII_corrected']
lam_hrts = fsh['lam_hrts']
rad_hrts = fsh['rad_hrts']
erad_hrts = fsh['erad_hrts']
rad_hrts_conv = fsh['rad_hrts_conv']
erad_hrts_conv = fsh['erad_hrts_conv']
rad_hrts_conv_scaled = fsh['rad_hrts_conv_scaled']
erad_hrts_conv_scaled = fsh['erad_hrts_conv_scaled']
lam_hrts_cropNeVIII = fsh['lam_hrts_cropNeVIII']
rad_hrts_cropNeVIII = fsh['rad_hrts_cropNeVIII']
erad_hrts_cropNeVIII = fsh['erad_hrts_cropNeVIII']
rad_hrts_conv_scaled_cropNeVIII = fsh['rad_hrts_conv_scaled_cropNeVIII']
erad_hrts_conv_scaled_cropNeVIII = fsh['erad_hrts_conv_scaled_cropNeVIII']
"""



