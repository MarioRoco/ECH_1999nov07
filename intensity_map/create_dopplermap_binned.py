# INPUTS

# Binning
bin_lat = 4
bin_lon = 1

# Line
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', 'cold_line'

show_spectral_image_binned = 'no'
show_spectral_ranges = 'no'
show_intensity_map_binned = 'yes'
show_scaling_factor_maps = 'no'
show_dopplermaps = 'yes'
show_dopplermaps_lessmedian = 'yes'
show_chi2red_of_dopplermaps = 'no'

threshold_value_type = 'max' #'max', 'min', 'mean', 'median'
range_percentage = [0., 4.]

wavelength_range_analysis = [1540.1, 1541.5] #Angstroem
wavelength_range_scalefactor = [1542., 1544.] #Angstroem

# Pixel address
lat_img__indices_binned = 46, 156
lat_img__indices = bin_lat*lat_img__indices_binned[0], bin_lon*lat_img__indices_binned[1]

## FWHM of the cold lines in SUMER:
fwhm_mean_weighted_sumer =  0.17740
fwhm_std_sumer =  0.02094
fwhm_unc_weighted_sumer =  0.00131
fwhm_synthetic_Si = 0.03
fwhm_sumer_to_convolve = fwhm_mean_weighted_sumer - fwhm_synthetic_Si
fwhm_to_convolve = fwhm_sumer_to_convolve #Usser can addapt this value
fwhm_to_convolve = 1.95*0.043


#####################################
# Bin data
#exec(open("bin_data_interpolated.py").read()) #it is inside create_intensitymap_binned.py
#exec(open("create_intensitymap_binned.py").read())
exec(open("create_scale_factor_map_binned.py").read())

## Outputs:
### exec(open("bin_data_interpolated.py").read()) 
#spectral_image_interpolated_list
#spectral_image_unc_interpolated_list
#spectral_image_interpolated_croplat_list
#spectral_image_unc_interpolated_croplat_list
#lam_sumer
#lam_sumer_unc
#row_reference
#w_px_range'+line_label, '= w_px_range
#w_cal_range_pixcenter'+line_label, '= w_cal_range_pixcenter
#w_px_range_bckg'+line_label, '= w_px_range_bckg')
#w_cal_range_pixcenter_bckg'+line_label, '= w_cal_range_pixcenter_bckg
#spectral_image_interpolated_croplat_list_binned')
#spectral_image_unc_interpolated_croplat_list_binned
#pixelscale_list_croplat_binned
#pixelscale_intercept_list_croplat_binned
#x_HPlon_rotcomp_binned
#y_HPlat_crop_binned

### exec(open("create_intensitymap_binned.py").read())
#Line:', line_label
#w_px_range'+line_label, '= w_px_range'
#w_cal_range_pixcenter'+line_label, '= w_cal_range_pixcenter'
#w_px_range_bckg'+line_label, '= w_px_range_bckg'
#w_cal_range_pixcenter_bckg'+line_label, '= w_cal_range_pixcenter_bckg'
#intensitymap_croplat'+line_label, '= intensitymap_croplat'
#intensitymap_croplat_binned'+line_label, '= intensitymap_croplat_binned'

### exec(open("create_scale_factor_map_binned.py").read())
#lower_bound
#upper_bound
#lower_bound_binned
#upper_bound_binned
#lam_hrtsa
#rad_hrtsa
#label_hrtsa
#lam_hrtsb
#rad_hrtsb
#label_hrtsb
#lam_hrtsl
#rad_hrtsl
#label_hrtsl
#scaling_factor_map_binned_qra
#scaling_factor_map_binned_qrb
#scaling_factor_map_binned_qrl
#chi2redSF_map_binned_qra
#chi2redSF_map_binned_qrb
#chi2redSF_map_binned_qrl

######################################################

# Rest wavelength used
rest_wavelength_label = 'Peter_and_Judge_1999' #'SUMER_atlas', 'Peter_1998', 'Dammasch_1999', 'Peter_and_Judge_1999', 'Kelly_database'
lam_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][0] #Angstrom
lam_unc_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][1] #Angstrom
print('Rest wavelength Ne VIII (2nd order):', lam_0, r'$\pm$', lam_unc_0, '\u212B')

# uncertainty of the rest wavelength in km/s
v_unc_0 = vkms_doppler_unc(lamb=lam_0, lamb_unc=lam_unc_0, lamb_0=lam_0, lamb_0_unc=lam_unc_0) 


#####################################
# 1) Calculate Dopplermap, 2) Plot Dopplermap (fitting a single Gaussian) 3) Plot map of chi^2_red

## HRTS not subtracted
dopplershift_map_binned, chi2red_map_binned = create_SUMER_dopplermap_single_gaussianfit(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', show__row_col='no', y_scale='linear', show_legend='no') #show__row_col=lat_img__indices_binned
## HRTS subtracted, QR-A
dopplershift_map_binned_HRTSsub_qra, chi2red_map_binned_HRTSsub_qra = create_SUMER_dopplermap_single_gaussianfit(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, subtract_HRTS='yes', lam_hrts=lam_hrtsa, rad_hrts=rad_hrtsa, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qra, show__row_col='no', y_scale='linear', show_legend='no') #show__row_col=lat_img__indices_binned
## HRTS subtracted, QR-B
dopplershift_map_binned_HRTSsub_qrb, chi2red_map_binned_HRTSsub_qrb = create_SUMER_dopplermap_single_gaussianfit(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, subtract_HRTS='yes', lam_hrts=lam_hrtsb, rad_hrts=rad_hrtsb, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qrb, show__row_col='no', y_scale='linear', show_legend='no') #show__row_col=lat_img__indices_binned
## HRTS subtracted, QR-L
dopplershift_map_binned_HRTSsub_qrl, chi2red_map_binned_HRTSsub_qrl = create_SUMER_dopplermap_single_gaussianfit(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, subtract_HRTS='yes', lam_hrts=lam_hrtsl, rad_hrts=rad_hrtsl, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qrl, show__row_col='no', y_scale='linear', show_legend='no') #show__row_col=lat_img__indices_binned


#####################################
# Subtract median for each row
dopplershift_map_binned_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned)
dopplershift_map_binned_HRTSsub_qra_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qra)
dopplershift_map_binned_HRTSsub_qrb_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qrb)
dopplershift_map_binned_HRTSsub_qrl_lessmedian = subtract_median_rows(arr_2D=dopplershift_map_binned_HRTSsub_qrl)

#####################################
# Show images

if show_dopplermaps == 'yes':
    vmin_vmax__dopplermap = [-12, 12]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qra, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qrb, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-B', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qrl, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-L', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)
    

if show_dopplermaps_lessmedian == 'yes':
    vmin_vmax__dopplermap = [-12, 12]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_lessmedian, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qra_lessmedian, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-A, median subtracted', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qrb_lessmedian, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-B, median subtracted', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(dopplershift_map_binned_HRTSsub_qrl_lessmedian, cmap='seismic', vmin=vmin_vmax__dopplermap[0], vmax=vmin_vmax__dopplermap[1], aspect='auto')
    ax.set_title('Dopplershift map, HRTS: QR-L, median subtracted', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)



if show_chi2red_of_dopplermaps == 'yes':

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(chi2red_map_binned, cmap='Greys_r', aspect='auto')
    ax.set_title(r'$chi^2_{\rm red}$ of the Dopplershift map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(chi2red_map_binned_HRTSsub_qra, cmap='Greys_r', aspect='auto')
    ax.set_title(r'$chi^2_{\rm red}$ of the Dopplershift map, HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(chi2red_map_binned_HRTSsub_qrb, cmap='Greys_r', aspect='auto')
    ax.set_title(r'$chi^2_{\rm red}$ of the Dopplershift map, HRTS: QR-B', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(chi2red_map_binned_HRTSsub_qrl, cmap='Greys_r', aspect='auto')
    ax.set_title(r'$chi^2_{\rm red}$ of the Dopplershift map, HRTS: QR-L', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)



#####################################

print('--------------------------------------')
print('rest_wavelength_label')
print('lam_0')
print('lam_unc_0')
print('v_unc_0')
print('dopplershift_map_binned')
print('chi2red_map_binned')
print('dopplershift_map_binned_HRTSsub_qra')
print('dopplershift_map_binned_HRTSsub_qrb')
print('dopplershift_map_binned_HRTSsub_qrl')
print('dopplershift_map_binned_HRTSsub_qra_lessmedian')
print('dopplershift_map_binned_HRTSsub_qrb_lessmedian')
print('dopplershift_map_binned_HRTSsub_qrl_lessmedian')
print('chi2red_map_binned_HRTSsub_qra')
print('chi2red_map_binned_HRTSsub_qrb')
print('chi2red_map_binned_HRTSsub_qrl')
print('--------------------------------------')



######################################################






