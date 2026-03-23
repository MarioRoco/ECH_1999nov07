# INPUTS

# Binning
bin_lat = 4
bin_lon = 1

# Line
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', 'cold_line'

show_spectral_image_binned = 'yes'
show_spectral_ranges = 'yes'
show_intensity_map_binned = 'yes'
show_scaling_factor_maps = 'yes'

threshold_value_type = 'max' #'max', 'min', 'mean', 'median'
range_percentage = [0., 4.]

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
exec(open("create_intensitymap_binned.py").read())


## Outputs:
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

#Line:', line_label
#w_px_range'+line_label, '= w_px_range'
#w_cal_range_pixcenter'+line_label, '= w_cal_range_pixcenter'
#w_px_range_bckg'+line_label, '= w_px_range_bckg'
#w_cal_range_pixcenter_bckg'+line_label, '= w_cal_range_pixcenter_bckg'
#intensitymap_croplat'+line_label, '= intensitymap_croplat'
#intensitymap_croplat_binned'+line_label, '= intensitymap_croplat_binned'

#####################################
# Import HRTS data

# Import HRTS spectra
# QR A
from hrts_spectra.data__qqr_a_xdr import lambda__qqr_a_xdr, radiance__qqr_a_xdr #[nm]
lam_hrtsa, rad_hrtsa = 10.*lambda__qqr_a_xdr, 0.1*radiance__qqr_a_xdr #[Angstrom]
label_hrtsa = r'QR-A close to the disk center ($\cos \theta = 0.92 - 0.98$)'
# QR B
from hrts_spectra.data__qqr_b_xdr import lambda__qqr_b_xdr, radiance__qqr_b_xdr #[nm]
lam_hrtsb, rad_hrtsb = 10.*lambda__qqr_b_xdr, 0.1*radiance__qqr_b_xdr #[Angstrom]
label_hrtsb = r'QR-B ($\cos \theta \sim 0.68$)'
# QR L
from hrts_spectra.data__qqr_l_xdr import lambda__qqr_l_xdr, radiance__qqr_l_xdr #[nm]
lam_hrtsl, rad_hrtsl = 10.*lambda__qqr_l_xdr, 0.1*radiance__qqr_l_xdr #[Angstrom]
label_hrtsl = r'QR-L close to the solar limb ($\cos \theta \sim 0.18$)'


#####################################
# Map of scaling factors not binned

# Create the map of scaling factors and the reduced chi^2
#scaling_factor_map, chi2redSF_map = get_factor_SUMER_HRTS(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list, unc_spectralimage_interp_list_=spectral_image_interpolated_croplat_list, lam_hrts=lam_hrts, rad_hrts=rad_hrts, fwhm_conv=fwhm_to_convolve, wavelength_range_preliminary=wavelength_range_scalefactor, show__row_col='no', y_scale='linear', show_legend='yes')

# Plot the map of scaling factors
#plot_2Darray_with_contours_and_pixel(array_2D=scaling_factor_map, spectroheliogram=intensitymap_croplat_NeVIII, bound=upper_bound_NeVIII, row_col__indices=lat_img__indices, show_contours='yes', color_contours='orange', show_pixel='yes', color_pixel='red', title='Scaling factor map', x_label='auto', y_label='auto', colorbar_label=r'Integrated radiance [W/sr/m$^2$]', cmap='Greys_r', z_scale='log', vmin_vmax='auto')

# Plot the map of reduced chi^2 of the scaling factors
#plot_2Darray_with_contours_and_pixel(array_2D=chi2redSF_map, spectroheliogram=intensitymap_croplat_NeVIII, bound=upper_bound_NeVIII, row_col__indices=lat_img__indices, show_contours='yes', color_contours='orange', show_pixel='yes', color_pixel='red', title=r'$\chi^2_{\rm red}$ of the scaling factor fit', x_label='auto', y_label='auto', colorbar_label=r'$\chi^2_{\rm red}$', cmap='Greys_r', z_scale='log', vmin_vmax='auto')


#####################################
# Map of scaling factors binned

# Create the map of scaling factors and the reduced chi^2
## HRTS QR-A
scaling_factor_map_binned_qra, chi2redSF_map_binned_qra = get_factor_SUMER_HRTS(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, lam_hrts=lam_hrtsa, rad_hrts=rad_hrtsa, fwhm_conv=fwhm_to_convolve, wavelength_range_preliminary=wavelength_range_scalefactor, show__row_col=lat_img__indices_binned, y_scale='linear', show_legend='yes')
## HRTS QR-B
scaling_factor_map_binned_qrb, chi2redSF_map_binned_qrb = get_factor_SUMER_HRTS(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, lam_hrts=lam_hrtsb, rad_hrts=rad_hrtsb, fwhm_conv=fwhm_to_convolve, wavelength_range_preliminary=wavelength_range_scalefactor, show__row_col=lat_img__indices_binned, y_scale='linear', show_legend='yes')
## HRTS QR-L
scaling_factor_map_binned_qrl, chi2redSF_map_binned_qrl = get_factor_SUMER_HRTS(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, lam_hrts=lam_hrtsl, rad_hrts=rad_hrtsl, fwhm_conv=fwhm_to_convolve, wavelength_range_preliminary=wavelength_range_scalefactor, show__row_col=lat_img__indices_binned, y_scale='linear', show_legend='yes')


#####################################
# 
lower_bound, upper_bound = get_bounds(intensitymap_croplat=intensitymap_croplat, range_percentage=range_percentage, threshold_value_type=threshold_value_type)
lower_bound_binned, upper_bound_binned = get_bounds(intensitymap_croplat=intensitymap_croplat_binned, range_percentage=range_percentage, threshold_value_type=threshold_value_type)


#####################################
#  Extent in pixels 
extent_sumer_px_contours = [0., intensitymap_croplat.shape[1]-1, intensitymap_croplat.shape[0]-1, 0.]
extent_sumer_px_image = [-0.5, intensitymap_croplat.shape[1]-1+0.5, intensitymap_croplat.shape[0]-1+0.5, -0.5]
extent_sumer_binned_px_contours = [0., intensitymap_croplat_binned.shape[1]-1, intensitymap_croplat_binned.shape[0]-1, 0.]
extent_sumer_binned_px_image = [-0.5, intensitymap_croplat_binned.shape[1]-1+0.5, intensitymap_croplat_binned.shape[0]-1+0.5, -0.5]

#####################################
# 

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
label_size = 18
ax.imshow(scaling_factor_map_binned_qra, norm=LogNorm(), cmap='Greys_r', aspect='auto')
ax.set_title('Scaling factor map, HRTS: QR-A', fontsize=18)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
plt.show(block=False)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
label_size = 18
ax.imshow(scaling_factor_map_binned_qrb, norm=LogNorm(), cmap='Greys_r', aspect='auto')
ax.set_title('Scaling factor map, HRTS: QR-B', fontsize=18)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
plt.show(block=False)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
label_size = 18
ax.imshow(scaling_factor_map_binned_qrl, norm=LogNorm(), cmap='Greys_r', aspect='auto')
ax.set_title('Scaling factor map, HRTS: QR-L', fontsize=18)
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
plt.show(block=False)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
label_size = 18
ax.set_title('Intensity map binned with contours not binned', fontsize=18)
ax.imshow(intensitymap_croplat_binned, norm=LogNorm(), cmap='Greys_r', aspect='auto')
ax.set_aspect('auto')
ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
fig.supylabel('Latitude dimension (pixels)', fontsize=17)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
plt.show(block=False)

#####################################

print('--------------------------------------')
print('lower_bound')
print('upper_bound')
print('lower_bound_binned')
print('upper_bound_binned')
print('lam_hrtsa')
print('rad_hrtsa')
print('label_hrtsa')
print('lam_hrtsb')
print('rad_hrtsb')
print('label_hrtsb')
print('lam_hrtsl')
print('rad_hrtsl')
print('label_hrtsl')
print('scaling_factor_map_binned_qra')
print('scaling_factor_map_binned_qrb')
print('scaling_factor_map_binned_qrl')
print('chi2redSF_map_binned_qra')
print('chi2redSF_map_binned_qrb')
print('chi2redSF_map_binned_qrl')
print('--------------------------------------')




