# INPUTS
"""
# Binning
bin_lat = 4
bin_lon = 1

# Line
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', 'cold_line'

show_spectral_image_binned = 'yes'
show_spectral_ranges = 'yes'
show_intensity_map_binned = 'yes'
"""

#####################################
# Bin data
exec(open("bin_data_interpolated.py").read())

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



#####################################
# 
# Calculate range of pixels   # For the wavelength ranges in the input: find the pixel range and the exact wavelength range (from the center of the pixels of the range)
pixelscale_reference = pixelscale_list[row_reference]
pixelscale_intercept_reference = pixelscale_intercept_list[row_reference]
w_px_range, w_cal_range_pixcenter = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=wavelength_range_spectroheliogram, slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference) # Line to analyze
w_px_range_bckg, w_cal_range_pixcenter_bckg = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=wavelength_range_spectroheliogram_bckg, slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference) # Background to subtract



# Binned. Show spectral image with vertical lines showing the wavelength range, and averaged profile
if show_spectral_ranges == 'yes':
    spectral_image_interp_croplat_binned_0 = spectral_image_interpolated_croplat_list_binned[0]
    unc_spectral_image_interp_croplat_binned_0 = spectral_image_unc_interpolated_croplat_list_binned[0]

    # Plot the spectral image with these ranges
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    lam_sumer_px = np.arange(spectral_image_interp_croplat_binned_0.shape[1]) 
    HPlat_crop_binned_px = np.arange(spectral_image_interp_croplat_binned_0.shape[0])
    img=ax.pcolormesh(lam_sumer_px, HPlat_crop_binned_px, spectral_image_interp_croplat_binned_0, cmap='Greys_r', norm=LogNorm())
    cax = fig.add_axes([0.99, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(r'Radiance [W/sr/m$^2$/''\u212B]', fontsize=16)
    ax.set_xlabel('wavelength direction (pixels)', fontsize=16)
    ax.set_ylabel('helioprojective latitude (pixels)', fontsize=16)
    ax.set_title(r'SUMER spectral image interpolated and cropped in $y$-axis, 'f'{line_center_label}''\n marking the wavelength ranges for the intensity. Binned', fontsize=18) 
    ax.set_aspect('auto') #'auto', 'equal'
    ax.axvline(w_px_range[0], color='blue', linewidth=1.2, label='Range for intensity map')
    ax.axvline(w_px_range[1], color='blue', linewidth=1.2)
    ax.axvline(w_px_range_bckg[0], color='red', linewidth=1.2, label='Range for background of the intensity map')
    ax.axvline(w_px_range_bckg[1], color='red', linewidth=1.2)
    # Define the forward (pixel -> wavelength) and inverse (wavelength -> pixel) mappings
    # Fit linear mapping λ = a*pix + b
    slope_x = (lam_sumer[-1]-lam_sumer[0]) / (lam_sumer_px[-1]-lam_sumer_px[0])
    intercept_x = lam_sumer[0]
    def pixel_to_lambda(x):
        return slope_x * x + intercept_x
    def lambda_to_pixel(l):
        return (l - intercept_x) / slope_x
    # Create top axis
    secax = ax.secondary_xaxis('top', functions=(pixel_to_lambda, lambda_to_pixel))
    secax.set_xlabel('Wavelength [Å]', fontsize=16)
    slope_y = (y_HPlat_crop_binned[-1]-y_HPlat_crop_binned[0]) / (HPlat_crop_binned_px[-1]-HPlat_crop_binned_px[0])
    intercept_y = y_HPlat_crop_binned[-1] - slope_y*HPlat_crop_binned_px[-1]
    def pixel_to_HPlat(x):
        return slope_y * x + intercept_y
    def HPlat_to_pixel(lt):
        return (lt - intercept_y) / slope_y
    secay = ax.secondary_yaxis('right', functions=(pixel_to_HPlat, HPlat_to_pixel))
    secay.set_ylabel('Helioprojective latitude (arcsec)', fontsize=16)
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    plt.show(block=False)




    # Profile 
    x_profile = lam_sumer_px
    y_profile = spectral_image_interp_croplat_binned_0.mean(axis=0)
    yerr_profile = (1/spectral_image_interp_croplat_binned_0.shape[0]) * np.sqrt((unc_spectral_image_interp_croplat_binned_0**2).sum(axis=0))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    ax.errorbar(x=x_profile, y=y_profile, yerr=yerr_profile, color='black', linewidth=1.5, label='Averaged profile along the slit')
    ax.set_xlabel('wavelength direction (pixels)', fontsize=16)
    ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', fontsize=16)
    ax.set_title('SUMER spectral profile averaged along\n the spatial direction of the interpolated spectral image. Binned', fontsize=18) 
    ax.set_aspect('auto') #'auto', 'equal'
    ax.axvspan(w_px_range[0], w_px_range[1], color='blue', alpha=0.15, label='Range for intensity map')
    ax.axvspan(w_px_range_bckg[0], w_px_range_bckg[1], color='red', alpha=0.15, label='Range for background of the intensity map')
    # Define the forward (pixel -> wavelength) and inverse (wavelength -> pixel) mappings
    # Fit linear mapping λ = a*pix + b
    slope_x = (lam_sumer[-1]-lam_sumer[0]) / (lam_sumer_px[-1]-lam_sumer_px[0])
    intercept_x = lam_sumer[0]
    def pixel_to_lambda(x):
        return slope_x * x + intercept_x
    def lambda_to_pixel(l):
        return (l - intercept_x) / slope_x
    # Create top axis
    secax = ax.secondary_xaxis('top', functions=(pixel_to_lambda, lambda_to_pixel))
    secax.set_xlabel('Wavelength [Å]', fontsize=16)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    plt.show(block=False)


#####################################
# Create intensity map (binned and not binned)
intensitymap_croplat = create_spectroheliogram_from_interpolated_data(spectralimage_interpolated_list=spectral_image_interpolated_croplat_list, w_px_range=w_px_range, w_px_range_bckg=w_px_range_bckg, slope_list=pixelscale_list_croplat, intercept_list=pixelscale_intercept_list_croplat)
intensitymap_croplat_binned = create_spectroheliogram_from_interpolated_data(spectralimage_interpolated_list=spectral_image_interpolated_croplat_list_binned, w_px_range=w_px_range, w_px_range_bckg=w_px_range_bckg, slope_list=pixelscale_list_croplat_binned, intercept_list=pixelscale_intercept_list_croplat_binned)

# Show intensity maps
if show_intensity_map_binned == 'yes':
    plot_2Darray(array_2D=intensitymap_croplat_binned, x_axis='pixels', y_axis='pixels', title=f'Intensity map binned {line_center_label}', x_label='auto', y_label='auto', z_label=r'Integrated radiance [W/sr/m$^2$]', cmap='Greys_r', z_scale='log', vmin_vmax='auto')


print('--------------------------------------')
print('Line:', line_label)
print('w_px_range'+line_label, '= w_px_range')
print('w_cal_range_pixcenter'+line_label, '= w_cal_range_pixcenter')
print('w_px_range_bckg'+line_label, '= w_px_range_bckg')
print('w_cal_range_pixcenter_bckg'+line_label, '= w_cal_range_pixcenter_bckg')
print('intensitymap_croplat'+line_label, '= intensitymap_croplat')
print('intensitymap_croplat_binned'+line_label, '= intensitymap_croplat_binned')
print('--------------------------------------')


