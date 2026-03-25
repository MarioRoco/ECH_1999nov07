# INPUTS

# Binning
bin_lat = 4
bin_lon = 1

# Line
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', 'cold_line'

show_spectral_image_binned = 'no'
show_spectral_ranges = 'no'
show_intensity_map_binned = 'no'
show_scaling_factor_maps = 'no'
show_BRmap = 'yes'
show_BRmap_lessmedian = 'yes'
show_chi2red_of_dopplermaps = 'no'

threshold_value_type = 'max' #'max', 'min', 'mean', 'median'
range_percentage = [0., 4.]

wavelength_range_analysis = [1540.1, 1541.5] #Angstroem
wavelength_range_scalefactor = [1542., 1544.] #Angstroem

# Pixel address
lat_img__indices_binned = 46, 15
lat_img__indices = bin_lat*lat_img__indices_binned[0], bin_lon*lat_img__indices_binned[1]

## FWHM of the cold lines in SUMER:
fwhm_mean_weighted_sumer =  0.17740
fwhm_std_sumer =  0.02094
fwhm_unc_weighted_sumer =  0.00131
fwhm_synthetic_Si = 0.03
fwhm_sumer_to_convolve = fwhm_mean_weighted_sumer - fwhm_synthetic_Si
fwhm_to_convolve = fwhm_sumer_to_convolve #Usser can addapt this value
fwhm_to_convolve = 1.95*0.04215 #Angstrom

BR_distance_centroid = 50. #[km/s] Distance of the center of the range from the centroid of the fitted gaussian
BR_width = 50. #[km/s]

filename_eit = 'SOHO_EIT_195_19991107T042103_L1.fits'


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

######################################################
# 


def BR_asymmetry_one_spectrum_with_centroid_of_gaussian_division(x_data, y_data, y_unc_data, BR_distance_centroid, BR_width, BR_centroid, show_profile='no', show_legend='yes'):

    # width of the pixels (supposing same width for all of them)
    pixel_width = x_data[1]-x_data[0] 
    
    # Convert masked values to NaNs:
    y_data = np.ma.filled(y_data, np.nan)
    
    # Create the interpolation function (linear unterpolation)
    interp_func = interp1d(x_data, y_data, kind='linear')
    
    # Edges of the pixels
    pixel_low = x_data - pixel_width/2
    pixel_high = x_data + pixel_width/2
    
    # Calculate the ranges for blue and red parts.
    ## Blue part
    range_blue0 = BR_centroid - BR_distance_centroid - BR_width/2
    range_blue1 = BR_centroid - BR_distance_centroid + BR_width/2
    range_B = [range_blue0, range_blue1]
    ## Red part
    range_red0 = BR_centroid + BR_distance_centroid - BR_width/2
    range_red1 = BR_centroid + BR_distance_centroid + BR_width/2
    range_R = [range_red0, range_red1]
    
    # Indices inside the range
    ## Blue
    ilB = np.argmax(pixel_low > range_B[0]) # find the first index where the data point is greater than range_[0]
    ihB = np.argmax(pixel_high > range_B[1])-1 # find the first index where the data point is smaller than range_[1]
    ## Red
    ilR = np.argmax(pixel_low > range_R[0]) # find the first index where the data point is greater than range_[0]
    ihR = np.argmax(pixel_high > range_R[1])-1 # find the first index where the data point is smaller than range_[1]
    
    # Data points (entire pixels) that fit inside each range 
    ## Blue
    indices_inrange_B = np.arange(ilB,ihB+1)
    x_inrange_B = x_data[indices_inrange_B]
    y_inrange_B = y_data[indices_inrange_B]
    y_unc_data_inrange_B = y_unc_data[indices_inrange_B]
    ## Red
    indices_inrange_R = np.arange(ilR,ihR+1)
    x_inrange_R = x_data[indices_inrange_R]
    y_inrange_R = y_data[indices_inrange_R]
    y_unc_data_inrange_R = y_unc_data[indices_inrange_R]
    
    # Adding the partial pixels of the edge of the range
    ## Blue
    x_interpolated_left_B = np.mean([range_B[0], x_data[ilB]-pixel_width/2])
    x_interpolated_right_B = np.mean([range_B[1], x_data[ihB]+pixel_width/2])
    x_analysis_B = np.concatenate([[x_interpolated_left_B], x_inrange_B, [x_interpolated_right_B]])
    y_analysis_B = interp_func(x_analysis_B)
    ## Red
    x_interpolated_left_R = np.mean([range_R[0], x_data[ilR]-pixel_width/2])
    x_interpolated_right_R = np.mean([range_R[1], x_data[ihR]+pixel_width/2])
    x_analysis_R = np.concatenate([[x_interpolated_left_R], x_inrange_R, [x_interpolated_right_R]])
    y_analysis_R = interp_func(x_analysis_R)
    
    # Delta pixels (of the range)
    ## Blue
    deltapixel_inrange_B = pixel_width * np.ones(len(x_inrange_B))
    deltapixel_left_B = abs((x_data[ilB]-pixel_width/2) - range_B[0])
    deltapixel_right_B = abs(range_B[1] - (x_data[ihB]+pixel_width/2))
    deltapixel_analysis_B = np.concatenate([[deltapixel_left_B], deltapixel_inrange_B, [deltapixel_right_B]])
    ## Red
    deltapixel_inrange_R = pixel_width * np.ones(len(x_inrange_R))
    deltapixel_left_R = abs((x_data[ilR]-pixel_width/2) - range_R[0])
    deltapixel_right_R = abs(range_R[1] - (x_data[ihR]+pixel_width/2))
    deltapixel_analysis_R = np.concatenate([[deltapixel_left_R], deltapixel_inrange_R, [deltapixel_right_R]])
    
    # Mask the NaN values (otherwise the result is a nan). If the input has masked values, they are converted to NaN values, so here we need to mask them.
    ## Blue
    mask_B = np.isnan(y_analysis_B)  # Create a boolean mask where ia has NaNs
    y_analysis_masked_B = y_analysis_B[~mask_B]  # Select elements of ia where mask is False (i.e., not NaN)
    deltapixel_analysis_masked_B = deltapixel_analysis_B[~mask_B]  # Similarly filter the weights
    ia_B, w_B = y_analysis_masked_B, deltapixel_analysis_masked_B
    y_weighted_mean_B = np.sum(w_B*ia_B) / np.sum(w_B)
    ## Red
    mask_R = np.isnan(y_analysis_R)  # Create a boolean mask where ia has NaNs
    y_analysis_masked_R = y_analysis_R[~mask_R]  # Select elements of ia where mask is False (i.e., not NaN)
    deltapixel_analysis_masked_R = deltapixel_analysis_R[~mask_R]  # Similarly filter the weights
    ia_R, w_R = y_analysis_masked_R, deltapixel_analysis_masked_R
    y_weighted_mean_R = np.sum(w_R*ia_R) / np.sum(w_R)
    
    # Calculate B-R asymmetry
    BR_asymmetry = y_weighted_mean_R / y_weighted_mean_B
    
    
    # 9) Show profile
    if show_profile=='yes':
        # Generate new x values for smooth curve
        interp_xcurve = np.linspace(min(x_data), max(x_data), 500) # More points for smooth line
        interp_ycurve = interp_func(interp_xcurve)  # Interpolated y values

        # Wavelengths corresponding to the edges of the pixels
        pixel_edges = np.concatenate([pixel_low, [pixel_high[-1]]])

        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,6))
        ax.errorbar(x=x_data, y=y_data, yerr=y_unc_data, linewidth=0, elinewidth=1, marker='.', color='black')#, label='Intensities of every pixel (center)') # Plot the original data points
        for pe in pixel_edges:
            ax.axvline(pe, color='black', linewidth=0.5)
        ax.plot([],[], color='black', linewidth=0.5, label='Edges of the pixels')
        ax.plot(interp_xcurve, interp_ycurve, linestyle='--', linewidth=0.5, color='blue')#, label='Linear interpolation') # Plot the interpolated line
        ax.axvspan(range_B[0], range_B[1], color='blue', linestyle=':', alpha=0.15)#, label='Blue range\n'f'{range_B[0]} - {range_B[1]} {xunits}')
        ax.axvspan(range_R[0], range_R[1], color='red', linestyle=':', alpha=0.15)#, label='Red range\n'f'{range_R[0]} - {range_R[1]} {xunits}')

        # Adding the partial pixels of the edge of the range
        ax.scatter(x_analysis_B, y_analysis_B, color='blue', marker='s', s=20)#, label='"Artificial" pixels at the edges of the range (blue)') # Plot the original data points
        ax.scatter(x_analysis_R, y_analysis_R, color='red', marker='s', s=20)#, label='"Artificial" pixels at the edges of the range (red)') # Plot the original data points
        
        # Delta pixel
        ax.errorbar(x=x_analysis_B, y=y_analysis_B, xerr=deltapixel_analysis_masked_B/2, color='blue', linewidth=0, elinewidth=1.2)
        ax.errorbar(x=x_analysis_R, y=y_analysis_R, xerr=deltapixel_analysis_masked_R/2, color='red', linewidth=0, elinewidth=1.2)

        # Horizontal line as weighted mean
        ax.plot(range_B, [y_weighted_mean_B, y_weighted_mean_B], color='blue', linestyle='--')#, label=f'Weighted mean of the intensity (blue): {y_weighted_mean_B}')
        ax.plot(range_R, [y_weighted_mean_R, y_weighted_mean_R], color='red', linestyle='--')#, label=f'Weighted mean of the intensity (red): {y_weighted_mean_R}')

        # Vertical lines: centroid and rest wavelength
        ax.axvline(x=0, color='green', linestyle=':', label=f'Rest wavelength')
        ax.axvline(x=BR_centroid, color='purple', linestyle=':', label=f'Centroid: {np.round(BR_centroid,3)} km/s')
        
        ax.set_xlabel('Doppler velocity (km/s)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'B-R asymmetry (with interpolation): {np.round(BR_asymmetry,5)}')
        if show_legend=='yes': ax.legend(fontsize=10)
        plt.show(block=False)
        #ax.set_xlim([-110,-50])
        ax.legend()
    
        
    return BR_asymmetry
    #return [interp_func, BR_asymmetry, interp_func, pixel_width, pixel_low, pixel_high, range_B, x_inrange_B, y_inrange_B, y_unc_data_inrange_B, x_interpolated_left_B, x_interpolated_right_B, x_analysis_B, y_analysis_B, deltapixel_inrange_B, deltapixel_left_B, deltapixel_right_B, deltapixel_analysis_B, y_weighted_mean_B, range_R, x_inrange_R, y_inrange_R, y_unc_data_inrange_R, x_interpolated_left_R, x_interpolated_right_R, x_analysis_R, y_analysis_R, deltapixel_inrange_R, deltapixel_left_R, deltapixel_right_R, deltapixel_analysis_R, y_weighted_mean_R]

##################################################
# create BR asymmetry map

def create_BRasymmetrymap_with_centroid_of_gaussian_division(lam_sumer, spectralimage_interp_list_, unc_spectralimage_interp_list_, wavelength_range_preliminary, rest_wavelength, BR_distance_centroid, BR_width, subtract_HRTS='no', lam_hrts='no', rad_hrts='no', fwhm_conv='no', scalefactor_hrts_2Darr='no', show__row_col='no', y_scale='linear', show_legend='yes'):
    
    # 1) Crop SUMER wavelength array in the range selected in the input
    lam_sumer_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer, range_values=wavelength_range_preliminary)
    col1, col2 = idx_sumer_crop    
    v_sumer_crop = vkms_doppler(lamb=lam_sumer_crop, lamb_0=rest_wavelength)
    
    N_img = len(spectralimage_interp_list_)
    N_rows = spectralimage_interp_list_[0].shape[0]
    #N_cols_full = spectralimage_interp_list_[0].shape[1]
    #N_cols = col2-col1+1
    BR_asymmetry_NT = []
    for img_ in tqdm(range(N_img)):
        BR_asymmetry_1sp_allrows = []
        for row_ in range(N_rows):
                        
            # 2) Take the data of the current spectral image and row
            lam_sumer = lam_sumer #or lam_sumer[col1:col2+1]
            rad_sumer = spectralimage_interp_list_[img_][row_,:]
            erad_sumer = unc_spectralimage_interp_list_[img_][row_,:]
            
            # 3) Crop SUMER data
            lam_sumer_crop = lam_sumer_crop #or lam_sumer[col1:col2+1]
            rad_sumer_crop = rad_sumer[col1:col2+1]
            erad_sumer_crop = erad_sumer[col1:col2+1]
            
            if subtract_HRTS!='no':
                # 4) Prepare HRTS data for the subtraction
                scale_factor_hrts = scalefactor_hrts_2Darr[row_, img_]
                interp_func_hrts, lam_hrts, rad_hrts_conv_scaled = prepare_HRTS_for_subtraction(lam_hrts=lam_hrts, rad_hrts=rad_hrts, fwhm_conv=fwhm_conv, scale_factor=scale_factor_hrts)

                # 5) Get data of HRTS corresponding to SUMER's wavelength grid (of the cropped range)
                rad_hrts_SUMERgrid = interp_func_hrts(lam_sumer_crop)
                
                # 6) Crop HRTS
                lam_hrts_crop, idx_hrts_crop = crop_range(list_to_crop=lam_hrts, range_values=wavelength_range_preliminary)
                rad_hrts_conv_scaled_crop = rad_hrts_conv_scaled[idx_hrts_crop[0]:idx_hrts_crop[1]+1]

                # 7) Subtract HRTS to SUMER
                rad_sumer_crop_tofit = rad_sumer_crop - rad_hrts_SUMERgrid
                erad_sumer_crop_tofit = erad_sumer_crop #TODO: not considering HRTS erorbars (wich we don't know)
            
            else: 
                rad_sumer_crop_tofit = rad_sumer_crop
                erad_sumer_crop_tofit = erad_sumer_crop
                
            # Take the data of the current spectral image and row
            v_data = v_sumer_crop
            y_data = rad_sumer_crop_tofit
            y_unc_data = erad_sumer_crop_tofit
            
            # Fit a single gaussian (and background) to the data
            fit_results = fit_automatic_gaussian_with_background(x_data=v_data, y_data=y_data, y_unc_data=y_unc_data)

            # Calculate the Doppler shift of the centroid of the line, and the uncertainty. And save in lists.
            if fit_results!=None: 
                BR_centroid_i_j = fit_results['mean'][0]
            else:
                BR_centroid_i_j = np.nan
            
            if show__row_col!='no' and show__row_col[0]==row_ and show__row_col[1]==img_: show_profile_ij='yes'
            else: show_profile_ij='no'
            
            # Calculate B-R asymmetry of a single spectrum, and save it in a list
            threshold_left = BR_centroid_i_j - BR_distance_centroid - BR_width/2
            threshold_right = BR_centroid_i_j + BR_distance_centroid + BR_width/2
            if threshold_left < min(v_data) or threshold_right > max(v_data): 
                BR_asymmetry_1sp_allrows.append(np.nan)
            else:
                BR_asymmetry_i_j = BR_asymmetry_one_spectrum_with_centroid_of_gaussian_division(x_data=v_data, y_data=y_data, y_unc_data=y_unc_data, BR_centroid=BR_centroid_i_j, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width, show_profile=show_profile_ij)
                BR_asymmetry_1sp_allrows.append(BR_asymmetry_i_j)

            
        # save the above lists (which represent the different spectral images) in lists to crate the 2D-array
        BR_asymmetry_NT.append(BR_asymmetry_1sp_allrows)

    # Convert to array and transpose
    BR_asymmetry_map = np.array(BR_asymmetry_NT).T 
    
    return BR_asymmetry_map

######################################################
# Create asymmetry map by subtracting Red-Blue
"""
# HRTS not subtracted
BR_asymmetry_map_gaussian_binned, BR_asymmetry_map_gaussian_binned_normalized = create_BRasymmetrymap_with_centroid_of_gaussian(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width, subtract_HRTS='no', lam_hrts=lam_hrtsa, rad_hrts=rad_hrtsa, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qra, show__row_col=lat_img__indices_binned, y_scale='linear', show_legend='yes')
"""

# HRTS subtracted, QR A
BR_asymmetry_map_gaussian_binned_HRTSsub_qra, BR_asymmetry_map_gaussian_binned_HRTSsub_qra_normalized = create_BRasymmetrymap_with_centroid_of_gaussian(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width, subtract_HRTS='yes', lam_hrts=lam_hrtsa, rad_hrts=rad_hrtsa, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qra, show__row_col=lat_img__indices_binned, y_scale='linear', show_legend='yes')

"""
# HRTS subtracted, QR B
BR_asymmetry_map_gaussian_binned_HRTSsub_qrb, BR_asymmetry_map_gaussian_binned_HRTSsub_qrb_normalized = create_BRasymmetrymap_with_centroid_of_gaussian(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width, subtract_HRTS='yes', lam_hrts=lam_hrtsb, rad_hrts=rad_hrtsb, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qrb, show__row_col=lat_img__indices_binned, y_scale='linear', show_legend='yes')

# HRTS subtracted, QR L
BR_asymmetry_map_gaussian_binned_HRTSsub_qrl, BR_asymmetry_map_gaussian_binned_HRTSsub_qrl_normalized = create_BRasymmetrymap_with_centroid_of_gaussian(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width, subtract_HRTS='yes', lam_hrts=lam_hrtsl, rad_hrts=rad_hrtsl, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qrl, show__row_col=lat_img__indices_binned, y_scale='linear', show_legend='yes')
"""
######################################################
# Create asymmetry map by dividing Red/Blue
"""
# HRTS not subtracted
BR_asymmetry_map_gaussian_binned_division = create_BRasymmetrymap_with_centroid_of_gaussian_division(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width, subtract_HRTS='no', lam_hrts=lam_hrtsa, rad_hrts=rad_hrtsa, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qra, show__row_col='yes', y_scale='linear', show_legend='yes')
"""
# HRTS subtracted, QR A
BR_asymmetry_map_gaussian_binned_HRTSsub_qra_division = create_BRasymmetrymap_with_centroid_of_gaussian_division(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width, subtract_HRTS='yes', lam_hrts=lam_hrtsa, rad_hrts=rad_hrtsa, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qra, show__row_col=lat_img__indices_binned, y_scale='linear', show_legend='yes')
"""
# HRTS subtracted, QR B
BR_asymmetry_map_gaussian_binned_HRTSsub_qrb_division = create_BRasymmetrymap_with_centroid_of_gaussian_division(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width, subtract_HRTS='yes', lam_hrts=lam_hrtsb, rad_hrts=rad_hrtsb, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qrb, show__row_col='yes', y_scale='linear', show_legend='yes')

# HRTS subtracted, QR L
BR_asymmetry_map_gaussian_binned_HRTSsub_qrl_division = create_BRasymmetrymap_with_centroid_of_gaussian_division(lam_sumer=lam_sumer, spectralimage_interp_list_=spectral_image_interpolated_croplat_list_binned, unc_spectralimage_interp_list_=spectral_image_unc_interpolated_croplat_list_binned, wavelength_range_preliminary=wavelength_range_analysis, rest_wavelength=lam_0, BR_distance_centroid=BR_distance_centroid, BR_width=BR_width, subtract_HRTS='yes', lam_hrts=lam_hrtsl, rad_hrts=rad_hrtsl, fwhm_conv=fwhm_to_convolve, scalefactor_hrts_2Darr=scaling_factor_map_binned_qrl, show__row_col='yes', y_scale='linear', show_legend='yes')
"""


#####################################
# 

if show_BRmap == 'yes':
    vmin_vmax = [-0.1, 0.1]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(BR_asymmetry_map_gaussian_binned_HRTSsub_qra, cmap='seismic', vmin=vmin_vmax[0], vmax=vmin_vmax[1], aspect='auto')
    ax.set_title('BR assymetry map not normalized, corrected with HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)
    
    
    vmin_vmax = [-0.1, 0.1]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(BR_asymmetry_map_gaussian_binned_HRTSsub_qra_normalized, cmap='seismic', vmin=vmin_vmax[0], vmax=vmin_vmax[1], aspect='auto')
    ax.set_title('BR assymetry map normalized, corrected with HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)
    
    vmin_vmax = [-0.1, 0.1]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(BR_asymmetry_map_gaussian_binned_HRTSsub_qra_division, cmap='seismic', vmin=vmin_vmax[0], vmax=vmin_vmax[1], aspect='auto')
    ax.set_title('R/B normalized, corrected with HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)

    
    
    


if show_BRmap_lessmedian == 'yes':
    vmin_vmax = [-0.1, 0.1]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(subtract_median_rows(arr_2D=BR_asymmetry_map_gaussian_binned_HRTSsub_qra), cmap='seismic', vmin=vmin_vmax[0], vmax=vmin_vmax[1], aspect='auto')
    ax.set_title('BR assymetry map not normalized, corrected with HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(subtract_median_rows(arr_2D=BR_asymmetry_map_gaussian_binned_HRTSsub_qra_normalized), cmap='seismic', vmin=vmin_vmax[0], vmax=vmin_vmax[1], aspect='auto')
    ax.set_title('BR assymetry map normalized, corrected with HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,14))
    label_size = 18
    ax.imshow(subtract_median_rows(arr_2D=BR_asymmetry_map_gaussian_binned_HRTSsub_qra_division), cmap='seismic', vmin=vmin_vmax[0], vmax=vmin_vmax[1], aspect='auto')
    ax.set_title('R/B normalized, corrected with HRTS: QR-A', fontsize=18)
    ax.set_aspect('auto')
    ax.set_xlabel('Longitude dimension (pixels)', fontsize=17)
    fig.supylabel('Latitude dimension (pixels)', fontsize=17)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
    contour_lower = ax.contour(intensitymap_croplat[::-1], levels=[lower_bound], colors='red', linewidths=2, extent=extent_sumer_binned_px_contours)
    contour_upper = ax.contour(intensitymap_croplat[::-1], levels=[upper_bound], colors='blue', linewidths=2, extent=extent_sumer_binned_px_contours)
    plt.show(block=False)








"""
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


"""
""



