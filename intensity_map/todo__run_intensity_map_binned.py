
# Primary inputs

# Binning
bin_lat = 4
bin_lon = 1

# Inputs: Percentage of the mean intensity of the spectral image, for contours. I use mean because it is approx the same for binned and not binned intensity map, but if I use max, they both are significantly different. 
percentage_mean_intensity_binned_SiII = [0., 60.]
percentage_mean_intensity_binned_NeVIII = [0., 60.]
percentage_mean_intensity_SiII = percentage_mean_intensity_binned_SiII
percentage_mean_intensity_NeVIII = percentage_mean_intensity_binned_NeVIII

# Pixel address
lat_img__indices_binned = 46, 156
lat_img__indices = bin_lat*lat_img__indices_binned[0], bin_lon*lat_img__indices_binned[1]

## Ranges of wavelength
wavelength_range_analysis = [1540.1, 1541.5] #Angstroem
wavelength_range_scalefactor = [1542., 1544.] #Angstroem

## Type of data of HRTS
type_hrts = 'A' #'A', 'B', or 'L'



## FWHM of the cold lines in SUMER:
fwhm_mean_weighted_sumer =  0.17740
fwhm_std_sumer =  0.02094
fwhm_unc_weighted_sumer =  0.00131
fwhm_synthetic_Si = 0.03
fwhm_sumer_to_convolve = fwhm_mean_weighted_sumer - fwhm_synthetic_Si
fwhm_to_convolve = fwhm_sumer_to_convolve #Usser can addapt this value
fwhm_to_convolve = 1.95*0.043
print('fwhm_to_convolve =', fwhm_to_convolve, '\u212B')


#####################################
# Create intensity maps of Si II and Ne VIII


# Si II
line_label = 'SiII' #'NeVIII', 'SiII', 'CIV', 'cold_line'
show_spectral_ranges = 'no'
show_intensity_maps = 'yes'
exec(open("create_intensitymap_binned_to_run.py").read())
intensitymap_croplat_SiII = np.copy(intensitymap_croplat)
intensitymap_croplat_binned_SiII = np.copy(intensitymap_croplat_binned)


# Ne VIII
line_label = 'NeVIII' #'NeVIII', 'SiII', 'CIV', 'cold_line'
show_spectral_ranges = 'yes'
show_intensity_maps = 'yes'
exec(open("create_intensitymap_binned_to_run.py").read())
intensitymap_croplat_NeVIII = np.copy(intensitymap_croplat)
intensitymap_croplat_binned_NeVIII = np.copy(intensitymap_croplat_binned)


#####################################
# Rest wavelength used
rest_wavelength_label = 'Peter_and_Judge_1999' #'SUMER_atlas', 'Peter_1998', 'Dammasch_1999', 'Peter_and_Judge_1999', 'Kelly_database'
lam_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][0] #Angstrom
lam_unc_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][1] #Angstrom
print('Rest wavelength Ne VIII (2nd order):', lam_0, r'$\pm$', lam_unc_0, '\u212B')

# uncertainty of the rest wavelength in km/s
v_unc_0 = vkms_doppler_unc(lamb=lam_0, lamb_unc=lam_unc_0, lamb_0=lam_0, lamb_0_unc=lam_unc_0) 

#####################################

# Average all spectra of the raster

N_img = len(spectralimage_interp_croplat_list)
N_rows = spectralimage_interp_croplat_list[0].shape[0]
N_cols_full = spectralimage_interp_croplat_list[0].shape[1] #or len(lam_sumer_full)

lam_sumer_avfullscan = lam_sumer_full

lam_sumer_avfullscan_crop, idx_sumer_crop = crop_range(list_to_crop=lam_sumer_avfullscan, range_values=wavelength_range_analysis)
N_cols_crop = idx_sumer_crop[1]-idx_sumer_crop[0]+1


rad_sumer_crop_avfullscan, erad_sumer_crop_avfullscan = np.zeros(N_cols_crop), np.zeros(N_cols_crop)
for img_ in tqdm(range(N_img)):
    for row_ in range(N_rows):
        
        ## 1) Take the data of the current spectral image and row (of SUMER)
        rad_sumer = spectralimage_interp_croplat_list[img_][row_, :]
        erad_sumer = unc_spectralimage_interp_croplat_list[img_][row_, :]
        ## 2) Crop 
        rad_sumer_crop = rad_sumer[idx_sumer_crop[0]:idx_sumer_crop[1]+1]
        erad_sumer_crop = erad_sumer[idx_sumer_crop[0]:idx_sumer_crop[1]+1]
        ## 3) Sum all spectra
        rad_sumer_crop_avfullscan = rad_sumer_crop_avfullscan + rad_sumer_crop
        erad_sumer_crop_avfullscan = erad_sumer_crop_avfullscan + erad_sumer_crop**2

# Wavelength
lam_sumer_avfullscan_crop = lam_sumer_avfullscan_crop
# Radiance not normalized
rad_sumer_crop_avfullscan = 1/(N_img * N_rows) * rad_sumer_crop_avfullscan
erad_sumer_crop_avfullscan = 1/(N_img * N_rows) * np.sqrt(erad_sumer_crop_avfullscan)


"""
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=lam_sumer_avfullscan_crop, y=rad_sumer_crop_avfullscan, yerr=erad_sumer_crop_avfullscan, color='black', linewidth=0.5, marker='.', markersize=10, label='SUMER average')
#ax.set_yscale('log')
ax.set_title(f'SUMER. Average all spectral profiles of the raster.', fontsize=18) 
ax.set_xlabel('Wavelength direction (\u212B)', color='black', fontsize=16)
ax.set_ylabel(r'Radiance [W/sr/m$^2$/''\u212B]', color='black', fontsize=16)
ax.axvline(lam_0, color='green', linewidth=1., label=f'Rest wavelength ({lam_0})'' \u212B')
ax.legend(fontsize=10)
plt.show(block=False)
"""








