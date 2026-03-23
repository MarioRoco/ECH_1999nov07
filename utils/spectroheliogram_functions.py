import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from utils.aux_functions import crop_range
import matplotlib.lines as mlines

#########################################################################
#########################################################################
# Conversion between pixels and wavelength (calibrated)
def pixels_to_wavelength(pixel, slope_cal, intercept_cal): 
    """
    This function converts the pixel numbers of the x axis (wavelength direction) in the image to calibrated wavelength values (in Angstrom). 
    """
    wavelength = slope_cal*pixel + intercept_cal
    return wavelength #[Angstrom]


def wavelength_to_pixels(wavelength, slope_cal, intercept_cal):
    """
    This function is the inverse of image_pixels_to_calibrated_wavelength()
    """
    m = 1/slope_cal
    b = -intercept_cal/slope_cal
    pixel = m * wavelength + b 
    return pixel


#################################
# Uncertainties

def pixels_to_wavelength_uncertainty(pixel, unc_pixel, slope_cal, unc_slope_cal, intercept_cal, unc_intercept_cal): 
    """
    This function converts the pixel numbers of the x axis (wavelength direction) in the image to calibrated wavelength values (in Angstrom). 
    """
    x = pixel
    dx = unc_pixel
    m = slope_cal
    dm = unc_slope_cal
    b = intercept_cal
    db = unc_intercept_cal
    
    unc_wavelength = np.sqrt( (m*dx)**2 + (x*dm)**2 + db**2 )
    return unc_wavelength #[Angstrom]


def wavelength_to_pixels_uncertainty(wavelength, unc_wavelength, slope_cal, unc_slope_cal, intercept_cal, unc_intercept_cal):
    """
    This function is the inverse of image_pixels_to_calibrated_wavelength()
    """
    x = wavelength
    dx = unc_wavelength
    m = slope_cal
    dm = unc_slope_cal
    b = intercept_cal
    db = unc_intercept_cal
    
    M = 1/m
    B = -b/m
    dM = abs(dm/M**2)
    dB = np.sqrt( (db/m)**2 + (b*dm/m**2)**2 )
    
    unc_pixel = np.sqrt( (M*dx)**2 + (x*dM)**2 + dB**2 )
    
    return unc_pixel

#########################################################################
#########################################################################
# Interpolate intensities in a given wavelength range

def unc_linear_interpolation_1point(x_interp, x_data, y_unc_data): #(float, list, list)
    """
    This function finds the uncertainty of an interpolant (y-axis uncertainty of the interpolated point).
    Inputs:
        - x_interp: independent variable whose y-value should be found using the interpolation. 
        - x_data: x-axis values to interpolate.
        - y_unc_data: y-axis uncertainties corresponding to the x_data values.
    """

    # Find the index where x_interp would fit in x_data_arr
    idx = np.searchsorted(x_data, x_interp)

    # Get the x1 and x2 values, and the corresponding y values
    x1 = x_data[idx - 1]
    x2 = x_data[idx]
    y1_unc = y_unc_data[idx - 1]
    y2_unc = y_unc_data[idx]
    
    # Calculate the y-uncertainty of the interpolated point "x_interp"
    x = x_interp
    A = (x-x2)/(x1-x2)*y1_unc
    B = (x-x1)/(x2-x1)*y2_unc
    y_unc_interp = np.sqrt(A**2 + B**2)
    return y_unc_interp


def unc_linear_interpolation_setpoints(x_interp_list, x_data, y_unc_data): #(list, list, list)
    """
    This function finds the uncertainties of a set of interpolants (y-axis uncertainties of the interpolated points).
    Inputs:
        - x_interp_list: independent variable whose y-values should be found using the interpolation. 
        - x_data: x-axis values to interpolate.
        - y_unc_data: y-axis uncertainties corresponding to the x_data values.
    """
    y_unc_interp_list = []
    for x_interp_i in x_interp_list:
        y_unc_interp_i = unc_linear_interpolation_1point(x_interp=x_interp_i, x_data=x_data, y_unc_data=y_unc_data)
        y_unc_interp_list.append(y_unc_interp_i)
    y_unc_interp_arr = np.array(y_unc_interp_list)
    return y_unc_interp_arr


def range__interpolation_artificial_data(wavelength_range, N_interp_range, intensity_allcols, unc_intensity_allcols, slope_cal, intercept_cal):
    columns_indices = np.arange(0, len(intensity_allcols))
    
    # Convert masked values to NaNs:
    intensity_allcols = np.ma.filled(intensity_allcols, np.nan)
    unc_intensity_allcols = np.ma.filled(unc_intensity_allcols, np.nan)

    # Convert pixel indices to wavelength
    pixel_centerwavelength_allcols = pixels_to_wavelength(pixel=columns_indices, slope_cal=slope_cal, intercept_cal=intercept_cal)

    # Edges of the pixels
    pixel_halfwidth = slope_cal/2
    pixel_low = pixel_centerwavelength_allcols - pixel_halfwidth
    pixel_high = pixel_centerwavelength_allcols + pixel_halfwidth

    # Inside the range
    column_index_edgelow = np.argmax(pixel_high > wavelength_range[0]) # find the first (column) index where element (wavelength) is greater than wavelength_range[0]
    column_index_edgehigh = np.argmax(pixel_high > wavelength_range[1]) # find the first (column) index where element (wavelength) is smaller than wavelength_range[1]

    # Select the points that we will interpolate, adding one more dot at each side of the range
    il = column_index_edgelow
    ih = column_index_edgehigh
    indices_interpolate = np.concatenate([[il-1], np.arange(il,ih+1), [ih+1]]) 
    pixel_centerwavelength_interpolate = pixel_centerwavelength_allcols[indices_interpolate]
    pixel_intensity_interpolate = intensity_allcols[indices_interpolate]
    unc_pixel_intensity_interpolate = unc_intensity_allcols[indices_interpolate]

    # Create an interpolation function
    interp_func = interp1d(pixel_centerwavelength_interpolate, pixel_intensity_interpolate, kind='linear')
    
    # Interpolated points ("artificial" data)
    interp_wavelength = np.linspace(wavelength_range[0], wavelength_range[1], N_interp_range)
    interp_intensity = interp_func(interp_wavelength)
    """
    artificial_centerpixel_left = np.mean([wavelength_range[0], (pixel_centerwavelength_interpolate[0] - pixel_halfwidth)])
    artificial_centerpixel_right = np.mean([wavelength_range[1], (pixel_centerwavelength_interpolate[-1] + pixel_halfwidth)])
    interp_wavelength = np.concatenate([[artificial_centerpixel_left], pixel_centerwavelength_interpolate[1:-1], [artificial_centerpixel_right]])
    interp_intensity = interp_func(interp_wavelength)
    """
    
    # Uncertainties
    interp_intensity_unc = unc_linear_interpolation_setpoints(x_interp_list=interp_wavelength, x_data=pixel_centerwavelength_interpolate, y_unc_data=unc_pixel_intensity_interpolate)
    
    return [interp_wavelength, interp_intensity, interp_intensity_unc]


def range__interpolation_artificial_data_pixelfixed(wavelength_range, N_interp_range, intensity_allcols, unc_intensity_allcols, slope_cal, intercept_cal):
    columns_indices = np.arange(0, len(intensity_allcols))
    
    # Convert masked values to NaNs:
    intensity_allcols = np.ma.filled(intensity_allcols, np.nan)
    unc_intensity_allcols = np.ma.filled(unc_intensity_allcols, np.nan)

    # Convert pixel indices to wavelength
    pixel_centerwavelength_allcols = pixels_to_wavelength(pixel=columns_indices, slope_cal=slope_cal, intercept_cal=intercept_cal)

    # Edges of the pixels
    pixel_halfwidth = slope_cal/2
    pixel_low = pixel_centerwavelength_allcols - pixel_halfwidth
    pixel_high = pixel_centerwavelength_allcols + pixel_halfwidth

    # Inside the range
    column_index_edgelow = np.argmax(pixel_high > wavelength_range[0]) # find the first (column) index where element (wavelength) is greater than wavelength_range[0]
    column_index_edgehigh = np.argmax(pixel_high > wavelength_range[1]) # find the first (column) index where element (wavelength) is smaller than wavelength_range[1]

    # Select the points that we will interpolate, adding one more dot at each side of the range
    il = column_index_edgelow
    ih = column_index_edgehigh
    indices_interpolate = np.concatenate([[il-1], np.arange(il,ih+1), [ih+1]]) 
    pixel_centerwavelength_interpolate = pixel_centerwavelength_allcols[indices_interpolate]
    pixel_intensity_interpolate = intensity_allcols[indices_interpolate]
    unc_pixel_intensity_interpolate = unc_intensity_allcols[indices_interpolate]

    # Create an interpolation function
    interp_func = interp1d(pixel_centerwavelength_interpolate, pixel_intensity_interpolate, kind='linear')
    
    # Interpolated points ("artificial" data)
    interp_wavelength = np.linspace(wavelength_range[0], wavelength_range[1], N_interp_range)
    interp_intensity = interp_func(interp_wavelength)
    
    # Uncertainties
    interp_intensity_unc = unc_linear_interpolation_setpoints(x_interp_list=interp_wavelength, x_data=pixel_centerwavelength_interpolate, y_unc_data=unc_pixel_intensity_interpolate)
    
    return [interp_wavelength, interp_intensity, interp_intensity_unc]


    
def range__interpolation_average_intensity(wavelength_range, intensity_allcols, slope_cal, intercept_cal):
    
    columns_indices = np.arange(0, len(intensity_allcols))
    
    # Convert masked values to NaNs:
    intensity_allcols = np.ma.filled(intensity_allcols, np.nan)

    # Convert pixel indices to wavelength
    pixel_centerwavelength_allcols = pixels_to_wavelength(pixel=columns_indices, slope_cal=slope_cal, intercept_cal=intercept_cal)

    # Edges of the pixels
    pixel_halfwidth = slope_cal/2
    pixel_low = pixel_centerwavelength_allcols - pixel_halfwidth
    pixel_high = pixel_centerwavelength_allcols + pixel_halfwidth

    # Inside the range
    column_index_edgelow = np.argmax(pixel_high > wavelength_range[0]) # find the first (column) index where element (wavelength) is greater than wavelength_range[0]
    column_index_edgehigh = np.argmax(pixel_high > wavelength_range[1]) # find the first (column) index where element (wavelength) is smaller than wavelength_range[1]

    # Select the points that we will interpolate, adding one more dot at each side of the range
    il = column_index_edgelow
    ih = column_index_edgehigh
    indices_interpolate = np.concatenate([[il-1], np.arange(il,ih+1), [ih+1]]) 
    pixel_centerwavelength_interpolate = pixel_centerwavelength_allcols[indices_interpolate]
    pixel_intensity_interpolate = intensity_allcols[indices_interpolate]

    # Create an interpolation function
    interp_func = interp1d(pixel_centerwavelength_interpolate, pixel_intensity_interpolate, kind='linear')
    
    # Adding the partial pixels of the edge of the range
    pixel_centerwavelength_inrange = pixel_centerwavelength_interpolate[2:-2]
    pixel_halfwidth = slope_cal/2
    wavelength_analysis_left = np.mean([wavelength_range[0], pixel_centerwavelength_inrange[0]-pixel_halfwidth])
    wavelength_analysis_right = np.mean([wavelength_range[1], pixel_centerwavelength_inrange[-1]+pixel_halfwidth])
    wavelength_analysis = np.concatenate([[wavelength_analysis_left], pixel_centerwavelength_inrange, [wavelength_analysis_right]])
    intensity_analysis = interp_func(wavelength_analysis)
    
    # Delta pixels (of the range)
    wavelength_deltapixel_inrange_entirepixels = slope_cal * np.ones(len(pixel_centerwavelength_inrange))
    wavelength_deltapixel_left = wavelength_range[0] - (pixel_centerwavelength_inrange[0]-pixel_halfwidth)
    wavelength_deltapixel_right = (pixel_centerwavelength_inrange[-1]+pixel_halfwidth) - wavelength_range[1]
    wavelength_deltapixel_inrange = np.concatenate([[wavelength_deltapixel_left], wavelength_deltapixel_inrange_entirepixels, [wavelength_deltapixel_right]])

    # Return: weighted mean of the intensity
    # Mask the NaN values (otherwise the result is a nan). If the input has masked values, they are converted to NaN values, so here we need to mask them.
    mask = np.isnan(intensity_analysis)  # Create a boolean mask where ia has NaNs
    intensity_analysis_masked = intensity_analysis[~mask]  # Select elements of ia where mask is False (i.e., not NaN)
    wavelength_deltapixel_inrange_masked = wavelength_deltapixel_inrange[~mask]  # Similarly filter the weights
    w, ia = wavelength_deltapixel_inrange_masked, intensity_analysis_masked
    intensity_weighted_mean = np.sum(w*ia) / np.sum(w)
    
    return intensity_weighted_mean

def range__interpolation_average_intensity_AND_uncertainty(wavelength_range, intensity_allcols, unc_intensity_allcols, slope_cal, intercept_cal):
    
    columns_indices = np.arange(0, len(intensity_allcols))
    
    # Convert masked values to NaNs:
    intensity_allcols = np.ma.filled(intensity_allcols, np.nan)
    unc_intensity_allcols = np.ma.filled(unc_intensity_allcols, np.nan)

    # Convert pixel indices to wavelength
    pixel_centerwavelength_allcols = pixels_to_wavelength(pixel=columns_indices, slope_cal=slope_cal, intercept_cal=intercept_cal)

    # Edges of the pixels
    pixel_halfwidth = slope_cal/2
    pixel_low = pixel_centerwavelength_allcols - pixel_halfwidth
    pixel_high = pixel_centerwavelength_allcols + pixel_halfwidth

    # Inside the range
    column_index_edgelow = np.argmax(pixel_high > wavelength_range[0]) # find the first (column) index where element (wavelength) is greater than wavelength_range[0]
    column_index_edgehigh = np.argmax(pixel_high > wavelength_range[1]) # find the first (column) index where element (wavelength) is smaller than wavelength_range[1]

    # Select the points that we will interpolate, adding one more dot at each side of the range
    il = column_index_edgelow
    ih = column_index_edgehigh
    indices_interpolate = np.concatenate([[il-1], np.arange(il,ih+1), [ih+1]]) 
    pixel_centerwavelength_interpolate = pixel_centerwavelength_allcols[indices_interpolate]
    pixel_intensity_interpolate = intensity_allcols[indices_interpolate]
    unc_pixel_intensity_interpolate = unc_intensity_allcols[indices_interpolate]

    # Create an interpolation function
    interp_func = interp1d(pixel_centerwavelength_interpolate, pixel_intensity_interpolate, kind='linear')
    
    # Adding the partial pixels of the edge of the range
    pixel_centerwavelength_inrange = pixel_centerwavelength_interpolate[2:-2]
    unc_pixel_intensity_inrange = unc_pixel_intensity_interpolate[2:-2]
    pixel_halfwidth = slope_cal/2
    wavelength_analysis_left = np.mean([wavelength_range[0], pixel_centerwavelength_inrange[0]-pixel_halfwidth])
    wavelength_analysis_right = np.mean([wavelength_range[1], pixel_centerwavelength_inrange[-1]+pixel_halfwidth])
    wavelength_analysis = np.concatenate([[wavelength_analysis_left], pixel_centerwavelength_inrange, [wavelength_analysis_right]])
    intensity_analysis = interp_func(wavelength_analysis)
    
    # Uncertainties
    unc_left = unc_linear_interpolation_1point(x_interp=wavelength_analysis_left, x_data=pixel_centerwavelength_interpolate, y_unc_data=unc_pixel_intensity_interpolate)
    unc_right = unc_linear_interpolation_1point(x_interp=wavelength_analysis_right, x_data=pixel_centerwavelength_interpolate, y_unc_data=unc_pixel_intensity_interpolate)
    unc_intensity_analysis = np.concatenate([[unc_left], unc_pixel_intensity_inrange, [unc_right]])

    # Delta pixels (of the range)
    wavelength_deltapixel_inrange_entirepixels = slope_cal * np.ones(len(pixel_centerwavelength_inrange))
    wavelength_deltapixel_left = wavelength_range[0] - (pixel_centerwavelength_inrange[0]-pixel_halfwidth)
    wavelength_deltapixel_right = (pixel_centerwavelength_inrange[-1]+pixel_halfwidth) - wavelength_range[1]
    wavelength_deltapixel_inrange = np.concatenate([[wavelength_deltapixel_left], wavelength_deltapixel_inrange_entirepixels, [wavelength_deltapixel_right]])

    # Return: weighted mean of the intensity
    # Mask the NaN values (otherwise the result is a nan). If the input has masked values, they are converted to NaN values, so here we need to mask them.
    mask = np.isnan(intensity_analysis)  # Create a boolean mask where ia has NaNs
    intensity_analysis_masked = intensity_analysis[~mask]  # Select elements of ia where mask is False (i.e., not NaN)
    unc_intensity_analysis_masked = unc_intensity_analysis[~mask]  #uncertainty
    wavelength_deltapixel_inrange_masked = wavelength_deltapixel_inrange[~mask]  # Similarly filter the weights
    w, ia, uia = wavelength_deltapixel_inrange_masked, intensity_analysis_masked, unc_intensity_analysis_masked
    intensity_weighted_mean = np.sum(w*ia) / np.sum(w)
    intensity_unc_weighted_mean = np.sqrt( np.sum( (w*uia)**2 ) ) / np.sum(w) #uncertainty
    
    return [intensity_weighted_mean, intensity_unc_weighted_mean]

    



#########################################################################
#########################################################################
# Conversion between wavelength range and closest range of pixels

def range__wavelength_to_closest_pixels(wavelength_range, slope_cal, intercept_cal):
    """
    Convert calibrated wavelength range to closest pixel range.
    """
    # Calculate the pixel range (floats)
    pixel_range0_float = wavelength_to_pixels(wavelength=wavelength_range[0], slope_cal=slope_cal, intercept_cal=intercept_cal)
    pixel_range1_float = wavelength_to_pixels(wavelength=wavelength_range[1], slope_cal=slope_cal, intercept_cal=intercept_cal)
    pixel_range_float = [pixel_range0_float, pixel_range1_float]
    
    # Round and convert to integer 
    pixel_range0_int = int(np.round(pixel_range0_float))
    pixel_range1_int = int(np.round(pixel_range1_float))
    pixel_range_pixcenter =  [pixel_range0_int,  pixel_range1_int]
    
    # Calculate the range of wavelength calibrated from the center of the pixels range
    wavelength_range0_pixcenter = pixels_to_wavelength(pixel=float(pixel_range0_int), slope_cal=slope_cal, intercept_cal=intercept_cal)
    wavelength_range1_pixcenter = pixels_to_wavelength(pixel=float(pixel_range1_int), slope_cal=slope_cal, intercept_cal=intercept_cal)
    wavelength_range_pixcenter = [wavelength_range0_pixcenter, wavelength_range1_pixcenter]
    
    return [wavelength_range_pixcenter, pixel_range_pixcenter, pixel_range_float]

def range__wavelength_to_closest_pixels_pixelfixed(wavelength_low, N_pixels, slope_cal, intercept_cal):
    """
    Convert calibrated wavelength range to first closest pixel of the range, but the number of pixels is fixed (you chose it in the input).
    """
    # Range of pixels
    ## 1) Calculate the first pixel of the range (float)
    pixel_start_float = wavelength_to_pixels(wavelength=wavelength_low, slope_cal=slope_cal, intercept_cal=intercept_cal)
    pixel_range_float =  [pixel_start_float, pixel_start_float+N_pixels-1]
    ## 2) Round and convert to integer 
    pixel_start = int(np.round(pixel_start_float))
    pixel_range_pixcenter =  [pixel_start, pixel_start+N_pixels-1]
    
    # Calculate the range of wavelength calibrated from the center of the pixels range
    wavelength_range0_pixcenter = pixels_to_wavelength(pixel=float(pixel_range_pixcenter[0]), slope_cal=slope_cal, intercept_cal=intercept_cal)
    wavelength_range1_pixcenter = pixels_to_wavelength(pixel=float(pixel_range_pixcenter[1]), slope_cal=slope_cal, intercept_cal=intercept_cal)
    wavelength_range_pixcenter = [wavelength_range0_pixcenter, wavelength_range1_pixcenter]
    
    return [wavelength_range_pixcenter, pixel_range_pixcenter, pixel_range_float]



#########################################################################
#########################################################################
# Average profiles from spectroheliogram pixels of an intensity range

def get_profiles_from__range__wavelength_to_closest_pixels_pixelfixed(wavelength_range, spectra_croplat_list, unc_spectra_croplat_list, rows_cols_of_spectroheliogram_croplat, slope_list, intercept_list):
    """
    Each spectrum array of the list, each spectrum uncertainty array of the list, and the spectroheliogram should be cropped in latitude according to the lists "slope_list" and "intercept_list".
    """
    # Calculate number of pixels of each profile
    pixel_low = wavelength_to_pixels(wavelength=wavelength_range[0], slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
    pixel_high = wavelength_to_pixels(wavelength=wavelength_range[1], slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
    N_pixels_range = int(round(pixel_high - pixel_low))
    
    profile_intensity_line_sum = np.zeros(N_pixels_range)
    profile_intensity_unc_line_sumsq = np.zeros(N_pixels_range)
    N_pixels_addresses = 0
    for i, [row_lat, idx_spec] in enumerate(rows_cols_of_spectroheliogram_croplat): #in the spectroheliogram, the rows correspond to latitude, and the columns correspond to the spectrum of the list of spectra
        
        # Calculate the range of pixels and Angstrom in the wavelength direction for each row
        rwcppf = range__wavelength_to_closest_pixels_pixelfixed(wavelength_low=wavelength_range[0], N_pixels=N_pixels_range, slope_cal=slope_list[row_lat], intercept_cal=intercept_list[row_lat])
        pixel_range_pixcenter = rwcppf[1]
        col0 = pixel_range_pixcenter[0]
        col1 = pixel_range_pixcenter[1]
                
        # Calculate profile of each row
        spectrum_array = spectra_croplat_list[idx_spec]
        spectrum_array_unc = unc_spectra_croplat_list[idx_spec]
        profile_intensity_line = spectrum_array[row_lat, col0:col1+1]
        profile_intensity_unc_line = spectrum_array_unc[row_lat, col0:col1+1]
        
        if not profile_intensity_line.mask.any(): # Process the array if there are no NaNs 
            profile_intensity_line_sum = profile_intensity_line_sum + profile_intensity_line
            profile_intensity_unc_line_sumsq = profile_intensity_unc_line_sumsq + profile_intensity_unc_line**2
            N_pixels_addresses+=1
        else: continue  # Skip the array if it contains NaNs
    
    profile_wavelength_line = pixels_to_wavelength(pixel=np.arange(col0, col1+1), slope_cal=slope_list[row_lat], intercept_cal=intercept_list[row_lat]) #From the last iteration
    profile_intensity_line_average = (1/N_pixels_addresses) * profile_intensity_line_sum
    profile_intensity_unc_line_average = (1/N_pixels_addresses) * np.sqrt(profile_intensity_unc_line_sumsq)
    
    return [profile_wavelength_line, profile_intensity_line_average, profile_intensity_unc_line_average]

    
def get_profiles_from__range__interpolation_artificial_data(wavelength_range, spectra_croplat_list, unc_spectra_croplat_list, rows_cols_of_spectroheliogram_croplat, slope_list, intercept_list, N_pixels_range='auto'):
    """
    Each spectrum array of the list, each spectrum uncertainty array of the list, and the spectroheliogram should be cropped in latitude according to the lists "slope_list" and "intercept_list".
    """
    if N_pixels_range=='auto':
        # Calculate number of pixels of each profile
        pixel_low = wavelength_to_pixels(wavelength=wavelength_range[0], slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
        pixel_high = wavelength_to_pixels(wavelength=wavelength_range[1], slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
        N_pixels_range = int(round(pixel_high - pixel_low))
    
    profile_intensity_line_sum = np.zeros(N_pixels_range)
    profile_intensity_unc_line_sumsq = np.zeros(N_pixels_range)
    N_pixels_addresses = 0
    for i, [row_lat, idx_spec] in enumerate(rows_cols_of_spectroheliogram_croplat): #in the spectroheliogram, the rows correspond to latitude, and the columns correspond to the spectrum of the list of spectra
        
        spectrum_array = spectra_croplat_list[idx_spec]
        spectrum_array_unc = unc_spectra_croplat_list[idx_spec]
        intensity_allcols = spectrum_array[row_lat, :]
        unc_intensity_allcols = spectrum_array_unc[row_lat, :]
        
        # Calculate profile of each row
        iad = range__interpolation_artificial_data(wavelength_range=wavelength_range, N_interp_range=N_pixels_range, intensity_allcols=intensity_allcols, unc_intensity_allcols=unc_intensity_allcols, slope_cal=slope_list[row_lat], intercept_cal=intercept_list[row_lat])
        profile_intensity_line = iad[1]
        profile_intensity_unc_line = iad[2]
        
        # Calculate the average of all the profiles
        if not np.isnan(profile_intensity_line).any(): # Process the array if there are no NaNs 
            profile_intensity_line_sum = profile_intensity_line_sum + profile_intensity_line
            profile_intensity_unc_line_sumsq = profile_intensity_unc_line_sumsq + profile_intensity_unc_line**2
            N_pixels_addresses+=1
        else: continue  # Skip the array if it contains NaNs
    
    profile_wavelength_line = iad[0] #From the last iteration
    profile_intensity_line_average = (1/N_pixels_addresses) * profile_intensity_line_sum
    profile_intensity_unc_line_average = (1/N_pixels_addresses) * np.sqrt(profile_intensity_unc_line_sumsq)
    
    return [profile_wavelength_line, profile_intensity_line_average, profile_intensity_unc_line_average]



#########################################################################
#########################################################################
# Create spectroheliogram

def crop_spectra_list(sumer_data_list, sumer_data_unc_list, slit_top_px, slit_bottom_px):
    sumer_data_list_croplat, sumer_data_unc_list_croplat = [],[]
    for i in range(len(sumer_data_list)):
        sumer_data_list_croplat.append(sumer_data_list[i][slit_top_px:slit_bottom_px+1, :])
        sumer_data_unc_list_croplat.append(sumer_data_unc_list[i][slit_top_px:slit_bottom_px+1, :])
    return [sumer_data_list_croplat, sumer_data_unc_list_croplat]


def create_spectroheliogram_from__range__wavelength_to_closest_pixels(spectra_croplat_list, wavelength_range, wavelength_range_bckg, slope_list, intercept_list):
    
    spectroheliogram_line_NT, spectroheliogram_bckg_NT, spectroheliogram_line_nobg_NT  = [],[],[]
    for spectrum_array in spectra_croplat_list:
        
        average_intensity_line_1col, average_intensity_bckg_1col, average_intensity_line_nobg_1col = [],[],[]
        for row_latcrop in range(len(slope_list)): #if the spectra are cropped in latitude, otherwise the index would be "i", and "row_croplat" would be row_list[i]"

            # Calculate the range of pixels and Angstrom in the wavelength direction for each row
            pixel_range_pixcenter = range__wavelength_to_closest_pixels(wavelength_range=wavelength_range, slope_cal=slope_list[row_latcrop], intercept_cal=intercept_list[row_latcrop])[1]
            pixel_range_pixcenter_bckg = range__wavelength_to_closest_pixels(wavelength_range=wavelength_range_bckg, slope_cal=slope_list[row_latcrop], intercept_cal=intercept_list[row_latcrop])[1]

            # Rows and columns indices (of the boundaries) for each row
            col0 = pixel_range_pixcenter[0]
            col1 = pixel_range_pixcenter[1]
            colbg0 = pixel_range_pixcenter_bckg[0]
            colbg1 = pixel_range_pixcenter_bckg[1]

            # Calculate profile of each row
            profile_intensity_line = spectrum_array[row_latcrop, col0:col1+1]
            N_line = col1-col0+1
            profile_intensity_bckg = spectrum_array[row_latcrop, colbg0:colbg1+1]
            N_bckg = colbg1-colbg0+1

            # Calculate the average of the profile of each row
            average_intensity_line_1row1col = np.mean(profile_intensity_line)
            average_intensity_bckg_1row1col = np.mean(profile_intensity_bckg)
            average_intensity_line_nobg_1row1col = average_intensity_line_1row1col - average_intensity_bckg_1row1col
            ## Save in lists
            average_intensity_line_1col.append(average_intensity_line_1row1col)
            average_intensity_bckg_1col.append(average_intensity_bckg_1row1col)
            average_intensity_line_nobg_1col.append(average_intensity_line_nobg_1row1col)
        
        # save the above lists (which represent the columns) in lists to crate the 2D-array of the spectroheliogram
        spectroheliogram_line_NT.append(average_intensity_line_1col) # Line with background (background not subtracted)
        spectroheliogram_bckg_NT.append(average_intensity_bckg_1col) # Background 
        spectroheliogram_line_nobg_NT.append(average_intensity_line_nobg_1col) # Line without background (background subtracted)
        
    # Convert to array and transpose
    spectroheliogram_line = np.array(spectroheliogram_line_NT).T # Line with background (background not subtracted)
    spectroheliogram_bckg = np.array(spectroheliogram_bckg_NT).T # Background 
    spectroheliogram_line_nobg = np.array(spectroheliogram_line_nobg_NT).T # Line without background (background subtracted)
    
    return [spectroheliogram_line, spectroheliogram_bckg, spectroheliogram_line_nobg]


def create_spectroheliogram_from__range__interpolation_average_intensity(spectra_croplat_list, wavelength_range, wavelength_range_bckg, slope_list, intercept_list):
    
    spectroheliogram_line_NT, spectroheliogram_bckg_NT, spectroheliogram_line_nobg_NT  = [],[],[]
    for spectrum_array in spectra_croplat_list:
    
        average_intensity_line_1col, average_intensity_bckg_1col, average_intensity_line_nobg_1col = [],[],[]
        for row_latcrop in range(len(slope_list)): #if the spectra are cropped in latitude, otherwise the index would be "i", and "row_croplat" would be row_list[i]"

            # Calculate the average of the profile of each row
            intensity_allcols = spectrum_array[row_latcrop, :]
            average_intensity_line_1row1col = range__interpolation_average_intensity(wavelength_range=wavelength_range, intensity_allcols=intensity_allcols, slope_cal=slope_list[row_latcrop], intercept_cal=intercept_list[row_latcrop]) # Line with background (background not subtracted)
            average_intensity_bckg_1row1col = range__interpolation_average_intensity(wavelength_range=wavelength_range_bckg, intensity_allcols=intensity_allcols, slope_cal=slope_list[row_latcrop], intercept_cal=intercept_list[row_latcrop]) # Background 
            average_intensity_line_nobg_1row1col = average_intensity_line_1row1col - average_intensity_bckg_1row1col # Line without background (background subtracted)

            # Save in lists
            average_intensity_line_1col.append(average_intensity_line_1row1col) # Line with background (background not subtracted)
            average_intensity_bckg_1col.append(average_intensity_bckg_1row1col) # Background 
            average_intensity_line_nobg_1col.append(average_intensity_line_nobg_1row1col) # Line without background (background subtracted)
        
        # save the above lists (which represent the columns) in lists to crate the 2D-array of the spectroheliogram
        spectroheliogram_line_NT.append(average_intensity_line_1col) # Line with background (background not subtracted)
        spectroheliogram_bckg_NT.append(average_intensity_bckg_1col) # Background 
        spectroheliogram_line_nobg_NT.append(average_intensity_line_nobg_1col) # Line without background (background subtracted)
        
    # Convert to array and transpose
    spectroheliogram_line = np.array(spectroheliogram_line_NT).T # Line with background (background not subtracted)
    spectroheliogram_bckg = np.array(spectroheliogram_bckg_NT).T # Background 
    spectroheliogram_line_nobg = np.array(spectroheliogram_line_nobg_NT).T # Line without background (background subtracted)
    
    return [spectroheliogram_line, spectroheliogram_bckg, spectroheliogram_line_nobg]


#########################################################################
#########################################################################
# Pixel addresses from EIT to SUMER spectroheliogram

def remove_repeated_pairs(list_of_pairs):
    list_of_pairs_nonrepeated = list({tuple(pair) for pair in list_of_pairs})
    return list_of_pairs_nonrepeated

def remove_pairs_with_impossible_values(list_of_pairs_to_filter, row_lowest, row_highest, col_lowest, col_highest):
    """
    This function filter the list of tuples (2 items each), so that:
        - Remove tuples whose first item is lower than row_lowest.
        - Remove tuples whose first item is higher than row_highest.
        - Remove tuples whose second item is lower than col_lowest.
        - Remove tuples whose second item is higher than col_highest.
    """
    filtered_tuples = [pt for pt in list_of_pairs_to_filter if row_lowest<=pt[0]<=row_highest and col_lowest<=pt[1]<=col_highest]
    return filtered_tuples


# Final function
def pixel_addresses_EIT_to_SUMER(rows_cols_eit, image_sumer, image_eit, X_HP_rotcomp, slit_top_px, slit_bottom_px, header_eit, header_sumer):
    rows_cols_sumer_conc = pixel_address_EIT_to_SUMER_delta_list(rows_cols_eit=rows_cols_eit, image_sumer=spectroheliogram_line_nobg, image_eit=data_eit_crop, X_HP_rotcomp=s3_HPlon_fullraster_rotcomp, slit_top_px=slit_top_px, slit_bottom_px=slit_bottom_px, header_eit=header_eit, header_sumer=header_sumer)
    rows_cols_sumer_nonrepeated = remove_repeated_pairs(list_of_pairs=rows_cols_sumer_conc)
    return rows_cols_sumer_nonrepeated
    

##############################################
# EIT to SUMER pixels


def get_bounds(intensitymap_croplat, range_percentage, threshold_value_type='max'):
    """
    Select the threshold value and the lower and upper bounds for the contours and average of profiles
    Variable threshold_value_type should be 'mean', 'max', 'min', or 'median'.
    """
    if threshold_value_type == 'mean': threshold_value = np.nanmean(intensitymap_croplat)
    elif threshold_value_type == 'max': threshold_value = np.nanmax(intensitymap_croplat)
    elif threshold_value_type == 'min': threshold_value = np.nanmin(intensitymap_croplat)
    elif threshold_value_type == 'median': threshold_value = np.nanmedian(intensitymap_croplat)
    
    lower_bound = range_percentage[0]/100. * threshold_value
    upper_bound = range_percentage[1]/100. * threshold_value

    return [lower_bound, upper_bound]
    
def get_bound(intensitymap_croplat, percentage_, threshold_value_type='max'):
    """
    Select the threshold value and the lower and upper bounds for the contours and average of profiles
    Variable threshold_value_type should be 'mean', 'max', 'min', or 'median'.
    """
    if threshold_value_type == 'mean': threshold_value = np.nanmean(intensitymap_croplat)
    elif threshold_value_type == 'max': threshold_value = np.nanmax(intensitymap_croplat)
    elif threshold_value_type == 'min': threshold_value = np.nanmin(intensitymap_croplat)
    elif threshold_value_type == 'median': threshold_value = np.nanmedian(intensitymap_croplat)
    
    bound__ = percentage_/100. * threshold_value

    return bound__



def range_intensity_addresses_of_SUMER_spectroheliogram(intensitymap_croplat, lower_bound, upper_bound, slit_top_px):
    
    rowscols_inside_range_arr = np.argwhere((intensitymap_croplat>=lower_bound) & (intensitymap_croplat<=upper_bound))
    
    # convert to list
    rowscols_inside_range = []
    for [row_, col_] in rowscols_inside_range_arr:
        rowscols_inside_range.append([row_+slit_top_px, col_]) 
        # row_ correspond to the cropped array, we sum slit_top_px because the data for the average spectra (the spectral images) are not cropped in y axis (latitude)

    return rowscols_inside_range #use for the the list of spectral images not cropped in latitude 

def convert_list_of_pairs_to_2_lists(list_of_pairs):
    y_row_list, x_col_list = [],[]
    for [row_, col_] in list_of_pairs:
        y_row_list.append(row_) 
        x_col_list.append(col_) 
    y_row_list = np.array(y_row_list)
    x_col_list = np.array(x_col_list)
    return [y_row_list, x_col_list]


def map_pixels_array1_to_array2(arr1, arr2, pixel_address_list_1):
    import math
    nrows1, ncols1 = arr1.shape
    nrows2, ncols2 = arr2.shape

    row_scale = nrows2 / nrows1
    col_scale = ncols2 / ncols1
    
    mapped_list = []
    for r1, c1 in pixel_address_list_1:
        row_start = max(0, math.floor(r1 * row_scale))
        row_end = min(nrows2 - 1, math.ceil((r1 + 1) * row_scale) - 1)
        col_start = max(0, math.floor(c1 * col_scale))
        col_end = min(ncols2 - 1, math.ceil((c1 + 1) * col_scale) - 1)
        for r2 in range(row_start, row_end + 1):
            for c2 in range(col_start, col_end + 1):
                mapped_list.append((r2, c2))

    # Remove duplicates
    mapped_list_unrepeated = list(set(mapped_list))
                    
    return mapped_list_unrepeated

#########################################################################
#########################################################################
# 

#####################################
# Pixel addresses, range of intensity

def range_intensity_addresses_of_image(range_percentage, image_array, threshold_value='max'):
    """
    This function extracts all the addresses (row index, column index) of the pixels whose intensities are between a certain range of percentage (given in the input) of the image's maximum intensity. 
    """
    proportion_range0 = range_percentage[0]/100.
    proportion_range1 = range_percentage[1]/100.
    
    if threshold_value=='max': thr_val = np.max(image_array) #maximum of the intensity map
    elif threshold_value=='average' or threshold_value=='mean': thr_val = np.mean(image_array) # minimum of the intensity map
    else: thr_val = threshold_value # value defined by the user
    
    lower_bound = proportion_range0 * thr_val
    upper_bound = proportion_range1 * thr_val
    rowscols_inside_range_arr = np.argwhere((image_array>=lower_bound) & (image_array<=upper_bound))
    
    # convert to list
    rowscols_inside_range = []
    for [row, col] in rowscols_inside_range_arr:
        rowscols_inside_range.append([row, col])
    
    return [rowscols_inside_range, lower_bound, upper_bound] #[[lower percentage, upper percentage], float, float] of the maximum intensity    




#####################################
# Plot intensity map, contours, highlights...

def plot_intensitymap_NeVIII(intensity_map, vmin_sumer=8e-3, vmax_sumer=1e-0, title='auto'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(intensity_map, cmap='Greys_r', aspect='auto', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
    cax = fig.add_axes([0.9, 0.05, 0.03, 0.90])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Intensity [W/sr/m^2]', fontsize=16)
    if title=='auto': ax.set_title(f'SOHO/SUMER intensity map of Ne VIII 770.428 \u212B', fontsize=18)
    else: ax.set_title(title, fontsize=18)
    ax.set_xlabel('Wavelength dimension (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial dimension (pixels)', color='black', fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)




def plot_intensitymap_NeVIII_with_contours(intensity_map, lower_bound, upper_bound, vmin_sumer=8e-3, vmax_sumer=1e-0, title='auto'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(intensity_map, cmap='Greys_r', aspect='auto', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
    cax = fig.add_axes([0.9, 0.05, 0.03, 0.90])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Intensity [W/sr/m^2]', fontsize=16)
    if title=='auto': ax.set_title(f'SOHO/SUMER intensity map of Ne VIII 770.428 \u212B', fontsize=18)
    else: ax.set_title(title, fontsize=18)
    ax.set_xlabel('Wavelength dimension (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial dimension (pixels)', color='black', fontsize=16)

    contour_lower = ax.contour(intensity_map, levels=[lower_bound], colors='cyan', linewidths=2)
    contour_upper = ax.contour(intensity_map, levels=[upper_bound], colors='blue', linewidths=2)

    legend_elements = [
        mlines.Line2D([],[],color='cyan', label=f'{lower_bound}'),
        mlines.Line2D([],[],color='blue', label=f'{upper_bound}')
    ]

    ax.legend(handles=legend_elements)#, title=f'')
    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)



def plot_intensitymap_NeVIII_with_contours_and_highlights_v2(intensity_map, lower_bound, upper_bound, vmin_sumer=8e-3, vmax_sumer=1e-0, title='auto', contours_highlights_both='highlights'):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    # Show image
    img = ax.imshow(intensity_map, cmap='Greys_r', aspect='auto', norm=LogNorm(vmin=vmin_sumer, vmax=vmax_sumer))
    cax = fig.add_axes([0.9, 0.05, 0.03, 0.90])
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Intensity [W/sr/m^2]', fontsize=16)
    
    if contours_highlights_both=='highlights' or contours_highlights_both=='both':
    
        # Highlight pixels within thresholds
        mask = (intensity_map >= lower_bound) & (intensity_map <= upper_bound)
        ys, xs = np.where(mask)
    
        # Compute marker size in points^2 so that each scatter square matches one imshow pixel
        fig_w, fig_h = fig.get_size_inches()*fig.dpi     # figure size in pixels
        ax_w, ax_h = ax.get_window_extent().width, ax.get_window_extent().height
        ny, nx = intensity_map.shape
        px_size_x = ax_w / nx   # pixel size in display units
        px_size_y = ax_h / ny
        px_size_points = 0.5*(px_size_x + px_size_y)# Convert to points (scatter uses area in points^2). Approximate with geometric mean of x and y pixel size
        marker_size = px_size_points**2  
    
        # Use scatter with square markers, size matches pixel size
        ax.scatter(xs, ys, marker='s', s=marker_size, facecolor='red', edgecolor='none', alpha=0.5, label=f'N pixels highlighted: {len(xs)}')
        
        
    if contours_highlights_both=='contours' or contours_highlights_both=='both':
        contour_lower = ax.contour(intensity_map, levels=[lower_bound], colors='cyan', linewidths=2)
        contour_upper = ax.contour(intensity_map, levels=[upper_bound], colors='blue', linewidths=2)

        legend_elements = [
            mlines.Line2D([],[],color='cyan', label=f'{np.round(lower_bound,4)}'),
            mlines.Line2D([],[],color='blue', label=f'{np.round(lower_bound,4)}')
        ]

    # Labels and title
    ax.set_title(f'SOHO/SUMER intensity map of Ne VIII 770.428 \u212B', fontsize=18)
    ax.set_xlabel('Wavelength dimension (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial dimension (pixels)', color='black', fontsize=16)
    ax.legend()

    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)



#####################################
# Functions to average spectra of the regions inside the intensity range
def w_image_pixels_to_calibrated_wavelength(w_px_img, slope_cal, intercept_cal): 
    """
    This function converts the pixel numbers of the x axis (wavelength direction) in the image to calibrated wavelength values (in Angstrom). 
    """
    w_cal = slope_cal*w_px_img + intercept_cal
    return w_cal #[Angstrom]


def w_calibrated_wavelength_to_image_pixels(w_cal, slope_cal, intercept_cal):
    """
    This function is the inverse of image_pixels_to_calibrated_wavelength()
    """
    m = 1/slope_cal
    b = -intercept_cal/slope_cal
    w_px_img = m * w_cal + b 
    return w_px_img #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(w_px) in case s_px is one number, or np.floor(w_px).astype(int) in case s_px is an array.


# Convert calibrated wavelength range to (closest) pixel range
def w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range, slope_cal, intercept_cal):
    # Calculate the pixel range (floats)
    w_px_range0_float = w_calibrated_wavelength_to_image_pixels(w_cal=w_cal_range[0], slope_cal=slope_cal, intercept_cal=intercept_cal)
    w_px_range1_float = w_calibrated_wavelength_to_image_pixels(w_cal=w_cal_range[1], slope_cal=slope_cal, intercept_cal=intercept_cal)
    
    # Round and convert to integer 
    w_px_range0 = int(np.round(w_px_range0_float))
    w_px_range1 = int(np.round(w_px_range1_float))
    w_px_range =  [w_px_range0,  w_px_range1]
    
    # Calculate the range of wavelength calibrated from the center of the pixels range
    w_cal_range0_pixcenter = w_image_pixels_to_calibrated_wavelength(w_px_img=float(w_px_range0), slope_cal=slope_cal, intercept_cal=intercept_cal)
    w_cal_range1_pixcenter = w_image_pixels_to_calibrated_wavelength(w_px_img=float(w_px_range1), slope_cal=slope_cal, intercept_cal=intercept_cal)
    w_cal_range_pixcenter = [w_cal_range0_pixcenter, w_cal_range1_pixcenter]
    
    return [w_px_range, w_cal_range_pixcenter]

def average_profiles_from_pixels_selected(wavelength_range_suggest, wavelength_array, spectra_interpolated_croplat_list, unc_spectra_interpolated_croplat_list, rows_cols_of_spectroheliogram_croplat):
    """
    Inputs:
        - wavelength_range_suggest: range of wavelength to analyze (to average)
        - wavelength_array: list or 1d-array of wavelength calibrated (corresponding to the center of the pixels)
        - spectra_interpolated_croplat_list: list of spectral images
        - unc_spectra_interpolated_croplat_list: uncertainties of "spectra_interpolated_croplat_list"
        - rows_cols_of_spectroheliogram_croplat: list of pixels addresses [row, col] of the spectroheliogram. Each row of the spectroheliogram corresponds to the same row in an spectral image, and each col (column) of the spectroheliogram corresponds to an spectral image.
    """
    
    # Parameters to convert pixels to wavelength
    pixelscale_reference = wavelength_array[1]-wavelength_array[0]
    pixelscale_intercept_reference = wavelength_array[0]
    
    # Calculate the range of pixels and Angstrom in the wavelength direction for each row
    w_px_range, w_cal_range_pixcenter = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=wavelength_range_suggest, slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference)
    col0, col1 = w_px_range
    N_pixels_range = col1 - col0 + 1
    
    profile_intensity_line_sum = np.zeros(N_pixels_range)
    unc_profile_intensity_line_sumsq = np.zeros(N_pixels_range)
    N_pixels_adresses = 0
    for row_lat, idx_spec in rows_cols_of_spectroheliogram_croplat:
        spectrum_array = spectra_interpolated_croplat_list[idx_spec]
        spectrum_array_unc = unc_spectra_interpolated_croplat_list[idx_spec]
        
        profile = spectrum_array[row_lat, col0:col1+1]
        profile_unc = spectrum_array_unc[row_lat, col0:col1+1]

        # Skip if NaNs present
        if np.any(np.isnan(profile)) or np.any(np.isnan(profile_unc)):
            continue

        profile_intensity_line_sum += profile
        unc_profile_intensity_line_sumsq += profile_unc**2
        N_pixels_adresses += 1
    
    # Wavelength array and average the summed intensities
    profile_wavelength_line = w_image_pixels_to_calibrated_wavelength(w_px_img=np.arange(col0, col1+1), slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference)
    profile_intensity_line_average = (1/N_pixels_adresses) * profile_intensity_line_sum
    profile_intensity_unc_line_average = (1/N_pixels_adresses) * np.sqrt(unc_profile_intensity_line_sumsq)
    
    return [profile_wavelength_line, profile_intensity_line_average, profile_intensity_unc_line_average]
    
    


def average_profiles_from_pixels_selected_normalized(wavelength_range_suggest, wavelength_array, spectra_interpolated_list, unc_spectra_interpolated_list, rows_cols_of_spectroheliogram, intensitymap_):
    """
    Inputs:
        - wavelength_range_suggest: range of wavelength to analyze (to average)
        - wavelength_array: list or 1d-array of wavelength calibrated (corresponding to the center of the pixels)
        - spectra_interpolated_list: list of spectral images
        - unc_spectra_interpolated_list: uncertainties of "spectra_interpolated_list"
        - rows_cols_of_spectroheliogram: list of pixels addresses [row, col] of the spectroheliogram. Each row of the spectroheliogram corresponds to the same row in an spectral image, and each col (column) of the spectroheliogram corresponds to an spectral image.
    """
    
    # Parameters to convert pixels to wavelength
    pixelscale_reference = wavelength_array[1]-wavelength_array[0]
    pixelscale_intercept_reference = wavelength_array[0]
    
    # Calculate the range of pixels and Angstrom in the wavelength direction for each row
    w_px_range, w_cal_range_pixcenter = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=wavelength_range_suggest, slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference)
    col0, col1 = w_px_range
    N_pixels_range = col1 - col0 + 1
    
    profile_intensity_line_sum = np.zeros(N_pixels_range)
    unc_profile_intensity_line_sumsq = np.zeros(N_pixels_range)
    N_pixels_adresses = 0
    for row_lat, idx_spec in rows_cols_of_spectroheliogram:
        spectrum_array = spectra_interpolated_list[idx_spec]
        spectrum_array_unc = unc_spectra_interpolated_list[idx_spec]
        
        factor_norm = profile * pixelscale_reference
        
        profile = spectrum_array[row_lat, col0:col1+1] / factor_norm
        profile_unc = spectrum_array_unc[row_lat, col0:col1+1] / factor_norm

        # Skip if NaNs present
        if np.any(np.isnan(profile)) or np.any(np.isnan(profile_unc)):
            continue

        profile_intensity_line_sum += profile
        unc_profile_intensity_line_sumsq += profile_unc**2
        N_pixels_adresses += 1
    
    # Wavelength array and average the summed intensities
    profile_wavelength_line = w_image_pixels_to_calibrated_wavelength(w_px_img=np.arange(col0, col1+1), slope_cal=pixelscale_reference, intercept_cal=pixelscale_intercept_reference)
    profile_intensity_line_average = (1/N_pixels_adresses) * profile_intensity_line_sum
    profile_intensity_unc_line_average = (1/N_pixels_adresses) * np.sqrt(unc_profile_intensity_line_sumsq)
    
    return [profile_wavelength_line, profile_intensity_line_average, profile_intensity_unc_line_average]



##############################################
# 

def average_profiles_from_pixels_selected_from_interpolated_data(wavelength_range_, data_interpolated_loaded_, rows_cols_of_spectroheliogram_croplat):
    """
    Inputs:
        - wavelength_range_: range of wavelength to analyze (to average)
        - data_interpolated_loaded_: dictionary of the interploated data
        - rows_cols_of_spectroheliogram_croplat: list of pixels addresses [row, col] of the spectroheliogram. Each row of the spectroheliogram corresponds to the same row in an spectral image, and each col (column) of the spectroheliogram corresponds to an spectral image. The spectroheliogram is cropped in latitude (in slit_top_px and slit_bottom_px), the indices addresses correspond to this cropped image, so do not add slit_top_px to the list of rows. 
    """

    # split dictionary of the interploated data
    spectralimage_interpolated_list_ = data_interpolated_loaded_['spectral_image_interpolated_list']
    unc_spectralimage_interpolated_list_ = data_interpolated_loaded_['spectral_image_unc_interpolated_list']
    spectralimage_interpolated_croplat_list_ = data_interpolated_loaded_['spectral_image_interpolated_croplat_list']
    unc_spectralimage_interpolated_croplat_list_ = data_interpolated_loaded_['spectral_image_unc_interpolated_croplat_list']
    lam_sumer_ = data_interpolated_loaded_['reference_wavelength']          # scalar (0‑d array; use a_loaded.item() for Python float)
    elam_sumer_ = data_interpolated_loaded_['unc_reference_wavelength'] 
    #row_reference_ = int(data_interpolated_loaded_['row_reference'])        # becomes a NumPy array or object array, so I conver it to integer again

    # Indices corresponding to the wavelength range 
    lam_sumer_crop_, idx_sumer_crop_ = crop_range(list_to_crop=lam_sumer_, range_values=wavelength_range_)
    col0, col1 = idx_sumer_crop_
    elam_sumer_crop_ = elam_sumer_[col0:col1+1]
    N_pixels_range = col1 - col0 + 1

    
    rad_sumer_crop_sum = np.zeros(N_pixels_range)
    erad_sumer_crop_sumsq = np.zeros(N_pixels_range)
    N_pixels_adresses = 0
    for row_lat, idx_spec in rows_cols_of_spectroheliogram_croplat:
        spectralimage_interpolated_croplat_ = spectralimage_interpolated_croplat_list_[idx_spec]
        spectralimage_interpolated_unc_croplat_ = unc_spectralimage_interpolated_croplat_list_[idx_spec]
        
        rad_sumer_crop_ = spectralimage_interpolated_croplat_[row_lat, col0:col1+1]
        erad_sumer_crop_ = spectralimage_interpolated_unc_croplat_[row_lat, col0:col1+1]

        # Skip if NaNs present
        if np.any(np.isnan(rad_sumer_crop_)) or np.any(np.isnan(erad_sumer_crop_)):
            continue

        rad_sumer_crop_sum += rad_sumer_crop_
        erad_sumer_crop_sumsq += erad_sumer_crop_**2
        N_pixels_adresses += 1
    
    # Wavelength array and average the summed intensities
    profile_wavelength_line = lam_sumer_crop_
    profile_wavelength_unc_line = elam_sumer_crop_
    rad_sumer_crop_average = (1/N_pixels_adresses) * rad_sumer_crop_sum
    erad_sumer_crop_average = (1/N_pixels_adresses) * np.sqrt(erad_sumer_crop_sumsq)
    
    return [profile_wavelength_line, profile_wavelength_unc_line, rad_sumer_crop_average, erad_sumer_crop_average]


#########################################################################
#########################################################################
# 

