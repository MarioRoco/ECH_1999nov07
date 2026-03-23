import numpy as np
from scipy.optimize import curve_fit
from scipy.odr import Model, RealData, ODR
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d

#from SOHO_aux_functions import *
#from calibration_parameters import *
#from spectroheliogram_functions import *
#from solar_rotation_variables import *

def pixel_to_physical_quantity(ref_pix, ref_quantity, delta_quantity, pix_list):
    """
    - ref_pix: reference pixel that corresponds to the physical quantity "ref_quantity" 
    - ref_quantity: physical quantity of pixel "ref_pix"
    - delta_quantity: wide of each pixel in units of the physical quantity 
    - pix_list: list of pixels that you can convert to physical quantitym
    """
    m = delta_quantity #slope, because slope = (y2-y1)/(x2-x1), where y2-y1=delta_quantity, and x2-x1=1 (x1 and x2 are consecutives, the same for y1 and y2 of course)
    b = ref_quantity #intercept
    x = pix_list
    x0 = ref_pix
    return m*(x-x0)+b

def find_value_in_1Darray(array1D, value_to_find):
    """
    This function finds, in a 1D-array called array1D, the closest value to "value_to_find" and the index of this closest value. 
    """
    closest_index = (np.abs(array1D - value_to_find)).argmin() #Index of the closest value to "value_to_find"
    closest_value = array1D[closest_index]
    return [closest_index, closest_value]

def find_list_in_1Darray(array1D, list_to_find):
    """
    This function finds, in a 1D-array called array1D, the closest values to the values of the list "list_to_find" and the indices of these closests values. 
    """
    
    #convert list_to_find to 1D-array format
    if type(list_to_find)==list: list_to_find=np.array(list_to_find) 
    elif type(list_to_find)==tuple: list_to_find=np.array(list_to_find) 
    
    closest_indices, closest_list = [],[]
    for vtf in list_to_find:
        closest_index, closest_value = find_value_in_1Darray(array1D=array1D, value_to_find=vtf)
        closest_indices.append(closest_index)
        closest_list.append(closest_value)
    return [closest_indices, closest_list]

def crop_range(list_to_crop, range_values):
    """
    This function crops a sequence in the closest values to "range_values" and the indices of this closest value. 
    """
    # Convert sequence to Numpy array (if it isn't)
    import numpy as np
    if type(list_to_crop)==list or type(list_to_crop)==tuple: list_to_crop = np.array(list_to_crop)
    
    # Indices of the closest value to range_values[0] and range_values[1]
    idx_start = (np.abs(list_to_crop - range_values[0])).argmin() 
    idx_end = (np.abs(list_to_crop - range_values[1])).argmin()
    idx_range = [idx_start, idx_end]
    
    # Crop
    list_cropped = list_to_crop[idx_start:idx_end+1]
    
    return [list_cropped, idx_range]

def subtract_median_rows(arr_2D):
    row_medians_ = np.nanmedian(arr_2D, axis=1)  # shape (n_rows,)
    arr_2D_minusmedian = arr_2D - row_medians_[:, np.newaxis]
    return arr_2D_minusmedian


def weighted_mean(values, uncertainties):# function from ChatGPT
    """
    Calculate the weighted mean and its uncertainty for a list of values and uncertainties.
    
    Parameters:
    - values: List of measured values (x_i).
    - uncertainties: List of uncertainties (sigma_i) corresponding to the values.
    
    Returns:
    - mean: Weighted mean of the values.
    - uncertainty: Uncertainty of the weighted mean.
    """

    # Convert uncertainties to weights
    weights = 1 / np.array(uncertainties) ** 2
    
    # Calculate weighted mean
    weighted_mean = np.sum(weights * np.array(values)) / np.sum(weights)
    
    # Calculate uncertainty of the weighted mean
    weighted_mean_uncertainty = np.sqrt(1 / np.sum(weights))
    
    return weighted_mean, weighted_mean_uncertainty

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def quadratic_function(x, a, b, c): 
    """
    f(x) = a*x^2 + b*x + c
    """
    return a*x**2 + b*x + c

def quadratic_function_uncertainty(x, dx, a, b, c, da, db, dc): 
    """
    f(x) = a*x^2 + b*x + c
    """
    return np.sqrt( ((2*a*x+b)*dx)**2 + (x**2*da)**2 + (x*db)**2 + dc**2 )
	
def line_from_points(x1, y1, x2, y2):
    if x1 == x2:
        raise ValueError("Slope is undefined for vertical line (x1 == x2).")
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b #slope, intercept

def line_from_points_uncertainty(x1, sx1, y1, sy1, x2, sx2, y2, sy2):
    if x1 == x2:
        raise ValueError("Slope is undefined for vertical line (x1 == x2).")
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    dm = 1/abs(x2-x1) * np.sqrt( dy1**2 + dy2**2 + (m*dx1)**2 + (m*dx2)**2 )
    db = np.sqrt( dy1**2 + (x1*dm)**2 + (m*dx1)**2 )
    return m, dm, b, db

def line_from_points_uncertainty_calculation(m, dm, b, db, x, dx=0):
    if type(x)==list or type(x)==tuple: x=np.array(x)
    y = m*x+b
    dy = np.sqrt((x*dm)**2 + (m*dx)**2 + db**2)
    return [y, dy]
	

def format_fancy_date(date_obj):
    """
    This function converts a date in datetime format to a fancy string that we can use in the labels.
    """
    
    # Extract day, month, year, and time
    day = date_obj.day
    month = date_obj.strftime('%b')  # Short month name (e.g., Nov)
    year = date_obj.year
    time = date_obj.strftime('%H:%M:%S')  # Hour, minute, second

    # Determine the day suffix
    if 11 <= day <= 13:  # Special case for 11th, 12th, 13th
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    
    # Format the final string
    fancy_date = f"{day}{suffix} {month} {year} at {time}"
    return fancy_date

def division_unc(Num, Den, Num_unc, Den_unc):
    """
    Calculate uncertainty of a division Num/Den. 
    """
    A = Num #numerator
    B = Den #denominator
    dA = Num_unc #numerator uncertainty
    dB = Den_unc #denominator uncertainty
    return np.sqrt( (dA/B)**2 + (-A*dB/(B**2))**2 )

def vkms_doppler(lamb, lamb_0):
    """
    Wavelength (delta) to doppler velocity. 
    """
    c = 299792.4580 #[km/s] speed of light
    return c*(lamb-lamb_0)/lamb_0

def vkms_doppler_unc(lamb, lamb_unc, lamb_0, lamb_0_unc):
    """
    Wavelength (delta) to doppler velocity. 
    """
    c = 299792.4580 #[km/s] speed of light
    return c/lamb_0 * np.sqrt( lamb_unc**2 + (lamb/lamb_0 * lamb_0_unc)**2 )

def lamb_doppler(lamb_0, v_kms):
    """
    Doppler velocity to wavelength.
    """
    c = 299792.4580 #[km/s] speed of light
    return lamb_0*((v_kms/c)+1)

def delta_lamb_doppler(lamb_0, v_unc_kms):
    """
    Doppler velocity to wavelength shift (delta).
    """
    c = 299792.4580 #[km/s] speed of light
    return np.sqrt((lamb_0*(v_unc_kms/c))**2)


def dispersion_relation_detectorA(lamb_Angstroms, order_m):
    """
    Dispersion relation for detector A of SOHO/SUMER. 
    Reference: Thesis of Luca Teriaca (Structure and dynamics of the solar outer atmosphere as inferred from EUV observations). 
    """
    m = order_m #order of diffraction
    l = lamb_Angstroms #wavelength (1D-array) in Angstroms 
    d = 2777.45 #Angstroms #grating spacing
    ra = 3200.78 * 1e7 #Angstrom #radius of the spherical concave grating
    PxA = 26.6 * 1e4 #Angstrom #spectral size of the pixels in detector A
    A = (m*l/d)**2
    B = 1 + np.sqrt(1-A)
    return d*PxA*B/m/ra

def straight_line(x, slope, intercept): 
    return slope * x + intercept

def straight_line_uncertainties(x, x_unc, slope, intercept, slope_unc, intercept_unc): 
    aa = (slope * x_unc)**2
    bb = (x * slope_unc)**2
    cc = intercept_unc**2
    return np.sqrt(aa+bb+cc)


def straight_line_v4(x1, y1, x2, y2, n_dots=300): #you provide the number dots for the x and y axis, it calculates the x and y axis
    m = (y2-y1)/(x2-x1) #slope
    b = y1-m*x1 #intercept
    x_vals = np.linspace(x1, x2, 300)
    y_vals = m * x_vals + b
    return [y_vals, m, b]

    
def straight_line_v2(x, x1, y1, x2, y2): #you provide the x axis
    m = (y2-y1)/(x2-x1) #slope
    b = y1-m*x1 #intercept
    y = m * x + b
    return [y, m, b]
    
def straight_line_v3(x, m, x1, y1): #you provide the x axis
    b = y1-m*x1 #intercept
    y = m * x + b
    return y
    
def straight_line_for_ODR(B, x):
    """
    Linear function B[0]=slope, B[1]=intercept
    Prepared to use with ODR.
    """
    return B[0]*x + B[1]
    
def straight_line_for_curvefit(x, m,b):
    """
    Linear function m=slope, b=intercept
    Prepared to use with scipy.curve_fit.
    """
    return m*x + b
    
def straight_line_for_ODR_uncertainties(B, B_unc, x, x_unc): 
    """
    Uncertainties calculation of the linear function "straight_line_for_ODR(B, x)" (=B[0]*x + B[1]) by error propagation. 
    Parameters: B[0]=slope, B[1]=intercept
    B_unc = uncertainties of the parameters B
    """
    aa = (B[0]*x_unc)**2
    bb = (x * B_unc[0])**2
    cc = B_unc[1]**2
    return np.sqrt(aa+bb+cc)


def gaussian_function(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x-mean)**2)/(2*stddev**2))

def gaussian_function_v2(x, amplitude, mean, FWHM_):
    stddev_ = FWHM_ / (2*np.sqrt(2*np.log(2)))
    return amplitude * np.exp(-((x-mean)**2)/(2*stddev_**2))
    
def multi_gaussian_function(x, amplitude_list, mean_list, stddev_list):
    gaussian_total = np.zeros(len(x))
    for i in range(len(x)):
        gf_i = gaussian_function(x, amplitude_list[i], mean_list[i], stddev_list[i])
        gaussian_total = gaussian_total + gf_i
    return gaussian_total
    
def gaussian_function_for_ODR(B, x):
    """
    Gaussian function B[0]=amplitude, B[1]=mean, B[2]=FWHM, B[3]=background
    Prepared to use with ODR.
    """
    amplitude_ = B[0]
    mean_ = B[1]
    FWHM_ = B[2]
    bckg_ = B[3]
    
    stddev_ = FWHM_ / (2*np.sqrt(2*np.log(2))) #convert FWHM to standard deviation
    XM2 = (x-mean_)**2
    SD2 = 2*stddev_**2
    return amplitude_ * np.exp(-XM2/SD2) + bckg_
    
def multi_gaussian_function_for_ODR(B, x): 
    """
    Gaussian function B[0]=background, B[1]=amplitude1, B[2]=mean1, B[3]=FWHM1, B[4]=amplitude2, B[5]=mean2, B[6]=FWHM2, B[7]=mean3,...
    Prepared to use with ODR.
    Number of gaussians to fit should be equal to (len(B)-1)//3
    """
    y = B[0] * np.ones(len(x)) #background
    
    n_gauss = (len(B)-1)//3 #number of gaussians to fit
    for i in range(0,n_gauss):
        #print('i:', 3*i+1, 3*i+2, 3*i+3)
        
        amplitude_ = B[3*i+1]
        mean_ = B[3*i+2]
        FWHM_ = B[3*i+3]
        
        stddev_ = FWHM_ / (2*np.sqrt(2*np.log(2))) #convert FWHM to standard deviation
        XM2 = (x-mean_)**2
        SD2 = 2*stddev_**2
        gauss_ = amplitude_ * np.exp(-XM2/SD2)
        y = y + gauss_
    
    return y

def multi_gaussian_function_uncertainties(B, B_unc, x, x_unc):
    """
    Function that calculates the uncertainties of the fit (model). It is the error propagation of the function "multi_gaussian_function_for_ODR", derivating x and the parameters
    B[0]=background, B[1]=amplitude1, B[2]=mean1, B[3]=FWHM1, B[4]=amplitude2, B[5]=mean2, B[6]=FWHM2, B[7]=mean3,...
    B_unc are the uncertainties of B.
    Number of gaussians to fit should be equal to (len(B)-1)//3
    """
    N_gauss = (len(B)-1)//3 #number of gaussians to fit
    T1, TE3, TE4, TE5 = 0,0,0,0
    for i in range(0,N_gauss):
        #print('i:', 3*i+1, 3*i+2, 3*i+3)
        
        # gaussian parameters of this iteration
        I_i = B[3*i+1]
        m_i = B[3*i+2]
        w_i = B[3*i+3]
        
        # uncertainties of above parameters
        dI_i = B_unc[3*i+1]
        dm_i = B_unc[3*i+2]
        dw_i = B_unc[3*i+3]
        
        # furmula of the uncertainty
        TEi = np.exp(-4*np.log(2)*(x-m_i)**2 / w_i**2)
        T1i = I_i*(x-m_i)/(w_i**2) * TEi
        T1 = T1 - 8*np.log(2)*x_unc*T1i
        T3i = dI_i
        T4i = 8*np.log(2)*I_i*(x-m_i)/(w_i**2) * dm_i
        T5i = 8*np.log(2)*I_i*(x-m_i)**2/(w_i**3) * dw_i
        TE3 = TE3 + (TEi*T3i)**2 
        TE4 = TE4 + (TEi*T4i)**2 
        TE5 = TE5 + (TEi*T5i)**2 
    
    T2 = B_unc[0]
    
    return np.sqrt(T1**2 + T2**2 + TE3 + TE4 + TE5)


def fit_multi_gaussian_ODR(x_data, y_data, x_unc_data, y_unc_data, init_parameters):  
    """
    This function calculates the multigaussian fit of a set of data and the residuals.
    
    - "guess_parameters": [background, amplitude1, mean1, FWHM1, amplitude2, mean2, FWHM2,...]

    """

    # save input data in a dictionary that will contain  all relevant results
    multigauss_fit_results = {'x_data': x_data} #create the dictionary starting with x_data
    multigauss_fit_results['x_unc_data'] = x_unc_data
    multigauss_fit_results['y_data'] = y_data
    multigauss_fit_results['y_unc_data'] = y_unc_data
    
    # Fit with ODR (Orthogonal Distance Regression)
    model = Model(multi_gaussian_function_for_ODR) # Define the model for ODR
    real_data = RealData(x=x_data, y=y_data, sx=x_unc_data, sy=y_unc_data) # Create a RealData object
    odr = ODR(real_data, model, beta0=init_parameters)  # Set up ODR with the model and data# Initial guess for parameters
    output = odr.run() # Run the regression
    
    # save output
    multigauss_fit_results['output'] = output

    # Results: curve of the model
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    x_unc_fit = x_unc_data[0]*np.ones(len(x_fit))
    y_fit = multi_gaussian_function_for_ODR(output.beta, x_fit)
    y_unc_fit = multi_gaussian_function_uncertainties(B=output.beta, B_unc=output.sd_beta, x=x_fit, x_unc=x_unc_fit)
    y_fit_length_data = multi_gaussian_function_for_ODR(output.beta, x_data)
    y_unc_fit_length_data = multi_gaussian_function_uncertainties(B=output.beta, B_unc=output.sd_beta, x=x_data, x_unc=x_unc_data)
    
    ## save 
    multigauss_fit_results['x_fit'] = x_fit
    multigauss_fit_results['x_unc_fit'] = x_unc_fit
    multigauss_fit_results['y_fit'] = y_fit
    multigauss_fit_results['y_unc_fit'] = y_unc_fit
    multigauss_fit_results['y_fit_length_data'] = y_fit_length_data
    multigauss_fit_results['y_unc_fit_length_data'] = y_unc_fit_length_data
    multigauss_fit_results['x_data'] = x_data
    multigauss_fit_results['y_data'] = y_data
    
    # First fit parameter: background
    bckg = output.beta[0]
    bckg_unc = output.sd_beta[0]
    
    # save background parameter
    multigauss_fit_results['bckg'] = bckg
    multigauss_fit_results['bckg_unc'] = bckg_unc

    # The other parameters
    color_list = ['blue','red','green','orange','purple','brown','pink','gray','olive','cyan','magenta','blue','red','green','orange','purple','brown','pink','gray','olive','cyan','magenta']
    n_gauss = (len(init_parameters)-1)//3 #number of gaussians to fit
    multigauss_fit_results['n_gauss'] = n_gauss #save number of gaussians
    for i in range(n_gauss):
        # output parameters of gaussian number "i"
        amplitude_i = output.beta[3*i+1]
        mean_i = output.beta[3*i+2]
        FWHM_i = output.beta[3*i+3]

        # uncertainty of output parameters of gaussian number "i"
        amplitude_unc_i = output.sd_beta[3*i+1]
        mean_unc_i = output.sd_beta[3*i+2]
        FWHM_unc_i = output.sd_beta[3*i+3]

        # fill dictionary with the results
        multigauss_fit_results['gaussian_'+str(i+1)] = {} #create sub-dictionary inside the dictionary multigauss_fit_results
        multigauss_fit_results['gaussian_'+str(i+1)]['amplitude'] = amplitude_i
        multigauss_fit_results['gaussian_'+str(i+1)]['amplitude_unc'] = amplitude_unc_i
        multigauss_fit_results['gaussian_'+str(i+1)]['mean'] = mean_i
        multigauss_fit_results['gaussian_'+str(i+1)]['mean_unc'] = mean_unc_i
        multigauss_fit_results['gaussian_'+str(i+1)]['FWHM'] = FWHM_i
        multigauss_fit_results['gaussian_'+str(i+1)]['FWHM_unc'] = FWHM_unc_i
        multigauss_fit_results['gaussian_'+str(i+1)]['color'] = color_list[i]
    
    # Residuals
    y_residuals = y_data - y_fit_length_data
    y_unc_residuals = np.sqrt(y_unc_data**2 + y_unc_fit_length_data**2)
    multigauss_fit_results['y_residuals'] = y_residuals
    multigauss_fit_results['y_unc_residuals'] = y_unc_residuals
    
    return multigauss_fit_results


#multigauss_fit_results = fit_multi_gaussian_ODR(x_data, y_data, x_unc_data, y_unc_data, init_parameters)
def plot_fit_multi_gaussian_ODR(multigauss_fit_results, y_lims='no'):  
    """
    This function plots the results, fit and residuals of the function "fit_multi_gaussian_ODR".
    """
    mgfr = multigauss_fit_results
    
    # Plot the results
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,8), sharex=True, gridspec_kw={'height_ratios': [2.5,1]})
    ax[0].set_title('SOHO/SUMER spectrum, multigaussian fit')
    #ax[0].plot(x_profile, y_profile, color='blue', marker='o', markersize=5, linewidth=0)
    ax[0].errorbar(x=mgfr['x_data'], y=mgfr['y_data'], xerr=mgfr['x_unc_data'], yerr=mgfr['y_unc_data'], color='blue', linewidth=1, marker='o', label='data', zorder=1)
    ax[0].errorbar(x=mgfr['x_fit'], y=mgfr['y_fit'], yerr=mgfr['y_unc_fit'], color='red', linewidth=1, label=r"fit errorbars ($y$)", alpha=0.2, zorder=2)
    ax[0].errorbar(x=mgfr['x_fit'], y=mgfr['y_fit'], color='red', label='fit', linewidth=1, zorder=2)
    ax[0].set_xlabel('Wavelength direction [pixels]')
    ax[0].set_ylabel(r'Intensity [$W\ sr^{-1}\ m^{-2}\ \overset{\circ}{\rm A}^{-1}$]')
    ax[0].set_yscale('linear')
    
    # Background
    ax[0].axhline(y=mgfr['bckg'], color='black', linestyle=':', linewidth=1.5, label=f'Bckg: {np.round(mgfr["bckg"],5)} \u00B1 {np.round(mgfr["bckg_unc"],5)}')
    ax[0].axhline(y=mgfr['bckg']-mgfr['bckg_unc'], color='black', linestyle=':', linewidth=0.5)
    ax[0].axhline(y=mgfr['bckg']+mgfr['bckg_unc'], color='black', linestyle=':', linewidth=0.5)
    
    # The other parameters
    for i in np.arange(1, mgfr['n_gauss']+1):
        
        amplitude_i = mgfr['gaussian_'+str(i)]['amplitude']
        amplitude_unc_i = mgfr['gaussian_'+str(i)]['amplitude_unc']
        mean_i = mgfr['gaussian_'+str(i)]['mean']
        mean_unc_i = mgfr['gaussian_'+str(i)]['mean_unc']
        FWHM_i = mgfr['gaussian_'+str(i)]['FWHM']
        FWHM_unc_i = mgfr['gaussian_'+str(i)]['FWHM_unc']
        color_i = mgfr['gaussian_'+str(i)]['color']
        
        # Show fit parameters mean and FWHM of each gaussian in the plot with vertical lines and color windows, respectively
        ax[0].axvspan(mean_i-FWHM_i/2, mean_i+FWHM_i/2, color=color_i, alpha=0.15, label=f'FWHM {i}: {np.round(FWHM_i,2)} \u00B1 {np.round(FWHM_unc_i,2)}', zorder=3)
        ax[0].axvline(x=mean_i, color=color_i, linestyle='--', linewidth=1.5, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}', zorder=4)
        ax[0].axvline(x=mean_i-mean_unc_i, color=color_i, linestyle='--', linewidth=0.5)
        ax[0].axvline(x=mean_i+mean_unc_i, color=color_i, linestyle='--', linewidth=0.5)
        #ax[0].axvspan(mean_i-mean_unc_i, mean_i+mean_unc_i, color=color_i, alpha=0.30, label=f'mean {i} \u00B1 std {np.round(mean_unc_i,3)}', zorder=3)

        # Add an annotation near the top-right of the line
        ax[0].annotate(i, xy=(mean_i, max(mgfr['y_data'])), xytext=(mean_i + 0.5, max(mgfr['y_data'])), fontsize=12, color=color_i)#arrowprops=dict(arrowstyle='->', color='black'),  # Optional arrow

    ax[0].legend(loc="upper left", bbox_to_anchor=(1, 1))  # Place the legend outside the figure
    ax[0].set_xlim([min(mgfr['x_data']), max(mgfr['x_data'])])
    if y_lims!='no': ax[0].set_ylim(y_lims)

    # Lower panel: residuals
    ax[1].errorbar(x=mgfr['x_data'], y=mgfr['y_residuals'], yerr=mgfr['y_unc_residuals'], color='black', linewidth=1, marker='o', label='residuals')
    ax[1].set_xlabel('Wavelength direction [pixels]')
    ax[1].set_ylabel(r'Residuals ($y_{\rm data} - y_{\rm fit}$)')
    ax[1].axhline(y=0, color='grey', linestyle='--', linewidth=1, label='y=0')
    ylimres_ = max(abs(min(mgfr['y_residuals'])), abs(max(mgfr['y_residuals'])))
    ylimres = 1.1*ylimres_
    ax[1].set_ylim([-ylimres, ylimres])
    ax[1].legend()
    
    # Adjust layout to make room for the legend
    #plt.tight_layout(rect=[0, 0, 1, 1.5])  # Adjust the right margin for the legend
    plt.subplots_adjust(left=0.05, right=0.8, bottom=0.05, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)


#multigauss_fit_results = fit_multi_gaussian_ODR(x_data, y_data, x_unc_data, y_unc_data, init_parameters)
def plot_fit_multi_gaussian_ODR_v2(multigauss_fit_results, y_lims='no'):  
    """
    This function plots the results, fit and residuals of the function "fit_multi_gaussian_ODR".
    """
    mgfr = multigauss_fit_results
    
    # Plot the results
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,8), sharex=True, gridspec_kw={'height_ratios': [2.5,1]})
    ax[0].set_title('SOHO/SUMER spectrum, multigaussian fit')
    #ax[0].plot(x_profile, y_profile, color='blue', marker='o', markersize=5, linewidth=0)
    ax[0].errorbar(x=mgfr['x_data'], y=mgfr['y_data'], xerr=mgfr['x_unc_data'], yerr=mgfr['y_unc_data'], color='black', linewidth=0, elinewidth=1, marker='o', label='data', zorder=1)
    ax[0].errorbar(x=mgfr['x_fit'], y=mgfr['y_fit'], yerr=mgfr['y_unc_fit'], color='grey', linewidth=1, label=r"fit errorbars ($y$)", alpha=0.2, zorder=2)
    ax[0].errorbar(x=mgfr['x_fit'], y=mgfr['y_fit'], color='black', label='fit', linewidth=1, zorder=2)
    ax[0].set_xlabel('Wavelength direction [pixels]')
    ax[0].set_ylabel(r'Intensity [$W\ sr^{-1}\ m^{-2}\ \overset{\circ}{\rm A}^{-1}$]')
    ax[0].set_yscale('linear')
    
    # Background
    ax[0].axhline(y=mgfr['bckg'], color='black', linestyle=':', linewidth=1.5, label=f'Bckg: {np.round(mgfr["bckg"],5)} \u00B1 {np.round(mgfr["bckg_unc"],5)}')
    ax[0].axhline(y=mgfr['bckg']-mgfr['bckg_unc'], color='black', linestyle=':', linewidth=0.5)
    ax[0].axhline(y=mgfr['bckg']+mgfr['bckg_unc'], color='black', linestyle=':', linewidth=0.5)
    
    # The other parameters
    for i in np.arange(1, mgfr['n_gauss']+1):
        
        amplitude_i = mgfr['gaussian_'+str(i)]['amplitude']
        amplitude_unc_i = mgfr['gaussian_'+str(i)]['amplitude_unc']
        mean_i = mgfr['gaussian_'+str(i)]['mean']
        mean_unc_i = mgfr['gaussian_'+str(i)]['mean_unc']
        FWHM_i = mgfr['gaussian_'+str(i)]['FWHM']
        FWHM_unc_i = mgfr['gaussian_'+str(i)]['FWHM_unc']
        color_i = mgfr['gaussian_'+str(i)]['color']
        
        # Show fit parameters mean and FWHM of each gaussian in the plot with vertical lines and color windows, respectively
        x_gauss_i = np.linspace(mean_i-FWHM_i, mean_i+FWHM_i, 100)
        y_gauss_i = gaussian_function_v2(x=x_gauss_i, amplitude=amplitude_i, mean=mean_i, FWHM_=FWHM_i)
        y_gauss_plusbckg_i = y_gauss_i + mgfr['bckg']
        
        ax[0].plot(x_gauss_i, y_gauss_i, color=color_i, linestyle='-', linewidth=1)
        ax[0].plot(x_gauss_i, y_gauss_plusbckg_i, color=color_i, linestyle=':', linewidth=2)
        #ax[0].axvline(x=mean_i, color=color_i, linestyle='--', linewidth=1.5, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}', zorder=4)
        #ax[0].axvline(x=mean_i-mean_unc_i, color=color_i, linestyle='--', linewidth=0.5)
        #ax[0].axvline(x=mean_i+mean_unc_i, color=color_i, linestyle='--', linewidth=0.5)
        ax[0].axvspan(mean_i-mean_unc_i, mean_i+mean_unc_i, color=color_i, alpha=0.15, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}', zorder=3)
        
        # Add an annotation near the top-right of the line
        #ax[0].annotate(i, xy=(mean_i, max(mgfr['y_data'])), xytext=(mean_i + 0.5, max(mgfr['y_data'])), fontsize=12, color=color_i)#arrowprops=dict(arrowstyle='->', color='black'),  # Optional arrow
    
    ax[0].plot([],[], linestyle=':', linewidth=2, color='black', label='Model gaussian + bckg')
    ax[0].legend(loc="upper left", bbox_to_anchor=(1, 1))  # Place the legend outside the figure
    ax[0].set_xlim([min(mgfr['x_data']), max(mgfr['x_data'])])
    if y_lims!='no': ax[0].set_ylim(y_lims)
    
    # Lower panel: residuals
    ax[1].errorbar(x=mgfr['x_data'], y=mgfr['y_residuals'], yerr=mgfr['y_unc_residuals'], color='black', linewidth=1, marker='o', label='residuals')
    ax[1].set_xlabel('Wavelength direction [pixels]')
    ax[1].set_ylabel(r'Residuals ($y_{\rm data} - y_{\rm fit}$)')
    ax[1].axhline(y=0, color='grey', linestyle='--', linewidth=1, label='y=0')
    ylimres_ = max(abs(min(mgfr['y_residuals'])), abs(max(mgfr['y_residuals'])))
    ylimres = 1.1*ylimres_
    ax[1].set_ylim([-ylimres, ylimres])
    ax[1].legend()
    
    # Adjust layout to make room for the legend
    #plt.tight_layout(rect=[0, 0, 1, 1.5])  # Adjust the right margin for the legend
    plt.subplots_adjust(left=0.08, right=0.76, bottom=0.08, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)
    
    
#multigauss_fit_results = fit_multi_gaussian_ODR(x_data, y_data, x_unc_data, y_unc_data, init_parameters)
def plot_fit_multi_gaussian_ODR_v3(multigauss_fit_results, title='auto', y_lims='no', show_means='no'):  
    """
    This function plots the results, fit and residuals of the function "fit_multi_gaussian_ODR".
    """
    mgfr = multigauss_fit_results
    
    # Plot the results
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12.5,8), sharex=True, gridspec_kw={'height_ratios': [2.5,1]})
    if title=='auto': ax[0].set_title('SOHO/SUMER spectrum, multigaussian fit')
    else: ax[0].set_title(title)
    #ax[0].plot(x_profile, y_profile, color='blue', marker='o', markersize=5, linewidth=0)
    ax[0].errorbar(x=mgfr['x_data'], y=mgfr['y_data'], xerr=mgfr['x_unc_data'], yerr=mgfr['y_unc_data'], color='black', linewidth=0, elinewidth=1, marker='o', label='data', zorder=1)
    ax[0].errorbar(x=mgfr['x_fit'], y=mgfr['y_fit'], yerr=mgfr['y_unc_fit'], color='grey', linewidth=1, alpha=0.2)#, label=r"fit errorbars ($y$)", zorder=2)
    ax[0].errorbar(x=mgfr['x_fit'], y=mgfr['y_fit'], color='black', label='fit', linewidth=1, zorder=2)
    ax[0].set_xlabel('Wavelength direction [pixels]')
    ax[0].set_ylabel(r'Intensity [$W\ sr^{-1}\ m^{-2}\ \overset{\circ}{\rm A}^{-1}$]')
    ax[0].set_yscale('linear')
    
    # Background
    ax[0].axhline(y=mgfr['bckg'], color='black', linestyle=':', linewidth=1.5, label=f'Bckg: {np.round(mgfr["bckg"],5)} \u00B1 {np.round(mgfr["bckg_unc"],5)}')
    ax[0].axhline(y=mgfr['bckg']-mgfr['bckg_unc'], color='black', linestyle=':', linewidth=0.5)
    ax[0].axhline(y=mgfr['bckg']+mgfr['bckg_unc'], color='black', linestyle=':', linewidth=0.5)
    
    # The other parameters
    for i in np.arange(1, mgfr['n_gauss']+1):
        
        amplitude_i = mgfr['gaussian_'+str(i)]['amplitude']
        amplitude_unc_i = mgfr['gaussian_'+str(i)]['amplitude_unc']
        mean_i = mgfr['gaussian_'+str(i)]['mean']
        mean_unc_i = mgfr['gaussian_'+str(i)]['mean_unc']
        FWHM_i = mgfr['gaussian_'+str(i)]['FWHM']
        FWHM_unc_i = mgfr['gaussian_'+str(i)]['FWHM_unc']
        color_i = mgfr['gaussian_'+str(i)]['color']
        
        # Show fit parameters mean and FWHM of each gaussian in the plot with vertical lines and color windows, respectively
        x_gauss_i = np.linspace(mean_i-FWHM_i, mean_i+FWHM_i, 100)
        y_gauss_i = gaussian_function_v2(x=x_gauss_i, amplitude=amplitude_i, mean=mean_i, FWHM_=FWHM_i)
        y_gauss_plusbckg_i = y_gauss_i + mgfr['bckg']
        
        #ax[0].plot(x_gauss_i, y_gauss_i, color=color_i, linestyle='-', linewidth=1)
        ax[0].plot(x_gauss_i, y_gauss_plusbckg_i, color=color_i, linestyle='-', linewidth=2, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}')
        if show_means=='yes':
            ax[0].axvline(x=mean_i, color=color_i, linestyle='--', linewidth=1.5, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}', zorder=4)
            #ax[0].axvline(x=mean_i-mean_unc_i, color=color_i, linestyle='--', linewidth=0.5)
            #ax[0].axvline(x=mean_i+mean_unc_i, color=color_i, linestyle='--', linewidth=0.5)
            ax[0].axvspan(mean_i-mean_unc_i, mean_i+mean_unc_i, color=color_i, alpha=0.15, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}', zorder=3)
            #ax[0].axvline(mean_i, color=color_i, linestyle='--', linewidth=1.5, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}')
        
        # Add an annotation near the top-right of the line
        #ax[0].annotate(i, xy=(mean_i, max(mgfr['y_data'])), xytext=(mean_i + 0.5, max(mgfr['y_data'])), fontsize=12, color=color_i)#arrowprops=dict(arrowstyle='->', color='black'),  # Optional arrow

    #ax[0].axvspan(mean_i, mean_i, color='grey', alpha=0.15, label=f'y-errorbars of the fit')
    #ax[0].plot([],[], linestyle='-', linewidth=2, color='black', label='Model gaussian + bckg')
    ax[0].legend(loc="upper left", fontsize=13)#, bbox_to_anchor=(1, 1))  # Place the legend outside the figure
    ax[0].set_xlim([min(mgfr['x_data']), max(mgfr['x_data'])])
    if y_lims!='no': ax[0].set_ylim(y_lims)
    
    # Lower panel: residuals
    ax[1].errorbar(x=mgfr['x_data'], y=mgfr['y_residuals'], yerr=mgfr['y_unc_residuals'], color='black', linewidth=1, marker='o', label='residuals')
    #ax[1].set_xlabel('Wavelength direction [pixels]')
    ax[1].set_xlabel('Doppler velocity (km/s)')
    ax[1].set_ylabel(r'Residuals ($y_{\rm data} - y_{\rm fit}$)')
    ax[1].axhline(y=0, color='grey', linestyle='--', linewidth=1, label='y=0')
    ylimres_ = max(abs(min(mgfr['y_residuals'])), abs(max(mgfr['y_residuals'])))
    ylimres = 1.1*ylimres_
    ax[1].set_ylim([-ylimres, ylimres])
    ax[1].legend()
    
    # legend of the upper panel and uncertainties of the fit line
    import matplotlib.patches as mpatches
    uncertainty_patch = mpatches.Patch(color='grey', alpha=0.15, label=r'uncertainty ($y$) of the fit') # Create a legend entry for the grey rectangle (uncertainties)
    ax[0].legend(handles=[uncertainty_patch, *ax[0].get_legend_handles_labels()[0]])# Show legend with the added patch
    # Ensure the legend includes all elements
    handles, labels = ax[0].get_legend_handles_labels()
    handles.insert(0, uncertainty_patch)  # Add the uncertainty patch first
    ax[0].legend(handles=handles, loc="upper left", fontsize=13)  # Final legend call
    
    # Adjust layout to make room for the legend
    #plt.tight_layout(rect=[0, 0, 1, 1.5])  # Adjust the right margin for the legend
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)
    

#multigauss_fit_results = fit_multi_gaussian_ODR(x_data, y_data, x_unc_data, y_unc_data, init_parameters)
def plot_fit_multi_gaussian_ODR_v4(multigauss_fit_results, title='auto', y_lims='no', show_means='no'):  
    """
    This function plots the results, fit and residuals of the function "fit_multi_gaussian_ODR".
    """
    mgfr = multigauss_fit_results
    
    # Plot the results
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12.5,8), sharex=True, gridspec_kw={'height_ratios': [2.5,1]})
    if title=='auto': ax[0].set_title('SOHO/SUMER spectrum, multigaussian fit')
    else: ax[0].set_title(title)
    #ax[0].plot(x_profile, y_profile, color='blue', marker='o', markersize=5, linewidth=0)
    ax[0].errorbar(x=mgfr['x_data'], y=mgfr['y_data'], xerr=mgfr['x_unc_data'], yerr=mgfr['y_unc_data'], color='black', linewidth=0, elinewidth=1, marker='o', label='data', zorder=1)
    #ax[0].errorbar(x=mgfr['x_fit'], y=mgfr['y_fit'], yerr=mgfr['y_unc_fit'], color='grey', linewidth=1, alpha=0.2)#, label=r"fit errorbars ($y$)", zorder=2)
    ax[0].errorbar(x=mgfr['x_fit'], y=mgfr['y_fit'], color='black', label='fit', linewidth=1, zorder=2)
    ax[0].set_xlabel('Wavelength direction [pixels]')
    ax[0].set_ylabel(r'Intensity [$W\ sr^{-1}\ m^{-2}\ \overset{\circ}{\rm A}^{-1}$]')
    ax[0].set_yscale('linear')
    
    # Background
    ax[0].axhline(y=mgfr['bckg'], color='black', linestyle=':', linewidth=1.5, label=f'Bckg: {np.round(mgfr["bckg"],5)} \u00B1 {np.round(mgfr["bckg_unc"],5)}')
    ax[0].axhline(y=mgfr['bckg']-mgfr['bckg_unc'], color='black', linestyle=':', linewidth=0.5)
    ax[0].axhline(y=mgfr['bckg']+mgfr['bckg_unc'], color='black', linestyle=':', linewidth=0.5)
    
    # The other parameters
    for i in np.arange(1, mgfr['n_gauss']+1):
        
        amplitude_i = mgfr['gaussian_'+str(i)]['amplitude']
        amplitude_unc_i = mgfr['gaussian_'+str(i)]['amplitude_unc']
        mean_i = mgfr['gaussian_'+str(i)]['mean']
        mean_unc_i = mgfr['gaussian_'+str(i)]['mean_unc']
        FWHM_i = mgfr['gaussian_'+str(i)]['FWHM']
        FWHM_unc_i = mgfr['gaussian_'+str(i)]['FWHM_unc']
        color_i = mgfr['gaussian_'+str(i)]['color']
        
        # Show fit parameters mean and FWHM of each gaussian in the plot with vertical lines and color windows, respectively
        x_gauss_i = np.linspace(mean_i-FWHM_i, mean_i+FWHM_i, 100)
        y_gauss_i = gaussian_function_v2(x=x_gauss_i, amplitude=amplitude_i, mean=mean_i, FWHM_=FWHM_i)
        y_gauss_plusbckg_i = y_gauss_i + mgfr['bckg']
        
        #ax[0].plot(x_gauss_i, y_gauss_i, color=color_i, linestyle='-', linewidth=1)
        ax[0].plot(x_gauss_i, y_gauss_plusbckg_i, color=color_i, linestyle='-', linewidth=2, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}')
        if show_means=='yes':
            ax[0].axvline(x=mean_i, color=color_i, linestyle='--', linewidth=1.5, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}', zorder=4)
            #ax[0].axvline(x=mean_i-mean_unc_i, color=color_i, linestyle='--', linewidth=0.5)
            #ax[0].axvline(x=mean_i+mean_unc_i, color=color_i, linestyle='--', linewidth=0.5)
            ax[0].axvspan(mean_i-mean_unc_i, mean_i+mean_unc_i, color=color_i, alpha=0.15, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}', zorder=3)
            #ax[0].axvline(mean_i, color=color_i, linestyle='--', linewidth=1.5, label=f'mean {i}:  {np.round(mean_i,3)} \u00B1 {np.round(mean_unc_i,3)}')
        
        # Add an annotation near the top-right of the line
        #ax[0].annotate(i, xy=(mean_i, max(mgfr['y_data'])), xytext=(mean_i + 0.5, max(mgfr['y_data'])), fontsize=12, color=color_i)#arrowprops=dict(arrowstyle='->', color='black'),  # Optional arrow

    #ax[0].axvspan(mean_i, mean_i, color='grey', alpha=0.15, label=f'y-errorbars of the fit')
    #ax[0].plot([],[], linestyle='-', linewidth=2, color='black', label='Model gaussian + bckg')
    ax[0].legend(loc="upper left", fontsize=13)#, bbox_to_anchor=(1, 1))  # Place the legend outside the figure
    ax[0].set_xlim([min(mgfr['x_data']), max(mgfr['x_data'])])
    if y_lims!='no': ax[0].set_ylim(y_lims)
    
    # Lower panel: residuals
    ax[1].errorbar(x=mgfr['x_data'], y=mgfr['y_residuals'], yerr=mgfr['y_unc_residuals'], color='black', linewidth=1, marker='o', label='residuals')
    #ax[1].set_xlabel('Wavelength direction [pixels]')
    ax[1].set_xlabel('Doppler velocity (km/s)')
    ax[1].set_ylabel(r'Residuals ($y_{\rm data} - y_{\rm fit}$)')
    ax[1].axhline(y=0, color='grey', linestyle='--', linewidth=1, label='y=0')
    ylimres_ = max(abs(min(mgfr['y_residuals'])), abs(max(mgfr['y_residuals'])))
    ylimres = 1.1*ylimres_
    ax[1].set_ylim([-ylimres, ylimres])
    ax[1].legend()
    """
    # legend of the upper panel and uncertainties of the fit line
    import matplotlib.patches as mpatches
    uncertainty_patch = mpatches.Patch(color='grey', alpha=0.15, label=r'uncertainty ($y$) of the fit') # Create a legend entry for the grey rectangle (uncertainties)
    ax[0].legend(handles=[uncertainty_patch, *ax[0].get_legend_handles_labels()[0]])# Show legend with the added patch
    # Ensure the legend includes all elements
    handles, labels = ax[0].get_legend_handles_labels()
    handles.insert(0, uncertainty_patch)  # Add the uncertainty patch first
    ax[0].legend(handles=handles, loc="upper left", fontsize=13)  # Final legend call
    """
    ax[0].legend(loc="upper left", fontsize=13)
    
    # Adjust layout to make room for the legend
    #plt.tight_layout(rect=[0, 0, 1, 1.5])  # Adjust the right margin for the legend
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)


def multigaussian_function_for_curvefit(x, *params): #params is a list: [background, amplitude1, mean1, fwhm1, ]
	# unpack params depending on your model
	# example: assuming each Gaussian has (amplitude, mean, sigma)
	n = (len(params) - 1)//3
	bckg_init = params[0] #the first item of "params" (list or tuple) is the background level, considered as constant here.
	result = np.full_like(x, bckg_init, dtype=float) #same length as x
	#result = np.array([bckg_init]*len(x)) #same length as x
	for i in range(n):
		A = params[3*i+1]
		mu = params[3*i+2]
		fwhm = params[3*i+3]
		sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
		result = result + A * np.exp(-(x - mu)**2 / (2 * sigma**2))
	return result
	
def gaussian_function_with_background(x, bckg, amplitude, mean, fwhm):
	sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
	return bckg + amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def multi_gaussian_function_uncertainties(B, B_unc, x, x_unc):
    """
    Function that calculates the uncertainties of the fit (model). It is the error propagation of the function "multi_gaussian_function_for_ODR", derivating x and the parameters
    B[0]=background, B[1]=amplitude1, B[2]=mean1, B[3]=FWHM1, B[4]=amplitude2, B[5]=mean2, B[6]=FWHM2, B[7]=mean3,...
    B_unc are the uncertainties of B.
    Number of gaussians to fit should be equal to (len(B)-1)//3
    """
    N_gauss = (len(B)-1)//3 #number of gaussians to fit
    T1, TE3, TE4, TE5 = 0,0,0,0
    for i in range(0,N_gauss):
        #print('i:', 3*i+1, 3*i+2, 3*i+3)
        
        # gaussian parameters of this iteration
        I_i = B[3*i+1]
        m_i = B[3*i+2]
        w_i = B[3*i+3]
        
        # uncertainties of above parameters
        dI_i = B_unc[3*i+1]
        dm_i = B_unc[3*i+2]
        dw_i = B_unc[3*i+3]
        
        # furmula of the uncertainty
        TEi = np.exp(-4*np.log(2)*(x-m_i)**2 / w_i**2)
        T1i = I_i*(x-m_i)/(w_i**2) * TEi
        T1 = T1 - 8*np.log(2)*x_unc*T1i
        T3i = dI_i
        T4i = 8*np.log(2)*I_i*(x-m_i)/(w_i**2) * dm_i
        T5i = 8*np.log(2)*I_i*(x-m_i)**2/(w_i**3) * dw_i
        TE3 = TE3 + (TEi*T3i)**2 
        TE4 = TE4 + (TEi*T4i)**2 
        TE5 = TE5 + (TEi*T5i)**2 
    
    T2 = B_unc[0]
    
    return np.sqrt(T1**2 + T2**2 + TE3 + TE4 + TE5)



def gaussian_fit(x_data, y_data, init_amplitude, init_mean, init_FWHM):
    
    init_stddev = init_FWHM / (2*np.sqrt(2*np.log(2)))
    initial_guess = [init_amplitude, init_mean, init_stddev]
    params, covariance = curve_fit(gaussian_function, x_data, y_data, p0=initial_guess)
    amplitude, mean, stddev = params
    FWHM = 2*np.sqrt(2*np.log(2))*stddev

    x_fit = np.linspace(min(x_data), max(x_data), 500)
    y_fit = gaussian_function(x_fit, amplitude, mean, stddev)

    return {'x_fit':x_fit, 'y_fit':y_fit, 'amplitude':amplitude, 'mean':mean, 'stddev':stddev, 'FWHM':FWHM}

def double_gaussian_function(x, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2):
    gf1 = gaussian_function(x, amplitude1, mean1, stddev1)
    gf2 = gaussian_function(x, amplitude2, mean2, stddev2)
    return gf1+gf2

def double_gaussian_fit(x_data, y_data, init_amplitude1, init_mean1, init_FWHM1, init_amplitude2, init_mean2, init_FWHM2):
    
    init_stddev1 = init_FWHM1 / (2*np.sqrt(2*np.log(2)))
    init_stddev2 = init_FWHM2 / (2*np.sqrt(2*np.log(2)))
    initial_guess = [init_amplitude1, init_mean1, init_stddev1, init_amplitude2, init_mean2, init_stddev2]
    params, covariance = curve_fit(double_gaussian_function, x_data, y_data, p0=initial_guess)
    amplitude1, mean1, stddev1, amplitude2, mean2, stddev2 = params
    FWHM1 = 2*np.sqrt(2*np.log(2))*stddev1
    FWHM2 = 2*np.sqrt(2*np.log(2))*stddev2

    x_fit = np.linspace(min(x_data), max(x_data), 500)
    y_fit = double_gaussian_function(x_fit, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2)

    return {'x_fit':x_fit, 'y_fit':y_fit, 'amplitude1':amplitude1, 'mean1':mean1, 'stddev1':stddev1, 'FWHM1':FWHM1, 'amplitude2':amplitude2, 'mean2':mean2, 'stddev2':stddev2, 'FWHM2':FWHM2}


def reduced_chi_squared(y_fit_, y_data_, y_unc_data_, n_parameters):
    """
    Calculation of reduced Chi square of a fit. 
    It uses only y-errors.
    """
    chi_square = np.sum(((y_data_ - y_fit_) / y_unc_data_)**2)  # Using only y-errors
    dof = len(y_data_) - n_parameters # Degrees of freedom
    reduced_chi_squared_ = chi_square / dof
    
    return [reduced_chi_squared_, chi_square]

def averaged_intensity_along_slit(I_ij, f_j, t_exp): 
    """
    - Inputs:
        - I_ij = 2D array of intensities (rows or y axis represent the spatial dimension (along the slit), and columns or y axis represent the wavelength direction).
        - f_j = 1D array of conversion factors (they depend on the wavelength dimension).
        - t_exp = exposure time of the spectrum, in seconds.
    - Outputs:
        - I_j = 1D array with the averaged intensities along the slit.
        - I_unc_j = 1D array with the uncertainties of I_j.
    """
    I_j = I_ij.mean(axis=0)
    
    N_j = I_ij.count(axis=0) # Number of valid rows per column (masked values are not taken into account)
    I_unc_j = np.sqrt(f_j/N_j/t_exp*I_j)
    
    return [I_j, I_unc_j]
    


############################################################################################
############################################################################################
# Functions for the change of coordinates of pixels

def detector_pixels_to_image_pixels_no_binned(Vdet, vdet_rs, vdet_re, vim_rs, vim_re):
    """
    The pixel number "Vdet" in the detector ("det") has number "Vim" in the image ("im"). This functions does this conversion. It is just a linear function.
    This function considers that the image is not binned.
    In the variables here, letter "v" means the direction, it could be "x" or "y".
    
    Info about variables:
        - Vdet = pixel in detector coordinates that we want to transform to coordinates in our image. For example the position of a defective pixel or the KBr coated part that we find in literature (they appear in detector coordinates).
        - Vim = Vdet converted to image coordinates. It is the number we are looking for. 
        - vdet_rs = pixel of readout start in detector coordinates. You can find it in the header as 'DETXSTRT' (x axis) or 'DETYSTRT' (y axis).
        - vdet_re = pixel of readout end in detector coordinates. You can find it in the header as 'DETXEND' (x axis) or 'DETYEND' (y axis).
        - vim_rs = pixel of readout start in image coordinates. It is always 0 in y axis (spatial direction), and "vim_length - 1" in x axis (wavelength direction).
        - vim_length = total number of pixels in the image in the direction we are working on ("v" direction in this case).
        - vim_re = pixel of readout end in image coordinates. It is always 0 in x axis (wavelength direction), and "vim_length - 1" in y axis (spatial direction). 
    """
    m = (vdet_re-vdet_rs)/(vim_re-vim_rs)
    b = vim_rs - m * vdet_rs
    Vim = m * Vdet  + b
    return Vim #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(Vim) in case s_px is one number, or np.floor(Vim).astype(int) in case s_px is an array.

def image_pixels_no_binned_to_detector_pixels(Vim, vdet_rs, vdet_re, vim_rs, vim_re):
    """
    This functions is the inverse of the function "detector_pixels_to_image_pixels_no_binned()".
    The pixel number "Vim" in the image ("im") has number "Vdet" in the detector ("det"). This functions does this conversion. It is just a linear function.
    This function considers that the image is not binned.
    In the variables here, letter "v" means the direction, it could be "x" or "y".
    
    Info about variables:
        - Vim = pixel in image coordinates that we want to transform to (original) coordinates in the detector. For example the position of a defective pixel or the KBr coated part that we find in literature (they appear in detector coordinates).
        - Vdet = Vim converted to detector coordinates. It is the number we are looking for. 
        - vdet_rs = pixel of readout start in detector coordinates. You can find it in the header as 'DETXSTRT' (x axis) or 'DETYSTRT' (y axis).
        - vdet_re = pixel of readout end in detector coordinates. You can find it in the header as 'DETXEND' (x axis) or 'DETYEND' (y axis).
        - vim_rs = pixel of readout start in image coordinates. It is always 0 in y axis (spatial direction), and "vim_length - 1" in x axis (wavelength direction).
        - vim_length = total number of pixels in the image in the direction we are working on ("v" direction in this case).
        - vim_re = pixel of readout end in image coordinates. It is always 0 in x axis (wavelength direction), and "vim_length - 1" in y axis (spatial direction). 
    """
    mm = (vdet_re-vdet_rs)/(vim_re-vim_rs)
    bb = vim_rs - mm * vdet_rs
    m = 1/mm
    b=-bb/m
    Vdet = m * Vim + b
    return Vdet #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(Vdet) in case s_px is one number, or np.floor(Vdet).astype(int) in case s_px is an array.

def image_pixels_to_preliminary_wavelength(w_px, fits_header): 
    """
    This function converts the pixel numbers of the x axis (wavelength direction) in the image to preliminary wavelength values (in Angstrom) according to the information of the fits header. 
    
    To perform this conversion we consider a straight line y=mx+b where:
    - y= preliminary wavelengths.
    - x= pixel number (or index, it's the same, because it starts at 0) of the wavelength dimension (x axis).
    - m= slope, which is given in the header with the keyword 'CDELT1'.
    - b= intercept, it is calculated with the reference values of x and y axis, given in the header with the keywords 'CRPIX1' and 'CRPIY1', respectively.
    """
    x = w_px
    m = fits_header['CDELT1'] #slope # Axis increments along axis 1 (Angstrom)
    x_ref = fits_header['CRPIX1'] # Reference pixel along axis 1 
    y_ref = fits_header['CRVAL1'] # Value at reference pixel of axis 1 (Angstrom)   
    b = y_ref - m*x_ref
    w_Ang = m*x+b
    return w_Ang

def preliminary_wavelength_to_image_pixels(w_Ang, fits_header):
    """
    This function is the inverse of image_pixels_to_preliminary_wavelength()
    
    To perform this conversion we consider a straight line y=mx+b where:
    - y= preliminary wavelengths.
    - x= pixel number (or index, it's the same, because it starts at 0) of the wavelength dimension (x axis).
    - mm= slope, which is given in the header with the keyword 'CDELT1'.
    - bb= intercept, it is calculated with the reference values of x and y axis, given in the header with the keywords 'CRPIX1' and 'CRPIY1', respectively.
    - m=1/mm = slope of the inverse of y=mx+b
    - b=-bb/mm = intercept of the inverse of y=mx+b
    """
    x = w_Ang
    mm = fits_header['CDELT1'] #slope # Axis increments along axis 1 (Angstrom)
    x_ref = fits_header['CRPIX1'] # Reference pixel along axis 1 
    y_ref = fits_header['CRVAL1'] # Value at reference pixel of axis 1 (Angstrom)   
    bb = y_ref - mm*x_ref
    m = 1/mm
    b = -bb/mm
    w_px = m * x + b 
    return w_px #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(w_px) in case s_px is one number, or np.floor(w_px).astype(int) in case s_px is an array.

def image_pixels_to_preliminary_spatial(s_px, fits_header): 
    """
    This function converts the pixel numbers of the x axis (spatial direction) in the image to preliminary spatial direction values (in arcsec) according to the information of the fits header. 
    
    To perform this conversion we consider a straight line y=mx+b where:
    - y= preliminary spatial direction.
    - x= pixel number (or index, it's the same, because it starts at 0) of the spatial direction (y axis).
    - m= slope, which is given in the header with the keyword 'CDELT2'.
    - b= intercept, it is calculated with the reference values of x and y axis, given in the header with the keywords 'CRPIX2' and 'CRPIY2', respectively.
    """
    x = s_px
    m = fits_header['CDELT2'] #slope # Axis increments along axis 2 (arcsec)
    x_ref = fits_header['CRPIX2'] # Reference pixel along axis 2 
    y_ref = fits_header['CRVAL2'] # Value at reference pixel of axis 2 (arcsec)   
    b = y_ref - m*x_ref
    s_arcsec = m*x+b
    return s_arcsec

def preliminary_space_to_image_pixels(s_arcsec, fits_header):
    """
    This function is the inverse of image_pixels_to_preliminary_space()
    
    To perform this conversion we consider a straight line y=mx+b where:
    - y= preliminary spatial direction.
    - x= pixel number (or index, it's the same, because it starts at 0) of the spatial direction (y axis).
    - mm= slope, which is given in the header with the keyword 'CDELT2'.
    - bb= intercept, it is calculated with the reference values of x and y axis, given in the header with the keywords 'CRPIX2' and 'CRPIY2', respectively.
    - m=1/mm = slope of the inverse of y=mx+b
    - b=-bb/mm = intercept of the inverse of y=mx+b
    """
    x = s_arcsec
    mm = fits_header['CDELT2'] #slope # Axis increments along axis 2 (arcsec)
    x_ref = fits_header['CRPIX2'] # Reference pixel along axis 2 
    y_ref = fits_header['CRVAL2'] # Value at reference pixel of axis 2 (arcsec)   
    bb = y_ref - mm*x_ref
    m = 1/mm
    b = -bb/mm
    s_px = m * x + b 
    
    return s_px #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(s_px) in case s_px is one number, or np.floor(s_px).astype(int) in case s_px is an array.



############################################################################################
############################################################################################
# Lines/functions to make SUMER class shorter

def detector_features_in_detector_coords_to_image_coords(detector_AB, KBr_pxdet, Lya_pxdet, header, w_len):
    """
    This function uses the function "detector_pixels_to_image_pixels_no_binned()" to convert features of the detector (in detector coordinates) to image coordinates. 
    These features are: x axis (wavelength direction) position of the coated (KBr) part of the detector, the bared part, and the Ly-alpha attenuators. 
    """
    AB = header['DETECTOR']
    
    # KBr coated part of the detector
    KBr_px=[np.nan, np.nan]
    KBr_px[0] = detector_pixels_to_image_pixels_no_binned(Vdet=KBr_pxdet[AB][0], vdet_rs=header['DETXSTRT'], vdet_re=header['DETXEND'], vim_rs=w_len-1, vim_re=0)
    KBr_px[1] = detector_pixels_to_image_pixels_no_binned(Vdet=KBr_pxdet[AB][1], vdet_rs=header['DETXSTRT'], vdet_re=header['DETXEND'], vim_rs=w_len-1, vim_re=0)
    
    # Ly-alpha attenuators
    Lya_px=[np.nan, np.nan, np.nan, np.nan]
    Lya_px[0] = detector_pixels_to_image_pixels_no_binned(Vdet=Lya_pxdet[AB][0], vdet_rs=header['DETXSTRT'], vdet_re=header['DETXEND'], vim_rs=w_len-1, vim_re=0)
    Lya_px[1] = detector_pixels_to_image_pixels_no_binned(Vdet=Lya_pxdet[AB][1], vdet_rs=header['DETXSTRT'], vdet_re=header['DETXEND'], vim_rs=w_len-1, vim_re=0)
    Lya_px[2] = detector_pixels_to_image_pixels_no_binned(Vdet=Lya_pxdet[AB][2], vdet_rs=header['DETXSTRT'], vdet_re=header['DETXEND'], vim_rs=w_len-1, vim_re=0)
    Lya_px[3] = detector_pixels_to_image_pixels_no_binned(Vdet=Lya_pxdet[AB][3], vdet_rs=header['DETXSTRT'], vdet_re=header['DETXEND'], vim_rs=w_len-1, vim_re=0)
    
    return [KBr_px, Lya_px]
#TODO: we have supposed that the image is not binned, if it is binned, this is wrong!!!

def detector_defectives_in_detector_coords_to_image_coords(defect_detectorA_xy_pxdet, defects_detectorA_xy_pxdet, header, w_len, s_len):
    """
    This function uses the function "detector_pixels_to_image_pixels_no_binned()" to convert defective pixels of the detector (in detector coordinates) to image coordinates. 
    """
    # Pixel of the literature
    xim_l = detector_pixels_to_image_pixels_no_binned(Vdet=defect_detectorA_xy_pxdet[0], vdet_rs=header['DETXSTRT'], vdet_re=header['DETXEND'], vim_rs=w_len-1, vim_re=0)
    yim_l = detector_pixels_to_image_pixels_no_binned(Vdet=defect_detectorA_xy_pxdet[1], vdet_rs=header['DETYSTRT'], vdet_re=header['DETYEND'], vim_rs=0, vim_re=s_len-1)
    defect_detectorA_xy_px = [int(xim_l), int(yim_l)]
    
    # All pixels I consider are wrong
    defects_detectorA_xy_px = []
    for xdet_i,ydet_i in defects_detectorA_xy_pxdet:
        xim_i = detector_pixels_to_image_pixels_no_binned(Vdet=xdet_i, vdet_rs=header['DETXSTRT'], vdet_re=header['DETXEND'], vim_rs=w_len-1, vim_re=0)
        yim_i = detector_pixels_to_image_pixels_no_binned(Vdet=ydet_i, vdet_rs=header['DETYSTRT'], vdet_re=header['DETYEND'], vim_rs=0, vim_re=s_len-1)
        defects_detectorA_xy_px.append([int(xim_i), int(yim_i)])
    
    return [defect_detectorA_xy_px, defects_detectorA_xy_px]
    
def mask_defective_pixels(intensity_nomask, defects_detectorA_xy_px):
    """
    This function put masks in defective pixels. 
    """
    # 1) Make a copy of the array of intensities
    intensity_negatives = np.copy(intensity_nomask)
    
    # 2) Put weird value (negative, e.g. -999) to these defective pixels
    for xim_,yim_ in defects_detectorA_xy_px:
        intensity_negatives[int(yim_), int(xim_)] = -999 #[y_,x_] because, remember, python reads coordinates in a 2D-array as (row,column) which is (y,x); (however when it plots, it does (x,y) which is (column,row).
    
    # 3) Mask the negative values
    intensity_masked = np.ma.masked_less(intensity_negatives, 0) #values less than zero are masked
    
    return intensity_masked

def create_profile(SUMER_object, w_px_range, s_px_range, rect_label='Region for profile', rect_linewidth=2, rect_edgecolor='black', rect_linestyle='-'):
    s0, s1 = s_px_range
    w0, w1 = w_px_range
    
    # crop 2D-array of data and create profile
    intensity_crop = SUMER_object.intensity[s0:s1+1, w0:w1+1]
    y_profile = intensity_crop.mean(axis=0)
    x_profile = np.arange(w0, w1+1)
    
    # Create a Rectangle patch for the 2D-spectrum showing the region used for the profile
    rect = patches.Rectangle(xy=[w0,s0], width=w1-w0, height=s1-s0, label=rect_label, linewidth=rect_linewidth, edgecolor=rect_edgecolor, linestyle=rect_linestyle, facecolor='none')
    return [x_profile, y_profile, intensity_crop, rect]
    
def create_profile_calibrated(SUMER_object, w_calAng_range, s_px_range, rect_label='Region for profile', rect_linewidth=2, rect_edgecolor='black', rect_linestyle='-'):
    
    w_px_range, w_calAng_range_adapted = SUMER_object.calibrated_wavelength_range_to_pixel_range(w_calAng_range=w_calAng_range)
    
    s0, s1 = s_px_range
    w0, w1 = w_px_range
    wc0, wc1 = w_calAng_range_adapted
    
    # crop 2D-array of data and create profile
    intensity_crop = SUMER_object.intensity[s0:s1+1, w0:w1+1]
    y_profile = intensity_crop.mean(axis=0)
    x_profile = SUMER_object.image_pixel_to_calibrated_wavelength(px_image=np.arange(w0, w1+1))
    
    # Create a Rectangle patch for the 2D-spectrum showing the region used for the profile
    rect = patches.Rectangle(xy=[wc0,s0], width=wc1-wc0, height=s1-s0, label=rect_label, linewidth=rect_linewidth, edgecolor=rect_edgecolor, linestyle=rect_linestyle, facecolor='none')
    return [x_profile, y_profile, intensity_crop, rect]

def plot_detector_features_profile(SUMER_object, ax, s_px_range):
    
    #print('pixel number of full detector | pixel number of our image')
        
    # KBr coating
    ax.axvline(x=SUMER_object.KBr_px[0], color='red', linestyle='--', linewidth=1, label=f"KBr coating") 
    ax.axvline(x=SUMER_object.KBr_px[1], color='red', linestyle='--', linewidth=1) 
    #print('KBr coating [x_end, x_start]:', SUMER_object.KBr_pxdet[SUMER_object.header['DETECTOR']], '|', SUMER_object.KBr_px)
    
    # Ly-alpha attenuators
    ax.axvline(x=SUMER_object.Lya_px[0], color='magenta', linestyle='-.', linewidth=1, label=r"Ly-$\alpha$ attenuators") 
    ax.axvline(x=SUMER_object.Lya_px[1], color='magenta', linestyle='-.', linewidth=1) 
    ax.axvline(x=SUMER_object.Lya_px[2], color='magenta', linestyle='-.', linewidth=1) 
    ax.axvline(x=SUMER_object.Lya_px[3], color='magenta', linestyle='-.', linewidth=1) 
    #print('Ly-alpha attenuators [x_right_end, x_right_start, x_left_end, x_left_start]:', SUMER_object.Lya_pxdet[SUMER_object.header['DETECTOR']], '|', SUMER_object.Lya_px)
    
    # Reference pixels 
    ax.axvline(x=SUMER_object.header['CRPIX1'], color='red', linestyle=':', linewidth=1, label=f'Reference x and y pixels') 
    
    # Defect of detector A
    if SUMER_object.header['DETECTOR']=='A': 
        qwerty=0 # to avoid too many lines in the legend
        for xim_, yim_ in SUMER_object.defects_detectorA_xy_px:
            if s_px_range[0]<=yim_<=s_px_range[-1]:
                ax.axvline(x=xim_, color='orange', linestyle=':', linewidth=1)
                qwerty=1
        if s_px_range[0]<=yim_<=s_px_range[-1] and qwerty==1: ax.plot([],[], color='orange', linestyle=':', linewidth=1, label="Defective pixels") #for the label
        #print('Largest defect in detector A [x,y]:', SUMER_object.defect_detectorA_xy_pxdet, '|', SUMER_object.defect_detectorA_xy_px)
    
    # Detector boundaries
    ax.axvline(x=SUMER_object.w_0det_px, color='purple', linestyle='-', linewidth=1, label=f"Detector boundaries") 
    ax.axvline(x=SUMER_object.w_1023det_px, color='purple', linestyle='-', linewidth=1) 
    
    return ax


def plot_detector_features_imshow(SUMER_object, ax):
    
    #print('pixel number of full detector | pixel number of our image')
        
    # KBr coating
    ax.axvline(x=SUMER_object.KBr_px[0], color='red', linestyle='--', linewidth=1, label=f"KBr coating") 
    ax.axvline(x=SUMER_object.KBr_px[1], color='red', linestyle='--', linewidth=1) 
    #print('KBr coating [x_end, x_start]:', SUMER_object.KBr_pxdet[SUMER_object.header['DETECTOR']], '|', SUMER_object.KBr_px)
    
    # Ly-alpha attenuators
    ax.axvline(x=SUMER_object.Lya_px[0], color='magenta', linestyle='-.', linewidth=1, label=r"Ly-$\alpha$ attenuators") 
    ax.axvline(x=SUMER_object.Lya_px[1], color='magenta', linestyle='-.', linewidth=1) 
    ax.axvline(x=SUMER_object.Lya_px[2], color='magenta', linestyle='-.', linewidth=1) 
    ax.axvline(x=SUMER_object.Lya_px[3], color='magenta', linestyle='-.', linewidth=1) 
    #print('Ly-alpha attenuators [x_right_end, x_right_start, x_left_end, x_left_start]:', SUMER_object.Lya_pxdet[SUMER_object.header['DETECTOR']], '|', SUMER_object.Lya_px)
    
    # Reference pixels 
    ax.axvline(x=SUMER_object.header['CRPIX1'], color='red', linestyle=':', linewidth=1, label=f'Reference x and y pixels') 
    ax.axhline(y=SUMER_object.header['CRPIX2'], color='red', linestyle=':', linewidth=1)
    
    # Defect of detector A
    if SUMER_object.header['DETECTOR']=='A': 
        for xim_, yim_ in SUMER_object.defects_detectorA_xy_px:
            ax.scatter(xim_, yim_, s=20, marker='+', color='orange')
        ax.scatter(SUMER_object.defects_detectorA_xy_px[0][0], SUMER_object.defects_detectorA_xy_px[0][1], s=40, marker='+', color='orange', label='Defective pixels')
        #print('Largest defect in detector A [x,y]:', SUMER_object.defect_detectorA_xy_pxdet, '|', SUMER_object.defect_detectorA_xy_px)
    
    # Detector boundaries
    ## x axis (wavelength direction)
    ax.axvline(x=SUMER_object.w_0det_px, color='purple', linestyle='-', linewidth=1, label=f"Detector boundaries") 
    ax.axvline(x=SUMER_object.w_1023det_px, color='purple', linestyle='-', linewidth=1) 
    ## y axis (spatial direction) (in case of 2D-spectrum)
    ax.axhline(y=SUMER_object.s_0det_px, color='purple', linestyle='-', linewidth=1) 
    ax.axhline(y=SUMER_object.s_359det_px, color='purple', linestyle='-', linewidth=1) 
    
    return ax
    
def plot_detector_features_imshow_calibrated(SUMER_object, ax):
    
    #print('pixel number of full detector | pixel number of our image')
        
    # KBr coating
    ax.axvline(x=SUMER_object.KBr_calAng[0], color='red', linestyle='--', linewidth=1, label=f"KBr coating") 
    ax.axvline(x=SUMER_object.KBr_calAng[1], color='red', linestyle='--', linewidth=1) 
    #print('KBr coating [x_end, x_start]:', SUMER_object.KBr_pxdet[SUMER_object.header['DETECTOR']], '|', SUMER_object.KBr_px)
    
    # Ly-alpha attenuators
    ax.axvline(x=SUMER_object.Lya_calAng[0], color='magenta', linestyle='-.', linewidth=1, label=r"Ly-$\alpha$ attenuators") 
    ax.axvline(x=SUMER_object.Lya_calAng[1], color='magenta', linestyle='-.', linewidth=1) 
    ax.axvline(x=SUMER_object.Lya_calAng[2], color='magenta', linestyle='-.', linewidth=1) 
    ax.axvline(x=SUMER_object.Lya_calAng[3], color='magenta', linestyle='-.', linewidth=1) 
    #print('Ly-alpha attenuators [x_right_end, x_right_start, x_left_end, x_left_start]:', SUMER_object.Lya_pxdet[SUMER_object.header['DETECTOR']], '|', SUMER_object.Lya_px)
    
    # Reference pixels 
    #ax.axvline(x=SUMER_object.slope_cal * SUMER_object.header['CRPIX1'] + SUMER_object.intercept_cal, color='red', linestyle=':', linewidth=1, label=f'Reference x and y pixels') 
    #ax.axhline(y=SUMER_object.header['CRPIX2'], color='red', linestyle=':', linewidth=1)
    
    # Defect of detector A
    if SUMER_object.header['DETECTOR']=='A': 
        for i in range(len(SUMER_object.defects_detectorA_w_calAng)):
            ax.scatter(SUMER_object.defects_detectorA_w_calAng[i], SUMER_object.defects_detectorA_s_px[i], s=20, marker='+', color='orange')
        ax.scatter(SUMER_object.defects_detectorA_w_calAng[0], SUMER_object.defects_detectorA_s_px[0], s=40, marker='+', color='orange', label='Defective pixels')
        #print('Largest defect in detector A [x,y]:', SUMER_object.defect_detectorA_xy_pxdet, '|', SUMER_object.defect_detectorA_xy_px)
    
    # Detector boundaries
    ## x axis (wavelength direction)
    ax.axvline(x=SUMER_object.w_0det_calAng, color='purple', linestyle='-', linewidth=1, label=f"Detector boundaries") 
    ax.axvline(x=SUMER_object.w_1023det_calAng, color='purple', linestyle='-', linewidth=1) 
    ## y axis (spatial direction) (in case of 2D-spectrum)
    ax.axhline(y=SUMER_object.s_0det_px, color='purple', linestyle='-', linewidth=1) 
    ax.axhline(y=SUMER_object.s_359det_px, color='purple', linestyle='-', linewidth=1) 
    
    return ax

def plot_detector_features_profile_calibrated(SUMER_object, ax, s_px_range):
    
    #print('pixel number of full detector | pixel number of our image')
        
    # KBr coating
    ax.axvline(x=SUMER_object.KBr_calAng[0], color='red', linestyle='--', linewidth=1, label=f"KBr coating") 
    ax.axvline(x=SUMER_object.KBr_calAng[1], color='red', linestyle='--', linewidth=1) 
    #print('KBr coating [x_end, x_start]:', SUMER_object.KBr_pxdet[SUMER_object.header['DETECTOR']], '|', SUMER_object.KBr_px)
    
    # Ly-alpha attenuators
    ax.axvline(x=SUMER_object.Lya_calAng[0], color='magenta', linestyle='-.', linewidth=1, label=r"Ly-$\alpha$ attenuators") 
    ax.axvline(x=SUMER_object.Lya_calAng[1], color='magenta', linestyle='-.', linewidth=1) 
    ax.axvline(x=SUMER_object.Lya_calAng[2], color='magenta', linestyle='-.', linewidth=1) 
    ax.axvline(x=SUMER_object.Lya_calAng[3], color='magenta', linestyle='-.', linewidth=1) 
    #print('Ly-alpha attenuators [x_right_end, x_right_start, x_left_end, x_left_start]:', SUMER_object.Lya_pxdet[SUMER_object.header['DETECTOR']], '|', SUMER_object.Lya_px)
    
    # Reference pixels 
    #ax.axvline(x=SUMER_object.slope_cal * SUMER_object.header['CRPIX1'] + SUMER_object.intercept_cal, color='red', linestyle=':', linewidth=1, label=f'Reference x and y pixels') 
    
    # Defect of detector A
    if SUMER_object.header['DETECTOR']=='A': 
        qwerty=0 # to avoid too many lines in the legend
        for i, yim_ in enumerate(SUMER_object.defects_detectorA_s_px):
            if s_px_range[0]<=yim_<=s_px_range[-1]:
                ax.axvline(x=SUMER_object.defects_detectorA_w_calAng[i], color='orange', linestyle=':', linewidth=1)
                qwerty=1
        if s_px_range[0]<=yim_<=s_px_range[-1] and qwerty==1: ax.plot([],[], color='orange', linestyle=':', linewidth=1, label="Defective pixels") #for the label
        #print('Largest defect in detector A [x,y]:', SUMER_object.defect_detectorA_xy_pxdet, '|', SUMER_object.defect_detectorA_xy_px)
    
    # Detector boundaries
    ## x axis (wavelength direction)
    ax.axvline(x=SUMER_object.w_0det_calAng, color='purple', linestyle='-', linewidth=1, label=f"Detector boundaries") 
    ax.axvline(x=SUMER_object.w_1023det_calAng, color='purple', linestyle='-', linewidth=1) 
    
    return ax
    
def functions_for_secondary_axes(SUMER_object, secondary_w_axis='preliminary wavelength', secondary_s_axis='original detector pixels'):
    """
    Secondary axis with the conversion of pixel reference and/or magnitudes 
    Inputs:
        - secondary_w_axis: it should be the string 'original detector pixels', 'preliminary wavelength', or a list with [newaxis_x, oldaxis_x, secondary_xlabel]
        - secondary_s_axis: it should be the string 'original detector pixels', 'preliminary arcseconds', or a list with [newaxis_y, oldaxis_y, secondary_ylabel]
    """
    # wavelength dimension (x axis)
    if secondary_w_axis=='original detector pixels':
        newaxis_x = SUMER_object.w_image_pixels_no_binned_to_detector_pixels
        oldaxis_x = SUMER_object.w_detector_pixels_to_image_pixels_no_binned
        secondary_xlabel = 'Wavelength dimension (original detector pixels)'
    elif secondary_w_axis=='preliminary wavelength':
        newaxis_x = SUMER_object.w_image_pixels_to_preliminary_wavelength
        oldaxis_x = SUMER_object.w_preliminary_wavelength_to_image_pixels
        secondary_xlabel = f'Preliminary wavelength ({SUMER_object.header["CUNIT1"]})'
    elif secondary_w_axis is not str:
        newaxis_x, oldaxis_x, secondary_xlabel = secondary_w_axis
    
    # spatial dimension (y axis)
    if secondary_s_axis=='original detector pixels':
        newaxis_y = SUMER_object.s_image_pixels_no_binned_to_detector_pixels
        oldaxis_y = SUMER_object.s_detector_pixels_to_image_pixels_no_binned
        secondary_ylabel = f'Spatial dimension (original detector pixels)'
    elif secondary_s_axis=='preliminary arcseconds':
        newaxis_y = SUMER_object.s_image_pixels_to_preliminary_spatial
        oldaxis_y = SUMER_object.s_preliminary_space_to_image_pixels
        secondary_ylabel = f'Preliminary spatial coordinates ({SUMER_object.header["CUNIT2"]})'
    elif secondary_s_axis is not str:
        newaxis_y, oldaxis_y, secondary_ylabel = secondary_s_axis
    
    return [newaxis_x, oldaxis_x, secondary_xlabel, newaxis_y, oldaxis_y, secondary_ylabel]
    
############################################################################################
############################################################################################

def create_quadrilateral(xll, yll, xlr, ylr, xhr, yhr, xhl, yhl, N):
    x_low = np.linspace(xll, xlr, N)
    y_low = np.linspace(yll, ylr, N)
    x_high = np.linspace(xhl, xhr, N)
    y_high = np.linspace(yhl, yhr, N)
    x_left = np.linspace(xhl, xll, N)
    y_left = np.linspace(yhl, yll, N)
    x_right = np.linspace(xlr, xhr, N)
    y_right = np.linspace(ylr, yhr, N)
    #return [x_low, y_low, x_high, y_high, x_left, y_left, x_right, y_right]
    x_quadrilateral = np.concatenate([x_low, x_right, x_high, x_left])
    y_quadrilateral = np.concatenate([y_low, y_right, y_high, y_left])
    return [x_quadrilateral, y_quadrilateral]

def create_rectangle(x_left, x_right, y_low, y_high, N):
    X_low = np.linspace(x_left, x_right, N)
    Y_low = y_low * np.ones(N)
    X_right = x_right * np.ones(N)
    Y_right = np.linspace(y_low, y_high, N)
    X_high = X_low[::-1]
    Y_high = y_high * np.ones(N)
    X_left = x_left * np.ones(N)
    Y_left = Y_right[::-1]
    x_rectangle = np.concatenate([X_low, X_right, X_high, X_left])
    y_rectangle = np.concatenate([Y_low, Y_right, Y_high, Y_left])
    return [x_rectangle, y_rectangle]

#############################################################
#############################################################
# 

def see_wavelength_ranges(spectrum_image, wavelength_range_spectroheliogram, wavelength_range_spectroheliogram_bckg, slope_list, intercept_list, row_list, x_lim='auto'):
    # Calculate number of pixels of each profile
    pixel_low = wavelength_to_pixels(wavelength=wavelength_range_spectroheliogram[0], slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
    pixel_high = wavelength_to_pixels(wavelength=wavelength_range_spectroheliogram[1], slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
    N_pixels_range = int(round(pixel_high - pixel_low))
    ## Background
    pixel_low_bckg = wavelength_to_pixels(wavelength=wavelength_range_spectroheliogram_bckg[0], slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
    pixel_high_bckg = wavelength_to_pixels(wavelength=wavelength_range_spectroheliogram_bckg[1], slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
    N_pixels_range_bckg = int(round(pixel_high_bckg - pixel_low_bckg))



    pixel_range_pixcenter_list0, pixel_range_float_list0, pixel_range_pixcenter_pixelfixed_list0, pixel_range_pixcenter_bckg_list0, pixel_range_float_bckg_list0, pixel_range_pixcenter_pixelfixed_bckg_list0 = [],[],[],[],[],[]
    pixel_range_pixcenter_list1, pixel_range_float_list1, pixel_range_pixcenter_pixelfixed_list1, pixel_range_pixcenter_bckg_list1, pixel_range_float_bckg_list1, pixel_range_pixcenter_pixelfixed_bckg_list1 = [],[],[],[],[],[]
    for i in range(len(slope_list)):
        row = row_list[i]

        # Line
        rwtcp = range__wavelength_to_closest_pixels(wavelength_range=wavelength_range_spectroheliogram, slope_cal=slope_list[i], intercept_cal=intercept_list[i])
        pixel_range_pixcenter_list0.append(rwtcp[1][0])
        pixel_range_pixcenter_list1.append(rwtcp[1][1])
        pixel_range_float_list0.append(rwtcp[2][0])
        pixel_range_float_list1.append(rwtcp[2][1])
        rwtcppf = range__wavelength_to_closest_pixels_pixelfixed(wavelength_low=wavelength_range_spectroheliogram[0], N_pixels=N_pixels_range, slope_cal=slope_list[i], intercept_cal=intercept_list[i])
        pixel_range_pixcenter_pixelfixed_list0.append(rwtcppf[1][0])
        pixel_range_pixcenter_pixelfixed_list1.append(rwtcppf[1][1])

        # Background
        rwtcp_bckg = range__wavelength_to_closest_pixels(wavelength_range=wavelength_range_spectroheliogram_bckg, slope_cal=slope_list[i], intercept_cal=intercept_list[i])
        pixel_range_pixcenter_bckg_list0.append(rwtcp_bckg[1][0])
        pixel_range_pixcenter_bckg_list1.append(rwtcp_bckg[1][1])
        pixel_range_float_bckg_list0.append(rwtcp_bckg[2][0])
        pixel_range_float_bckg_list1.append(rwtcp_bckg[2][1])
        rwtcppf_bckg = range__wavelength_to_closest_pixels_pixelfixed(wavelength_low=wavelength_range_spectroheliogram_bckg[0], N_pixels=N_pixels_range_bckg, slope_cal=slope_list[i], intercept_cal=intercept_list[i])
        pixel_range_pixcenter_pixelfixed_bckg_list0.append(rwtcppf_bckg[1][0])
        pixel_range_pixcenter_pixelfixed_bckg_list1.append(rwtcppf_bckg[1][1])
    
    if x_lim=='auto':
        all_pixels_of_range = np.concatenate([pixel_range_pixcenter_pixelfixed_list0, pixel_range_pixcenter_pixelfixed_list1, pixel_range_pixcenter_pixelfixed_bckg_list0, pixel_range_pixcenter_pixelfixed_bckg_list0])
        x_lim = [np.min(all_pixels_of_range)-2, np.max(all_pixels_of_range)+15]
    
        
        
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(spectrum_image, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title('Ranges for spectroheliogram. Decimal pixels (interpolate intensities)', fontsize=18) 
    ax.set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)

    ax.scatter(pixel_range_float_list0, row_list, s=3, color='blue')#, label='Decimal pixels (interpolate intensities)')
    ax.scatter(pixel_range_float_list1, row_list, s=3, color='blue')
    ax.scatter(pixel_range_float_bckg_list0, row_list, s=3, color='blue')
    ax.scatter(pixel_range_float_bckg_list1, row_list, s=3, color='blue')

    ax.axhline(y=slit_top_px, linestyle='--', color='green', linewidth=2, label='Slit top')
    ax.axhline(y=slit_bottom_px, linestyle='--', color='orange', linewidth=2, label='Slit bottom')
    ax.set_xlim(x_lim)

    ax.legend(fontsize=10)
    plt.show(block=False)
        
        

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(spectrum_image, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title('Ranges for spectroheliogram. Consider whole pixels, closer edges to the wl range', fontsize=18) 
    ax.set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)

    ax.scatter(pixel_range_pixcenter_list0, row_list, s=3, facecolors='none', edgecolors='cyan')#, label='Consider whole pixels,\n closer edges to the wl range')
    ax.scatter(pixel_range_pixcenter_list1, row_list, s=3, facecolors='none', edgecolors='cyan')
    ax.scatter(pixel_range_pixcenter_bckg_list0, row_list, s=3, facecolors='none', edgecolors='cyan')
    ax.scatter(pixel_range_pixcenter_bckg_list1, row_list, s=3, facecolors='none', edgecolors='cyan')

    ax.axhline(y=slit_top_px, linestyle='--', color='green', linewidth=2, label='Slit top')
    ax.axhline(y=slit_bottom_px, linestyle='--', color='orange', linewidth=2, label='Slit bottom')
    ax.set_xlim(x_lim)
    ax.legend(fontsize=10)
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(spectrum_image, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title('Ranges for spectroheliogram. Consider whole pixels, first closer to the wl range, N px fixed', fontsize=18) 
    ax.set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)

    ax.scatter(pixel_range_pixcenter_pixelfixed_list0, row_list, s=3, color='red')#, label='Consider whole pixels,\n first closer to the wl range, N px fixed')
    ax.scatter(pixel_range_pixcenter_pixelfixed_list1, row_list, s=3, color='red')
    ax.scatter(pixel_range_pixcenter_pixelfixed_bckg_list0, row_list, s=3, color='red')
    ax.scatter(pixel_range_pixcenter_pixelfixed_bckg_list1, row_list, s=3, color='red')

    ax.axhline(y=slit_top_px, linestyle='--', color='green', linewidth=2, label='Slit top')
    ax.axhline(y=slit_bottom_px, linestyle='--', color='orange', linewidth=2, label='Slit bottom')
    ax.set_xlim(x_lim)
    ax.legend(fontsize=10)
    plt.show(block=False)



    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(spectrum_image, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title(f'Ranges for spectroheliogram. Wavelength ranges comparison', fontsize=18) 
    ax.set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)    

    ax.axhline(y=slit_top_px, linestyle='--', color='green', linewidth=2, label='Slit top')
    ax.axhline(y=slit_bottom_px, linestyle='--', color='orange', linewidth=2, label='Slit bottom')

    # Set edges of the pixels
    for pixel_edge in np.arange(x_lim[0],x_lim[1]+2)-0.5:
        ax.axvline(x=pixel_edge, linestyle='-', color='white', linewidth=0.2)
    ax.plot([],[], linestyle='-', color='white', linewidth=0.2, label='pixel edges')

    ax.scatter(pixel_range_pixcenter_list0, row_list, s=3, facecolors='none', edgecolors='cyan', label='Consider whole pixels,\n closer edges to the wl range')
    ax.scatter(pixel_range_pixcenter_list1, row_list, s=3, facecolors='none', edgecolors='cyan')
    ax.scatter(pixel_range_pixcenter_bckg_list0, row_list, s=3, facecolors='none', edgecolors='cyan')
    ax.scatter(pixel_range_pixcenter_bckg_list1, row_list, s=3, facecolors='none', edgecolors='cyan')

    ax.scatter(pixel_range_float_list0, row_list, s=3, color='blue', label='Decimal pixels (interpolate intensities)')
    ax.scatter(pixel_range_float_list1, row_list, s=3, color='blue')
    ax.scatter(pixel_range_float_bckg_list0, row_list, s=3, color='blue')
    ax.scatter(pixel_range_float_bckg_list1, row_list, s=3, color='blue')

    ax.scatter(pixel_range_pixcenter_pixelfixed_list0, row_list, s=3, color='red', label='Consider whole pixels,\n first closer to the wl range, N px fixed')
    ax.scatter(pixel_range_pixcenter_pixelfixed_list1, row_list, s=3, color='red')
    ax.scatter(pixel_range_pixcenter_pixelfixed_bckg_list0, row_list, s=3, color='red')
    ax.scatter(pixel_range_pixcenter_pixelfixed_bckg_list1, row_list, s=3, color='red')

    ax.set_xlim(x_lim)
    ax.legend(fontsize=7)
    plt.show(block=False)
    

def see_wavelength_ranges_general(spectrum_image, wavelength_range, slope_list, intercept_list, row_list, x_lim='auto'):
    # Calculate number of pixels of each profile
    pixel_low = wavelength_to_pixels(wavelength=wavelength_range[0], slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
    pixel_high = wavelength_to_pixels(wavelength=wavelength_range[1], slope_cal=np.mean(slope_list), intercept_cal=np.mean(intercept_list))
    N_pixels_range = int(round(pixel_high - pixel_low))

    pixel_range_pixcenter_list0, pixel_range_float_list0, pixel_range_pixcenter_pixelfixed_list0 = [],[],[]
    pixel_range_pixcenter_list1, pixel_range_float_list1, pixel_range_pixcenter_pixelfixed_list1 = [],[],[]
    for i in range(len(slope_list)):
        row = row_list[i]

        # Line
        rwtcp = range__wavelength_to_closest_pixels(wavelength_range=wavelength_range, slope_cal=slope_list[i], intercept_cal=intercept_list[i])
        pixel_range_pixcenter_list0.append(rwtcp[1][0])
        pixel_range_pixcenter_list1.append(rwtcp[1][1])
        pixel_range_float_list0.append(rwtcp[2][0])
        pixel_range_float_list1.append(rwtcp[2][1])
        rwtcppf = range__wavelength_to_closest_pixels_pixelfixed(wavelength_low=wavelength_range[0], N_pixels=N_pixels_range, slope_cal=slope_list[i], intercept_cal=intercept_list[i])
        pixel_range_pixcenter_pixelfixed_list0.append(rwtcppf[1][0])
        pixel_range_pixcenter_pixelfixed_list1.append(rwtcppf[1][1])
    
    if x_lim=='auto':
        all_pixels_of_range = np.concatenate([pixel_range_pixcenter_pixelfixed_list0, pixel_range_pixcenter_pixelfixed_list1])
        x_lim = [np.min(all_pixels_of_range)-2, np.max(all_pixels_of_range)+15]
    
        
        
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(spectrum_image, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title('Decimal pixels (interpolate intensities)', fontsize=18) 
    ax.set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)

    ax.scatter(pixel_range_float_list0, row_list, s=3, color='blue')#, label='Decimal pixels (interpolate intensities)')
    ax.scatter(pixel_range_float_list1, row_list, s=3, color='blue')

    ax.axhline(y=slit_top_px, linestyle='--', color='green', linewidth=2, label='Slit top')
    ax.axhline(y=slit_bottom_px, linestyle='--', color='orange', linewidth=2, label='Slit bottom')
    ax.set_xlim(x_lim)

    ax.legend(fontsize=10)
    plt.show(block=False)
        
        

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(spectrum_image, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title('Consider whole pixels, closer edges to the wl range', fontsize=18) 
    ax.set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)

    ax.scatter(pixel_range_pixcenter_list0, row_list, s=3, facecolors='none', edgecolors='cyan')#, label='Consider whole pixels,\n closer edges to the wl range')
    ax.scatter(pixel_range_pixcenter_list1, row_list, s=3, facecolors='none', edgecolors='cyan')

    ax.axhline(y=slit_top_px, linestyle='--', color='green', linewidth=2, label='Slit top')
    ax.axhline(y=slit_bottom_px, linestyle='--', color='orange', linewidth=2, label='Slit bottom')
    ax.set_xlim(x_lim)
    ax.legend(fontsize=10)
    plt.show(block=False)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(spectrum_image, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title('Consider whole pixels, first closer to the wl range, N px fixed', fontsize=18) 
    ax.set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)

    ax.scatter(pixel_range_pixcenter_pixelfixed_list0, row_list, s=3, color='red')#, label='Consider whole pixels,\n first closer to the wl range, N px fixed')
    ax.scatter(pixel_range_pixcenter_pixelfixed_list1, row_list, s=3, color='red')

    ax.axhline(y=slit_top_px, linestyle='--', color='green', linewidth=2, label='Slit top')
    ax.axhline(y=slit_bottom_px, linestyle='--', color='orange', linewidth=2, label='Slit bottom')
    ax.set_xlim(x_lim)
    ax.legend(fontsize=10)
    plt.show(block=False)



    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(spectrum_image, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title(f'SOHO/SUMER, wavelength ranges comparison', fontsize=18) 
    ax.set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)    

    ax.axhline(y=slit_top_px, linestyle='--', color='green', linewidth=2, label='Slit top')
    ax.axhline(y=slit_bottom_px, linestyle='--', color='orange', linewidth=2, label='Slit bottom')

    # Set edges of the pixels
    for pixel_edge in np.arange(x_lim[0],x_lim[1]+2)-0.5:
        ax.axvline(x=pixel_edge, linestyle='-', color='white', linewidth=0.2)
    ax.plot([],[], linestyle='-', color='white', linewidth=0.2, label='pixel edges')

    ax.scatter(pixel_range_pixcenter_list0, row_list, s=3, facecolors='none', edgecolors='cyan', label='Consider whole pixels,\n closer edges to the wl range')
    ax.scatter(pixel_range_pixcenter_list1, row_list, s=3, facecolors='none', edgecolors='cyan')

    ax.scatter(pixel_range_float_list0, row_list, s=3, color='blue', label='Decimal pixels (interpolate intensities)')
    ax.scatter(pixel_range_float_list1, row_list, s=3, color='blue')

    ax.scatter(pixel_range_pixcenter_pixelfixed_list0, row_list, s=3, color='red', label='Consider whole pixels,\n first closer to the wl range, N px fixed')
    ax.scatter(pixel_range_pixcenter_pixelfixed_list1, row_list, s=3, color='red')

    ax.set_xlim(x_lim)
    ax.legend(fontsize=7)
    plt.show(block=False)
    


#############################################################
#############################################################
# Find peak
# TODO: are the uncertainties of a, b, c correct? They seem too large

def find_quadratic_coefficients_and_uncertainties(x, y, x_unc, y_unc):
    """
    This function gives coefficients of ax^2 +bx + c = y from 3 points, and their uncertainties. 
    """
    #Check that x and y have the same length, and that x has exactly 3 elements. If they do not, the program will raise an AssertionError.
    assert len(x) == len(y) 
    assert len(x) == 3 
    
    
    # Define general variables to make the code more clear
    ## Rename inputs
    x0, x1, x2 = x
    y0, y1, y2 = y
    dx0, dx1, dx2 = x_unc
    dy0, dy1, dy2 = y_unc
    ## Denominator of the coefficients formulas
    D = (x0 - x1) * (x0 - x2) * (x1 - x2)
    ## Numerator of the coefficients formulas
    Na = x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1) #for coefficient a
    Nb = x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2) #for coefficient b
    Nc = (x1 * x2 * y0 * (x1 - x2) + x0 * x2 * y1 * (x2 - x0) + x0 * x1 * y2 * (x0 - x1)) #for coefficient c
    ## Derivatives of the denominator D 
    dD_dx0 = (x1 - x2) * (2*x0 - x1 - x2)
    dD_dx1 = (x0 - x2) * (x0 - 2*x1 + x2)
    dD_dx2 = (x0 - x1) * (-x0 - x1 + 2*x2)
    
    
    # Calculating uncertainty of a
    ## Partial derivatives of a
    da_dx0 = ((y2 - y1) * D - Na * dD_dx0) / D**2
    da_dx1 = ((y0 - y2) * D - Na * dD_dx1) / D**2
    da_dx2 = ((y1 - y0) * D - Na * dD_dx2) / D**2
    da_dy0 = (x1 - x2) / D
    da_dy1 = (x2 - x0) / D
    da_dy2 = (x0 - x1) / D
    ## Uncertainty of coefficient a
    a_unc = np.sqrt(
            (da_dx0 * dx0)**2 +
            (da_dx1 * dx1)**2 +
            (da_dx2 * dx2)**2 +
            (da_dy0 * dy0)**2 +
            (da_dy1 * dy1)**2 +
            (da_dy2 * dy2)**2 )
    
    
    # Calculating uncertainty of b
    ## Partial derivatives of b
    db_dx0 = (2*x0*(y1 - y2)*D - Nb*dD_dx0) / D**2
    db_dx1 = (2*x1*(y2 - y0)*D - Nb*dD_dx1) / D**2
    db_dx2 = (2*x2*(y0 - y1)*D - Nb*dD_dx2) / D**2
    db_dy0 = (x2**2 - x1**2) / D
    db_dy1 = (x0**2 - x2**2) / D
    db_dy2 = (x1**2 - x0**2) / D
    ## Uncertainty of coefficient b
    b_unc = np.sqrt(
            (db_dx0 * dx0)**2 +
            (db_dx1 * dx1)**2 +
            (db_dx2 * dx2)**2 +
            (db_dy0 * dy0)**2 +
            (db_dy1 * dy1)**2 +
            (db_dy2 * dy2)**2 )
    
    
    # Calculating uncertainty of c
    ## Partial derivatives of c
    dc_dx0 = ((x2*y1*(x2 - 2*x0) + x1*y2*(2*x0 - x1))*D - Nc*dD_dx0) / D**2
    dc_dx1 = ((x2*y0*(2*x1 - x2) + x0*y2*(x0 - 2*x1))*D - Nc*dD_dx1) / D**2
    dc_dx2 = ((x1*y0*(x1 - 2*x2) + x0*y1*(2*x2 - x0))*D - Nc*dD_dx2) / D**2
    dc_dy0 = x1*x2*(x1 - x2) / D
    dc_dy1 = x2*x0*(x2 - x0) / D
    dc_dy2 = x0*x1*(x0 - x1) / D
    ## Uncertainty of coefficient c
    c_unc = np.sqrt(
            (dc_dx0 * dx0)**2 +
            (dc_dx1 * dx1)**2 +
            (dc_dx2 * dx2)**2 +
            (dc_dy0 * dy0)**2 +
            (dc_dy1 * dy1)**2 +
            (dc_dy2 * dy2)**2 )
    
    
    # Coefficients
    a = Na / D
    b = Nb / D
    c = Nc / D
    
    return [a, b, c, a_unc, b_unc, c_unc]

def quadratic_function_uncertainty(x, dx, a, b, c, da, db, dc):
    # Function value
    y = a * x**2 + b * x + c

    # Uncertainty propagation
    term_x = (2 * a * x + b) * dx
    term_a = x**2 * da
    term_b = x * db
    term_c = dc

    # Total uncertainty
    dy = np.sqrt(term_x**2 + term_a**2 + term_b**2 + term_c**2)
    
    return [y, dy]


def quadratic_vertex_and_uncertainty(a, b, c, a_unc, b_unc, c_unc):
    # x value
    vx = -b/(2*a)
    
    # x uncertainty
    term_vx1 = (b/(2*a**2)) * a_unc
    term_vx2 = (-1/(2*a)) * b_unc
    vx_unc = np.sqrt( term_vx1**2 + term_vx2**2 )
    
    # y value and uncertainty
    vy, vy_unc = quadratic_function_uncertainty(x=vx, dx=vx_unc, a=a, b=b, c=c, da=a_unc, db=b_unc, dc=c_unc)
    
    return [vx, vy, vx_unc, vy_unc]

def find_maximum_by_parabolic_interpolation(x_data, y_data, x_unc_data, y_unc_data, show_figure='yes', x_label='Doppler shift (km/s)', y_label=f'Radiance (W m^-2 sr^-1 nm^-1)'):
    # Calculate the maximum data point and the left and right closest data points
    idx_max = np.argmax(y_data) #Index of the maximum of the data
    x_nearmax = x_data[[idx_max-1, idx_max, idx_max+1]]
    y_nearmax = y_data[[idx_max-1, idx_max, idx_max+1]]
    x_unc_nearmax = x_unc_data[[idx_max-1, idx_max, idx_max+1]]
    y_unc_nearmax = y_unc_data[[idx_max-1, idx_max, idx_max+1]]
    
    # Calculate coefficients of the quadratic expression and their uncertainties
    a, b, c, a_unc, b_unc, c_unc = find_quadratic_coefficients_and_uncertainties(x=x_nearmax, y=y_nearmax, x_unc=x_unc_nearmax, y_unc=y_unc_nearmax)
    
    # Calculate the vertex of the parabola and the uncertainties
    x_vertex, y_vertex, x_unc_vertex, y_unc_vertex = quadratic_vertex_and_uncertainty(a=a, b=b, c=c, a_unc=a_unc, b_unc=b_unc, c_unc=c_unc)
        
    if show_figure=='yes':
        # Calculate curve of the parabola
        x_quad_curve = np.linspace(x_nearmax[0], x_nearmax[-1], 300)
        y_quad_curve, dy_quad_curve = quadratic_function_uncertainty(x=x_quad_curve, dx=0, a=a, b=b, c=c, da=a_unc, db=b_unc, dc=c_unc)
        
        fig, ax = plt.subplots(figsize=(12,6))
        #ax.errorbar(x=x_nearmax, y=y_nearmax, xerr=x_unc_nearmax, yerr=y_unc_nearmax, marker='.', color='blue', label='data')
        ax.errorbar(x=x_data, y=y_data, xerr=x_unc_data, yerr=y_unc_data, marker='.', color='blue', label='data')
        ax.errorbar(x=x_quad_curve, y=y_quad_curve, color='red', label='curve (interpolation)')
        ax.errorbar(x=x_quad_curve, y=y_quad_curve, yerr=dy_quad_curve, color='red', alpha=0.05, label='uncertainty of the curve (interpolation)')
        ax.errorbar(x=x_vertex, y=y_vertex, xerr=x_unc_vertex, yerr=y_unc_vertex, marker='^', color='green', label='vertex')
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.legend()
        plt.show(block=False)
    
    return {'x_vertex':x_vertex, 'y_vertex':y_vertex, 'x_unc_vertex':x_unc_vertex, 'y_unc_vertex':y_unc_vertex, 'a':a, 'b':b, 'c':c, 'a_unc':a_unc, 'b_unc':b_unc, 'c_unc':c_unc, 'idx_max':idx_max}

#############################################################
#############################################################
# 


# Find maximum
# TODO: are the uncertainties of a, b, c correct? They seem too large

def find_quadratic_coefficients_and_uncertainties(x, y, x_unc, y_unc):
    """
    This function gives coefficients of ax^2 +bx + c = y from 3 points, and their uncertainties. 
    """
    #Check that x and y have the same length, and that x has exactly 3 elements. If they do not, the program will raise an AssertionError.
    assert len(x) == len(y) 
    assert len(x) == 3 
    
    
    # Define general variables to make the code more clear
    ## Rename inputs
    x0, x1, x2 = x
    y0, y1, y2 = y
    dx0, dx1, dx2 = x_unc
    dy0, dy1, dy2 = y_unc
    ## Denominator of the coefficients formulas
    D = (x0 - x1) * (x0 - x2) * (x1 - x2)
    ## Numerator of the coefficients formulas
    Na = x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1) #for coefficient a
    Nb = x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2) #for coefficient b
    Nc = (x1 * x2 * y0 * (x1 - x2) + x0 * x2 * y1 * (x2 - x0) + x0 * x1 * y2 * (x0 - x1)) #for coefficient c
    ## Derivatives of the denominator D 
    dD_dx0 = (x1 - x2) * (2*x0 - x1 - x2)
    dD_dx1 = (x0 - x2) * (x0 - 2*x1 + x2)
    dD_dx2 = (x0 - x1) * (-x0 - x1 + 2*x2)
    
    
    # Calculating uncertainty of a
    ## Partial derivatives of a
    da_dx0 = ((y2 - y1) * D - Na * dD_dx0) / D**2
    da_dx1 = ((y0 - y2) * D - Na * dD_dx1) / D**2
    da_dx2 = ((y1 - y0) * D - Na * dD_dx2) / D**2
    da_dy0 = (x1 - x2) / D
    da_dy1 = (x2 - x0) / D
    da_dy2 = (x0 - x1) / D
    ## Uncertainty of coefficient a
    a_unc = np.sqrt(
            (da_dx0 * dx0)**2 +
            (da_dx1 * dx1)**2 +
            (da_dx2 * dx2)**2 +
            (da_dy0 * dy0)**2 +
            (da_dy1 * dy1)**2 +
            (da_dy2 * dy2)**2 )
    
    
    # Calculating uncertainty of b
    ## Partial derivatives of b
    db_dx0 = (2*x0*(y1 - y2)*D - Nb*dD_dx0) / D**2
    db_dx1 = (2*x1*(y2 - y0)*D - Nb*dD_dx1) / D**2
    db_dx2 = (2*x2*(y0 - y1)*D - Nb*dD_dx2) / D**2
    db_dy0 = (x2**2 - x1**2) / D
    db_dy1 = (x0**2 - x2**2) / D
    db_dy2 = (x1**2 - x0**2) / D
    ## Uncertainty of coefficient b
    b_unc = np.sqrt(
            (db_dx0 * dx0)**2 +
            (db_dx1 * dx1)**2 +
            (db_dx2 * dx2)**2 +
            (db_dy0 * dy0)**2 +
            (db_dy1 * dy1)**2 +
            (db_dy2 * dy2)**2 )
    
    
    # Calculating uncertainty of c
    ## Partial derivatives of c
    dc_dx0 = ((x2*y1*(x2 - 2*x0) + x1*y2*(2*x0 - x1))*D - Nc*dD_dx0) / D**2
    dc_dx1 = ((x2*y0*(2*x1 - x2) + x0*y2*(x0 - 2*x1))*D - Nc*dD_dx1) / D**2
    dc_dx2 = ((x1*y0*(x1 - 2*x2) + x0*y1*(2*x2 - x0))*D - Nc*dD_dx2) / D**2
    dc_dy0 = x1*x2*(x1 - x2) / D
    dc_dy1 = x2*x0*(x2 - x0) / D
    dc_dy2 = x0*x1*(x0 - x1) / D
    ## Uncertainty of coefficient c
    c_unc = np.sqrt(
            (dc_dx0 * dx0)**2 +
            (dc_dx1 * dx1)**2 +
            (dc_dx2 * dx2)**2 +
            (dc_dy0 * dy0)**2 +
            (dc_dy1 * dy1)**2 +
            (dc_dy2 * dy2)**2 )
    
    
    # Coefficients
    a = Na / D
    b = Nb / D
    c = Nc / D
    
    return [a, b, c, a_unc, b_unc, c_unc]

def quadratic_function_uncertainty(x, dx, a, b, c, da, db, dc):
    # Function value
    y = a * x**2 + b * x + c

    # Uncertainty propagation
    term_x = (2 * a * x + b) * dx
    term_a = x**2 * da
    term_b = x * db
    term_c = dc

    # Total uncertainty
    dy = np.sqrt(term_x**2 + term_a**2 + term_b**2 + term_c**2)
    
    return [y, dy]


def quadratic_vertex_and_uncertainty(a, b, c, a_unc, b_unc, c_unc):
    # x value
    vx = -b/(2*a)
    
    # x uncertainty
    term_vx1 = (b/(2*a**2)) * a_unc
    term_vx2 = (-1/(2*a)) * b_unc
    vx_unc = np.sqrt( term_vx1**2 + term_vx2**2 )
    
    # y value and uncertainty
    vy, vy_unc = quadratic_function_uncertainty(x=vx, dx=vx_unc, a=a, b=b, c=c, da=a_unc, db=b_unc, dc=c_unc)
    
    return [vx, vy, vx_unc, vy_unc]

def find_maximum_by_parabolic_interpolation(x_data, y_data, x_unc_data, y_unc_data, show_figure='yes'):
    # Calculate the maximum data point and the left and right closest data points
    idx_max = np.argmax(y_data) #Index of the maximum of the data
    x_nearmax = x_data[[idx_max-1, idx_max, idx_max+1]]
    y_nearmax = y_data[[idx_max-1, idx_max, idx_max+1]]
    x_unc_nearmax = x_unc_data[[idx_max-1, idx_max, idx_max+1]]
    y_unc_nearmax = y_unc_data[[idx_max-1, idx_max, idx_max+1]]
    
    # Calculate coefficients of the quadratic expression and their uncertainties
    a, b, c, a_unc, b_unc, c_unc = find_quadratic_coefficients_and_uncertainties(x=x_nearmax, y=y_nearmax, x_unc=x_unc_nearmax, y_unc=y_unc_nearmax)
    
    # Calculate the vertex of the parabola and the uncertainties
    x_vertex, y_vertex, x_unc_vertex, y_unc_vertex = quadratic_vertex_and_uncertainty(a=a, b=b, c=c, a_unc=a_unc, b_unc=b_unc, c_unc=c_unc)
        
    if show_figure=='yes':
        # Calculate curve of the parabola
        x_quad_curve = np.linspace(x_nearmax[0], x_nearmax[-1], 300)
        y_quad_curve, dy_quad_curve = quadratic_function_uncertainty(x=x_quad_curve, dx=0, a=a, b=b, c=c, da=a_unc, db=b_unc, dc=c_unc)
        
        fig, ax = plt.subplots(figsize=(12,6))
        #ax.errorbar(x=x_nearmax, y=y_nearmax, xerr=x_unc_nearmax, yerr=y_unc_nearmax, marker='.', color='blue', label='data')
        ax.errorbar(x=x_data, y=y_data, xerr=x_unc_data, yerr=y_unc_data, marker='.', color='blue', label='data')
        ax.errorbar(x=x_quad_curve, y=y_quad_curve, color='red', label='curve (interpolation)')
        ax.errorbar(x=x_quad_curve, y=y_quad_curve, yerr=dy_quad_curve, color='red', alpha=0.05, label='uncertainty of the curve (interpolation)')
        ax.errorbar(x=x_vertex, y=y_vertex, xerr=x_unc_vertex, yerr=y_unc_vertex, marker='^', color='green', label='vertex')
        ax.legend()
        plt.show(block=False)
    
    return {'x_vertex':x_vertex, 'y_vertex':y_vertex, 'x_unc_vertex':x_unc_vertex, 'y_unc_vertex':y_unc_vertex, 'a':a, 'b':b, 'c':c, 'a_unc':a_unc, 'b_unc':b_unc, 'c_unc':c_unc, 'idx_max':idx_max, 'x_nearmax': x_nearmax, 'y_nearmax': y_nearmax, 'x_unc_nearmax': x_unc_nearmax, 'y_unc_nearmax': y_unc_nearmax}

def find_maximum_by_parabolic_interpolation_adapted(wavelength, radiance, radiance_unc, show_figure='yes'):
    
    # Shift origin of the wavelength array closest to the peak of the line so that the calculation of the vertex is more accurate and with less uncertainty
    idx_max_rad = np.argmax(radiance) # index of the maximum value of radiance
    wavelength_radmax = wavelength[idx_max_rad] # maximum value of radiance
    wavelength_shifted = wavelength - wavelength_radmax # Shift the origin
    
    # Calculate the vertex of the parabola fitted
    fmbpi = find_maximum_by_parabolic_interpolation(x_data=wavelength_shifted, y_data=radiance, x_unc_data=np.zeros(len(wavelength)), y_unc_data=radiance_unc, show_figure='no')
    ## As the input was shifted, the x results are also shifted. We rename the variables:
    x_vertex_shifted = fmbpi['x_vertex']
    x_nearmax_shifted = fmbpi['x_nearmax']
    x_unc_nearmax_shifted = fmbpi['x_unc_nearmax']
    
    # Shift results to the original wavelength array "coordinates"
    x_vertex = x_vertex_shifted + wavelength_radmax
    x_nearmax = x_nearmax_shifted + wavelength_radmax
    x_unc_nearmax = x_unc_nearmax_shifted + wavelength_radmax
    
    # Show figure
    if show_figure=='yes':
        # Calculate curve of the parabola
        x_quad_curve_shifted = np.linspace(x_nearmax_shifted[0], x_nearmax_shifted[-1], 300)
        x_quad_curve = x_quad_curve_shifted + wavelength_radmax
        y_quad_curve, dy_quad_curve = quadratic_function_uncertainty(x=x_quad_curve_shifted, dx=0, a=fmbpi['a'], b=fmbpi['b'], c=fmbpi['c'], da=fmbpi['a_unc'], db=fmbpi['b_unc'], dc=fmbpi['c_unc'])
        
        fig, ax = plt.subplots(figsize=(12,6))
        ax.errorbar(x=wavelength, y=radiance, yerr=radiance_unc, marker='.', color='blue', label='data')
        #ax.scatter(x_nearmax, fmbpi['y_nearmax'], marker='s', s=50, color='orange', label='3 highest data points')
        ax.errorbar(x=x_quad_curve, y=y_quad_curve, color='red', label='curve (interpolation)')
        ax.errorbar(x=x_quad_curve, y=y_quad_curve, yerr=dy_quad_curve, color='red', alpha=0.05, label='uncertainty of the curve (interpolation)')
        ax.errorbar(x=x_vertex, y=fmbpi['y_vertex'], xerr=fmbpi['x_unc_vertex'], yerr=fmbpi['y_unc_vertex'], marker='^', color='green', label='vertex')
        ax.legend()
        plt.show(block=False)
    
    # Prepare result
    fmbpi['x_vertex'] = x_vertex
    fmbpi['x_nearmax'] = x_nearmax
    fmbpi['x_unc_nearmax'] = x_unc_nearmax
    
    return fmbpi


#############################################################
#############################################################
# 


def find_one_x_for_y(x_data, y_data, y_target, kind_interp='linear'):
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    
    interp_func = interp1d(x_data, y_data, kind=kind_interp, bounds_error=False, fill_value=np.nan)
    xs = []
    for i in range(len(x_data)-1):
        y1, y2 = interp_func(x_data[i]), interp_func(x_data[i+1])
        # Check if y_target is bracketed
        if (y1 - y_target) * (y2 - y_target) <= 0:
            try:
                root = brentq(lambda xx: interp_func(xx) - y_target, x_data[i], x_data[i+1])
                xs.append(root)
            except ValueError:
                pass
    return np.array(xs)

def find_x_for_y(x_data, y_data, y_target_list, kind_interp='linear'):
    xs_list = []
    for y_target in y_target_list:
        xs = find_one_x_for_y(x_data=x_data, y_data=y_data, y_target=y_target, kind_interp=kind_interp)
        xs_list.append(xs)
    return xs_list

def find_bisector(x_data, y_data, y_target_list, bisector_calculation='mean', kind_interp='linear'):
    x_found_list = find_x_for_y(x_data=x_data, y_data=y_data, y_target_list=y_target_list, kind_interp='linear')
    x_found_mean_list = []
    for x_start, x_end in x_found_list:
        if bisector_calculation=='mean': x_found_mean_list.append(np.mean([x_start, x_end]))
    return x_found_mean_list



def find_bisector(x_data, y_data, y_unc_data, y_target_list='auto', N_bisector_dots=50, kind_interp='linear', show_figure='yes'):
    
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    
    if y_target_list=='auto': y_target_list = np.linspace(np.min(y_data), np.max(y_data), N_bisector_dots)
    
    interp_func = interp1d(x_data, y_data, kind=kind_interp, bounds_error=False, fill_value=np.nan)
    
    x_found_mean_list, y_found_mean_list = [],[]
    for y_target in y_target_list:
        x_found = []
        for i in range(len(x_data)-1):
            y1, y2 = interp_func(x_data[i]), interp_func(x_data[i+1])
            # Check if y_target is bracketed
            if (y1 - y_target) * (y2 - y_target) <= 0:
                try:
                    root = brentq(lambda xx: interp_func(xx) - y_target, x_data[i], x_data[i+1])
                    x_found.append(root)
                except ValueError:
                    pass
        if len(x_found)==2: 
            x_found_mean_list.append(np.mean(x_found))
            y_found_mean_list.append(y_target)
    x_found_mean_list = np.array(x_found_mean_list)
    y_found_mean_list = np.array(y_found_mean_list)
    
    if show_figure=='yes':

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.errorbar(x=x_data, y=y_data, yerr=y_unc_data, color='blue', linewidth=1., elinewidth=1.0, marker='^', label='Data')
        ax.errorbar(x=x_found_mean_list, y=y_found_mean_list, color='red', marker='*', markersize=7, label='Bisector')
        ax.set_title(f'Bisector of the Ne VIII peak', fontsize=18) 
        ax.set_ylabel(r'Radiance of the peak (W/sr/m$^2$/''\u212B)', color='black', fontsize=16)
        ax.set_xlabel('Wavelength (\u212B)', color='black', fontsize=16)
        #ax.set_xscale('log')
        ax.legend()
        plt.show(block=False)


        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.errorbar(x=x_found_mean_list, y=y_found_mean_list, color='red', marker='*', markersize=7)
        ax.set_title(f'Bisector of the Ne VIII peak', fontsize=18) 
        ax.set_ylabel(r'Radiance of the peak (W/sr/m$^2$/''\u212B)', color='black', fontsize=16)
        ax.set_xlabel('Wavelength (\u212B)', color='black', fontsize=16)
        #ax.set_xscale('log')
        #ax.legend()
        plt.show(block=False)
    
    return [x_found_mean_list, y_found_mean_list]

#############################################################
#############################################################
# 



