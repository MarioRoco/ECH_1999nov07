import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datetime as dt
from astropy.io import fits
import matplotlib.patches as patches

sign_yaxis = -1 #1 or -1

#######################################################################
#######################################################################
# SOHO/SUMER. Slit magnification and displacement (Section 2.3.6.2 of Luca Teriaca's PhD thesis)

def magnification_factor_DetA(wavelength_Ang, order):
    """
    The projected size of the slit is wavelength dependent. The final size of the slit image on the detector plane is determined by the magnification factor (m_f). 
    """
    m = order #diffraction order
    l = wavelength_Ang #[Angstrom] 
    d = 2777.45 #[Angstrom] grating spacing
    ra = 3200.78 #[mm]
    fc = 399.60 #[mm]
    mld2 = 1-(m*l/d)**2 #[no units (Angstrom/Angstrom)]
    mld_sqrt = 1+np.sqrt(mld2) #[no units]
    m_f = ra/(fc*mld_sqrt) #[no units (mm/mm)]
    return m_f #[no units] magnification factor

def spatial_scale_DetA(wavelength_Ang, order):
    """
    The projected size of the slit is wavelength dependent. The final size of the slit image on the detector plane is determined by the magnification factor (m_f). With this parameter we can calculate the spatial scale (arcsec/pixel).
    """
    P_yA = 26.5 #[micrometers] spatial size of the detector pixels
    m_f = magnification_factor_DetA(order=order, wavelength_Ang=wavelength_Ang)
    ssA = P_yA/(6.316*m_f)
    return ssA #[arcsec/pixel] spatial scale of detector A    


# Function of the fit
def slit_center_DetA__preliminary(wavelength_Ang, degree=4):
    """
    There is a vertical displacement of the slit image as function of wavelength. 
    There is a software called POSITION.PRO written by W.Curdt that does this properly, but in pronciple we will do an interpolation to the values copied from figure 2.10 of Luca Teriaca's PhD thesis.
    """
    # Step 1: copy main values from figure 2.10 of Luca's thesis.
    x_wavelength = [800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250,  1300,  1350, 1400, 1450,  1500, 1550, 1600]
    y_center = [161, 163, 165, 167.5,  170,  172,  174, 175.8, 177, 177.8, 177.8, 177, 175.5, 172.5,  169,  164,  157] #[spatial detector (y) pixel]
    
    # Step 2: Fit a polynomial of a certain degree
    coeffs = np.polyfit(x_wavelength, y_center, degree)
    poly_fit = np.poly1d(coeffs)
    
    return poly_fit(wavelength_Ang) 


def get_slit_projection(w_px_range, w_cal_range_pixcenter, slit_length_arcsec, order_spectroheliogram=1):

    wavelength_spectroheliogram = np.mean(w_cal_range_pixcenter) #[Angstrom] wavelength of the line corresponding to the spectroheliogram (TODO: we use the theoretical value, but the correct way is to use the calibrated wavelength corresponding to the center of the interval selected)

    spatial_scale = spatial_scale_DetA(wavelength_Ang=wavelength_spectroheliogram, order=order_spectroheliogram) #[arcsec/pixel]
    slit_center = slit_center_DetA__preliminary(wavelength_Ang=wavelength_spectroheliogram, degree=4)
    slit_length_px = slit_length_arcsec / spatial_scale #[pixels] length of the slit
    
    slit_bottom_px_float = slit_center + slit_length_px/2
    slit_top_px_float = slit_center - slit_length_px/2

    #round and convert to integer
    slit_bottom_px  = int(np.round(slit_bottom_px_float))
    slit_top_px = int(np.round(slit_top_px_float))

    return [slit_bottom_px, slit_top_px]

def slit_projection_convert_pixel_to_helioprojective(slit_bottom_px, slit_top_px, header_sumer):
    slit_bottom_HPlat = s_image_pixels_to_helioprojective_latitude(s_px_img=slit_bottom_px, header_sumer=header_sumer)
    slit_top_HPlat = s_image_pixels_to_helioprojective_latitude(s_px_img=slit_top_px, header_sumer=header_sumer)
    return [slit_bottom_HPlat, slit_top_HPlat]

"""
## Example 1:

# Input
w_cal_range_suggest = [1547.5, 1549.0]
slit2_length_arcsec = 299.2 #[arcsec]
order_spectroheliogram = 1 #TODO: I think it's order 2
w_px_range, w_cal_range_pixcenter = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=w_cal_range_suggest, slope_cal=slope_cal, intercept_cal=intercept_cal)
wavelength_spectroheliogram = np.mean(w_cal_range_pixcenter) #[Angstrom] wavelength of the line corresponding to the spectroheliogram (TODO: we use the theoretical value, but the correct way is to use the calibrated wavelength corresponding to the center of the interval selected)

# Output
spatial_scale = spatial_scale_DetA(wavelength_Ang=wavelength_spectroheliogram, order=order_spectroheliogram) #[arcsec/pixel]
slit_center = slit_center_DetA__preliminary(wavelength_Ang=wavelength_spectroheliogram, degree=4)
slit2_length_px = slit2_length_arcsec / spatial_scale #[pixels] length of the slit

slit_bottom_px_float = slit_center + slit2_length_px/2
slit_top_px_float = slit_center - slit2_length_px/2

#round and convert to integer
slit_bottom_px, slit_top_px = int(np.round(slit_bottom_px_float)), int(np.round(slit_top_px_float))

print('Bottom pixel of the slit projection:', slit_bottom_px_float, '-->', slit_bottom_px)
print('Top pixel of the slit projection:   ', slit_top_px_float, '-->', slit_top_px)
"""

"""
## Example 2: show boundaries of the cropped (in HP latitude) monochromatic image
slit_bottom_HPlat, slit_top_HPlat = slit_projection_convert_pixel_to_helioprojective(slit_bottom_px=slit_bottom_px, slit_top_px=slit_top_px, header_sumer=header_sumer)
ax.axhline(y=slit_top_HPlat, color='red', label='top slit boundary')
ax.axhline(y=slit_bottom_HPlat, color='blue', label='bottom slit boundary')
"""


#######################################################################
#######################################################################
# SOHO/EIT. Conversion between pixels and helioprojective coordinates. This is preliminary, it could be valid a a rough approximation because we do not take into account the rotation respect to the solar rotation axis (which seems to be small). 

def X__pixel_to_HP(x_px, header_eit): 
    """
    X axis pixels to helioprojective longitude in arcseconds.
    """
    m = header_eit['CDELT1']
    x0_px = header_eit['CRPIX1']
    x0_HP = header_eit['CRVAL1']
    b = x0_HP - m * x0_px #intercept
    x_HP = m * x_px + b
    return x_HP #[arcsec]

def X__HP_to_pixel(x_HP, header_eit): 
    """
    Helioprojective longitude in arcseconds to X axis pixels.
    """
    m = header_eit['CDELT1']
    x0_px = header_eit['CRPIX1']
    x0_HP = header_eit['CRVAL1']
    b = x0_HP - m * x0_px #intercept
    x_px = (x_HP - b)/m
    return x_px #[pixel]


def Y__pixel_to_HP(y_px, header_eit): 
    """
    Y axis pixels to helioprojective latitude in arcseconds.
    """
    m = sign_yaxis * header_eit['CDELT2'] #negative sign is beacuse index (or pixel) 0 in Python starts in the top and the numbers increase downwards
    y0_px = header_eit['CRPIX2']
    y0_HP = header_eit['CRVAL2']
    b = y0_HP - m * y0_px #intercept
    y_HP = m * y_px + b
    return y_HP #[arcsec]

def Y__HP_to_pixel(y_HP, header_eit): 
    """
    Helioprojective latitude in arcseconds to Y axis pixels.
    """
    m = sign_yaxis * header_eit['CDELT1'] #negative sign is beacuse index (or pixel) 0 in Python starts in the top and the numbers increase downwards
    y0_px = header_eit['CRPIX1']
    y0_HP = header_eit['CRVAL1']
    b = y0_HP - m * y0_px #intercept
    y_px = (y_HP - b)/m
    return y_px #[pixel]


def helioprojective_extent_EIT(header_eit):
    """
    Function that creates the "extent" variable for "plt.imshow()" with the helioprojective coordinates. 
    """
    y_px_low = header_eit['NAXIS2']-1+0.5
    y_px_high = -0.5
    x_px_left = -0.5
    x_px_right = header_eit['NAXIS1']-1+0.5
    
    x_HP_left = X__pixel_to_HP(x_px=x_px_left, header_eit=header_eit)
    x_HP_right = X__pixel_to_HP(x_px=x_px_right, header_eit=header_eit)
    y_HP_low = Y__pixel_to_HP(y_px=y_px_low, header_eit=header_eit)
    y_HP_high = Y__pixel_to_HP(y_px=y_px_high, header_eit=header_eit)
    
    extent_eit_HP = [x_HP_left, x_HP_right, y_HP_low, y_HP_high]
    
    return extent_eit_HP #[arcsec, arcsec, arcsec, arcsec]

def pixel_extent_EIT(header_eit):
    """
    Function that creates the "extent" variable for "plt.imshow()" with the pixels in the order of EIT map (because Python inverts the order of the rows). 
    """
    y_px_low = -0.5
    y_px_high = header_eit['NAXIS2']-1+0.5
    x_px_left = -0.5
    x_px_right = header_eit['NAXIS1']-1+0.5
    
    extent_eit_px = [x_px_left, x_px_right, y_px_low, y_px_high]
    
    return extent_eit_px #[pixel, pixel, pixel, pixel]
    

def pixel_to_helioprojective_EIT(x_px, y_px, header_eit):
    """
    X and Y axes pixels to helioprojective longitude and latitude, respectively, in arcseconds.
    """
    # X axis (longitude)
    mx = header_eit['CDELT1']
    x0_px = header_eit['CRPIX1']
    x0_HP = header_eit['CRVAL1']
    bx = x0_HP - mx * x0_px #intercept
    x_HP = mx * x_px + bx

    # Y axis (latitude)
    my = sign_yaxis * header_eit['CDELT2'] #negative sign is beacuse index (or pixel) 0 in Python starts in the top and the numbers increase downwards
    y0_px = header_eit['CRPIX2']
    y0_HP = header_eit['CRVAL2']
    by = y0_HP - my * y0_px #intercept
    y_HP = my * y_px + by
    
    return [x_HP, y_HP] #[longitude, latitude] [arcsec, arcsec] helioprojective coordinates

################################################################
################################################################
# SOHO/SUMER. Conversion of detector pixels to other frames. 

def w_image_pixels_no_binned_to_detector_pixels(px_image, header_sumer): 
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
    # Variable names according to the above description
    Vim = px_image
    vdet_rs = header_sumer['DETXSTRT']
    vdet_re = header_sumer['DETXEND']
    vim_rs = header_sumer['NAXIS1']-1 #length of x axis (number of columns), -1 because indexation in Python starts in 0 and ends in length-1
    vim_re = 0
    
    # Calculation
    mm = (vdet_re-vdet_rs)/(vim_re-vim_rs)
    bb = vim_rs - mm * vdet_rs
    m = 1/mm
    b=-bb/m
    Vdet = m * Vim + b
    
    return Vdet #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(Vdet) in case s_px is one number, or np.floor(Vdet).astype(int) in case s_px is an array.


def w_detector_pixels_to_image_pixels_no_binned(px_detector, header_sumer): 
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
    # Variable names according to the above description
    Vdet = px_detector
    vdet_rs = header_sumer['DETXSTRT']
    vdet_re = header_sumer['DETXEND']
    vim_rs = header_sumer['NAXIS1']-1 #length of x axis (number of columns), -1 because indexation in Python starts in 0 and ends in length-1
    vim_re = 0
    
    # Calculation
    m = (vdet_re-vdet_rs)/(vim_re-vim_rs)
    b = vim_rs - m * vdet_rs
    Vim = m * Vdet  + b
    
    return Vim #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(Vim) in case s_px is one number, or np.floor(Vim).astype(int) in case s_px is an array.

# Create functions for main axis conversion ("extent")
def create_extent_x_detector_pixels(header_sumer):
    extent_x_left = w_image_pixels_no_binned_to_detector_pixels(px_image=0-0.5, header_sumer=header_sumer)
    extent_x_right = w_image_pixels_no_binned_to_detector_pixels(px_image=header_sumer['NAXIS1']-1+0.5, header_sumer=header_sumer)
    return [extent_x_left, extent_x_right]
    
"""
# Create functions for secondary x axis
def newaxis_x(px_image): return w_image_pixels_no_binned_to_detector_pixels(px_image=px_image, header_sumer=header_sumer)
def oldaxis_x(px_detector): return w_detector_pixels_to_image_pixels_no_binned(px_detector=px_detector, header_sumer=header_sumer)
"""

"""
## Example 1:
extent_x_left, extent_x_right = create_extent_x_detector_pixels(header_sumer=header_sumer)
extent_x_detector_pixels = [extent_x_left, extent_x_right, header_sumer['NAXIS2']-1+0.5, 0-0.5]

fig, ax = plt.subplots(figsize=(16, 6))
img = ax.imshow(data_sumer, extent=extent_detector_pixels, cmap='Greys', norm=LogNorm())
plt.show(block=False)
"""

"""
## Example 2: Show spectrum with secondary x axis
fig, ax = plt.subplots(figsize=(16, 6))
img = ax.imshow(data_sumer, cmap='Greys', norm=LogNorm())
secax_x = ax.secondary_xaxis('top', functions=(newaxis_x, oldaxis_x), color='green') # set secondary x axis
secax_x.set_xlabel('X detector pixels', color='green') # set label of secondary x axis
plt.show(block=False)
"""


def s_detector_pixels_to_image_pixels_no_binned(px_detector, header_sumer): 
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
    # Variable names according to the above description
    Vdet = px_detector
    vdet_rs = header_sumer['DETYSTRT']
    vdet_re = header_sumer['DETYEND']
    vim_rs = 0
    vim_re = header_sumer['NAXIS2']-1 #length of y axis (number of rows), -1 because indexation in Python starts in 0 and ends in length-1
    
    # Calculation
    m = sign_yaxis * (vdet_re-vdet_rs)/(vim_re-vim_rs)
    b = vim_rs - m * vdet_rs
    Vim = m * Vdet  + b
    
    return Vim #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(Vim) in case s_px is one number, or np.floor(Vim).astype(int) in case s_px is an array.


def s_image_pixels_no_binned_to_detector_pixels(px_image, header_sumer): 
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
    # Variable names according to the above description
    Vim = px_image
    vdet_rs = header_sumer['DETYSTRT']
    vdet_re = header_sumer['DETYEND']
    vim_rs = 0
    vim_re = header_sumer['NAXIS2']-1 #length of y axis (number of rows), -1 because indexation in Python starts in 0 and ends in length-1
    
    # Calculation
    mm = sign_yaxis * (vdet_re-vdet_rs)/(vim_re-vim_rs)
    bb = vim_rs - mm * vdet_rs
    m = 1/mm
    b=-bb/m
    Vdet = m * Vim + b
    
    return Vdet #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(Vdet) in case s_px is one number, or np.floor(Vdet).astype(int) in case s_px is an array.


# Create functions for main axis conversion ("extent")
def create_extent_y_detector_pixels(header_sumer):
    extent_y_bottom = s_image_pixels_no_binned_to_detector_pixels(px_image=0-0.5, header_sumer=header_sumer)
    extent_y_top = s_image_pixels_no_binned_to_detector_pixels(px_image=header_sumer['NAXIS2']-1+0.5, header_sumer=header_sumer)
    return [extent_y_bottom, extent_y_top]

"""
# Create functions for secondary y axis
def newaxis_y(px_image): return s_image_pixels_no_binned_to_detector_pixels(px_image=px_image, header_sumer=header_sumer)
def oldaxis_y(px_detector): return s_detector_pixels_to_image_pixels_no_binned(px_detector=px_detector, header_sumer=header_sumer)
"""

"""
## Example 1:
extent_y_bottom, extent_y_top = create_extent_y_detector_pixels(header_sumer=header_sumer)
extent_y_detector_pixels = [0-0.5, header_sumer['NAXIS1']-1+0.5, extent_y_bottom, extent_y_top]

fig, ax = plt.subplots(figsize=(16, 6))
img = ax.imshow(data_sumer, extent=extent_y_detector_pixels, cmap='Greys', norm=LogNorm())
plt.show(block=False)
"""

"""
## Example 2:
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(data_sumer, cmap='Greys', norm=LogNorm(), aspect='auto')
secax_y = ax.secondary_yaxis('right', functions=(newaxis_y, oldaxis_y), color='green') # set secondary y axis
secax_y.set_ylabel('Y detector pixels', color='green') # set label of secondary y axis
plt.show(block=False)
"""


def xy_detector_pixel_to_python_indices(x_det, y_det, header_sumer):
    # Convert from detector frame to image frame
    x_img = w_detector_pixels_to_image_pixels_no_binned(px_detector=x_det, header_sumer=header_sumer)
    y_img = s_detector_pixels_to_image_pixels_no_binned(px_detector=y_det, header_sumer=header_sumer)
    
    # Convert from imageimport numpy as np
    col_defective = int(round(x_img))
    row_defective = int(round(y_img)) #If we use the flipped array it is header_sumer['NAXIS2']-y_img, but if we use the original array it is y_img
    return[row_defective, col_defective]
    

################################################################
################################################################
# SOHO/SUMER. Defective pixels

def mask_defective_pixels(array_to_mask, header_sumer):
    """
    This function put masks in defective pixels. 
    """
    # 1) Defective pixels in the detector frame (list of [x,y] positions)
    if header_sumer['DETECTOR']=='A':
        defects_xy_pxdet = [[575, 255], [572, 254], [572, 253], [572, 252], [573, 255], [573, 254], [573, 253], [573, 252], [573, 251], [573, 250], [574, 255], [574, 254], [574, 253], [574, 252], [574, 251], [574, 250], [575, 256], [575, 255], [575, 254], [575, 253], [575, 252], [575, 251], [575, 250], [576, 256], [576, 255], [576, 254], [576, 253], [576, 252], [576, 251], [576, 250], [577, 256], [577, 255], [577, 254], [577, 253], [577, 252], [577, 251], [577, 250], [578, 255], [578, 254], [578, 253], [578, 252], [578, 251], [578, 250], [579, 254], [579, 253], [579, 252], [579, 251], [580, 253], [580, 252]] #all defective pixels (or we consider defective) in the detector A, around pixel (575,255). This is in the original coordinates system of the detector
    elif header_sumer['DETECTOR']=='B':
        defects_xy_pxdet = []
    
    # 2) Make a copy of the array of intensities
    array_to_mask_copy = np.copy(array_to_mask)
    
    # 3) #Assign a rare negative value to the defective pixels
    for x_det, y_det in defects_xy_pxdet:
        row_defective, col_defective = xy_detector_pixel_to_python_indices(x_det=x_det, y_det=y_det, header_sumer=header_sumer)
        array_to_mask_copy[row_defective, col_defective] = -999.9
    
    # 4) Mask the negative values
    array_masked = np.ma.masked_less(array_to_mask_copy, 0) #values less than zero are masked
    
    return array_masked
    
def mask_all_defective_pixels_DetA(array_to_mask):
    """
    This function put masks in all defective pixels detected.
    Use it when the spectrum array has been flipped in y-axis.
    """
    # 1) Make a copy of the array of intensities
    array_to_mask_copy = np.copy(array_to_mask)
    
    # 2) Defective pixels in the detector frame (list of [x,y] positions)
    defects_xy_px_0 = [[210, 255], [213, 254], [213, 253], [213, 252], [212, 255], [212, 254], [212, 253], [212, 252], [212, 251], [212, 250], [211, 255], [211, 254], [211, 253], [211, 252], [211, 251], [211, 250], [210, 256], [210, 255], [210, 254], [210, 253], [210, 252], [210, 251], [210, 250], [209, 256], [209, 255], [209, 254], [209, 253], [209, 252], [209, 251], [209, 250], [208, 256], [208, 255], [208, 254], [208, 253], [208, 252], [208, 251], [208, 250], [207, 255], [207, 254], [207, 253], [207, 252], [207, 251], [207, 250], [206, 254], [206, 253], [206, 252], [206, 251], [205, 253], [205, 252]]
    defects_xy_px_1 = [[442, 8], [443, 8], [444, 8], [445, 8], [446, 8], 
                    [442, 9], [443, 9], [444, 9], [445, 9], [446, 9], 
                    [442, 10], [443, 10], [444, 10], [445, 10], [446, 10], 
                    [442, 11], [443, 11], [444, 11], [445, 11], [446, 11], 
                    [442, 12], [443, 12], [444, 12], [445, 12], [446, 12]]
    defects_xy_px_2 = [[470, 19], [471, 19], [472, 19], [473, 19], [474, 19], 
                    [470, 20], [471, 20], [472, 20], [473, 20], [474, 20], 
                    [470, 21], [471, 21], [472, 21], [473, 21], [474, 21], 
                    [470, 22], [471, 22], [472, 22], [473, 22], [474, 22], 
                    [470, 23], [471, 23], [472, 23], [473, 23], [474, 23]]
    defects_xy_px_3 = [[471, 61], [472, 61], [473, 61], [474, 61], 
                    [471, 62], [472, 62], [473, 62], [474, 62], 
                    [471, 63], [472, 63], [473, 63], [474, 63], 
                    [471, 64], [472, 64], [473, 64], [474, 64]]
    defects_xy_px_4 = [[458, 92], [459, 92], [460, 92], [461, 92], [462, 92], [463, 92], 
                    [458, 93], [459, 93], [460, 93], [461, 93], [462, 93], [463, 93], 
                    [458, 94], [459, 94], [460, 94], [461, 94], [462, 94], [463, 94], 
                    [458, 95], [459, 95], [460, 95], [461, 95], [462, 95], [463, 95], 
                    [458, 96], [459, 96], [460, 96], [461, 96], [462, 96], [463, 96],
                    [458, 97], [459, 97], [460, 97], [461, 97], [462, 97], [463, 97]]
    defects_xy_px_5 = [[459, 88], [460, 88], [461, 88], [462, 88], [463, 88], 
                    [459, 89], [460, 89], [461, 89], [462, 89], [463, 89], 
                    [459, 90], [460, 90], [461, 90], [462, 90], [463, 90], 
                    [459, 91], [460, 91], [461, 91], [462, 91], [463, 91], 
                    [459, 92], [460, 92], [461, 92], [462, 92], [463, 92]]
    defects_xy_px_6 = [[488, 110], [489, 110], [490, 110], [491, 110], [492, 110], [493, 110], [494, 110], 
                    [488, 111], [489, 111], [490, 111], [491, 111], [492, 111], [493, 111], [494, 111], 
                    [488, 112], [489, 112], [490, 112], [491, 112], [492, 112], [493, 112], [494, 112], 
                    [488, 113], [489, 113], [490, 113], [491, 113], [492, 113], [493, 113], [494, 113], 
                    [488, 114], [489, 114], [490, 114], [491, 114], [492, 114], [493, 114], [494, 114],
                    [488, 115], [489, 115], [490, 115], [491, 115], [492, 115], [493, 115], [494, 115],
                    [488, 116], [489, 116], [490, 116], [491, 116], [492, 116], [493, 116], [494, 116]]
    defects_xy_px = np.concatenate([defects_xy_px_0, defects_xy_px_1, defects_xy_px_2, defects_xy_px_3, defects_xy_px_4, defects_xy_px_5, defects_xy_px_6])
    
    # 3) #Assign a rare negative value to the defective pixels
    for col_defective, row_defective in defects_xy_px:
        array_to_mask_copy[row_defective, col_defective] = -999.9
    
    # 4) Mask the negative values
    array_masked = np.ma.masked_less(array_to_mask_copy, 0) #values less than zero are masked
    
    return array_masked

"""
## Example:
data_sumer_masked = mask_other_defective_pixels(array_to_mask=data_sumer, header_sumer=header_sumer)
"""

#######################################################################
#######################################################################
# SOHO/EIT and SUMER. Functions for raster mode. 

def SUMERraster_get_data_and_header(sumer_filepath, sumer_filename_list):
    """
    Function to create a list with all the headers and another with all data arrays.
    """
    sumer_header_list, sumer_data_list = [],[]
    for filename_i in sumer_filename_list:
        # headers
        header_i = fits.getheader(sumer_filepath + filename_i)
        sumer_header_list.append(header_i)
        
        # data arrays
        data_i = fits.getdata(sumer_filepath + filename_i)
        data_i_masked = mask_defective_pixels(array_to_mask=data_i, header_sumer=header_i) # Mask defective pixels
        data_i_masked_flip = data_i_masked[::-1,:] #Note:we flip the row order of the spectrum array
        sumer_data_list.append(data_i_masked_flip)
    return [sumer_header_list, sumer_data_list]

def SUMERraster_get_data_header_and_datauncertainties(sumer_filepath, sumer_filename_list, factor_fullspectrum, t_exp_sec):
    """
    Function to create a list with all the headers, another with all data arrays, and another with the uncertainties of the data arrays.
    Uncertainties of the intensity for one pixel in one spectrum: DI_ij = sqrt(f_j*I_ij/t_exp), where:
        - f_j: factor intensity divided by rate (counts per second) for that wavelength. It is taken from SolarSoft (IDL). 
        - I_ij: Intensity of that pixel (i,j) od the data array.
        - t_exp: exposure time of the images (we assume all the images have the same exposure time.
    """
    sumer_header_list, sumer_data_list, sumer_data_unc_list = [],[],[]
    for filename_i in sumer_filename_list:
        # headers
        header_i = fits.getheader(sumer_filepath + filename_i)
        sumer_header_list.append(header_i)
        
        # data arrays
        data_i = fits.getdata(sumer_filepath + filename_i)[::-1,:]#Note:we reverse the row order of the spectrum array
        data_i_masked = mask_all_defective_pixels_DetA(array_to_mask=data_i) # Mask defective pixels
        sumer_data_list.append(data_i_masked)
        
        # Uncertainty data arrays
        data_unc_i_masked = np.sqrt(factor_fullspectrum * data_i_masked / t_exp_sec) #Uncertainties of the intensity for one pixel in one spectrum: DI_ij = sqrt(f_j*I_ij/t_exp)
        sumer_data_unc_list.append(data_unc_i_masked)
        
    return [sumer_header_list, sumer_data_list, sumer_data_unc_list]

def SUMERraster_average_spectra(sumer_data_list, sumer_data_unc_list):
    """
    Average of all the spectra of the raster.
    """
    raster_average = np.mean(sumer_data_list, axis=0)
    
    data_unc_sumsquare = np.zeros(raster_average.shape)
    for data_unc_i in sumer_data_unc_list:
        data_unc_sumsquare = data_unc_sumsquare + data_unc_i**2
    N = len(sumer_data_list)
    raster_average_unc = (1/N) * np.sqrt(data_unc_sumsquare)
    return [raster_average, raster_average_unc]


"""
## Example 1:
# Create a list with the names of the files as strings
sumer_filename_list = ['sum_19991107_00074270_15415_38.fits', 'sum_19991107_00101376_15415_38.fits', 'sum_19991107_00124466_15415_38.fits', etc.]

# Create a list with all the headers and another with all data arrays
sumer_header_list, sumer_data_list = SUMERraster_get_data_and_header(sumer_filepath=path_data_soho + 'sumer/', sumer_filename_list=sumer_filename_list)
"""


#sumer_header_list, sumer_data_list = SUMERraster_get_data_header(filepath=..., filename_list=[...])
def closest_time_EIT_SUMER(sumer_header_list, eit_header):
    """
    For raster mode. Find the closest time (date and hour) in the SUMER list to the time of EIT image, and its index.
    """
    ## Create list of times of the SUMER spectra
    times_sumer = []
    for header_i in sumer_header_list:
        time_sumer_str = header_i['DATE_OBS'] #start time of observation
        time_sumer_dt_ = dt.datetime.strptime(time_sumer_str, '%Y-%m-%dT%H:%M:%S.%f') #convert time in string format to datetime
        times_sumer.append(time_sumer_dt_)
    
    ## Extract time of the EIT image
    time_eit_str = eit_header['DATE-AVG'] #average time of observation
    time_eit = dt.datetime.strptime(time_eit_str, '%Y-%m-%dT%H:%M:%S.%f') #convert to datetime format

    ## Find the closest time and its index
    closest_index_sumer, closest_time_sumer = min(enumerate(times_sumer), key=lambda x: abs(x[1] - time_eit))
    """
    print("Time of EIT image:.......................", time_eit)
    print("Closest time of the SUMER raster:........", closest_time_sumer)
    print("Index of that SUMER file in the list:....", closest_index_sumer)
    """
    
    return [closest_index_sumer, closest_time_sumer, time_eit] 

def create_monochromatic_image(list_of_spectra, column_range, column_range_bckg='no'):
    w0, w1 = column_range
    if column_range_bckg!='no': w0b, w1b = column_range_bckg
    monochromatic_image_NT, monochromatic_image_bg_NT, monochromatic_image_nobg_NT = [],[],[]
    for spectrum_i in list_of_spectra:
        # Crop the spectrum array in the wavelength range (column range) selected
        ## Data to study
        spectrum_crop_i = spectrum_i[:, w0:w1+1] 
        ## Background to subtract
        if column_range_bckg!='no': bckg_i = spectrum_i[:, w0b:w1b+1]
        else: bckg_i = spectrum_crop_i #we put the same data to analyse to get zeros, so that if the forget to put the column range in "column_range_bckg" we are sure that we have not subtracted background
        
        # Average along the wavelength direction 
        y_profile = spectrum_crop_i.mean(axis=1) # Profile with background (background not subtracted)
        y_profile_bg = bckg_i.mean(axis=1) #Background
        y_profile_nobg = y_profile - y_profile_bg #Background subtracted
        
        # Create a 2D-array with the profiles of all the spectra to create the monochromatic image (or spectroheliogram)
        monochromatic_image_NT.append(y_profile)
        monochromatic_image_nobg_NT.append(y_profile_nobg)
        monochromatic_image_bg_NT.append(y_profile_bg)
        
    # Transpose
    monochromatic_image = np.array(monochromatic_image_NT).T # Profile with background (background not subtracted)
    monochromatic_image_nobg = np.array(monochromatic_image_nobg_NT).T #Background subtracted
    monochromatic_image_bg = np.array(monochromatic_image_bg_NT).T #Background
    
    return [monochromatic_image, monochromatic_image_nobg, monochromatic_image_bg]




#######################################################################
#######################################################################
# same as above functions SUMERraster_get_data_and_header(), SUMERraster_get_data_header_and_datauncertainties(), but for one spectrum (instead of a list of all the spectra of the raster). 

def SUMER_get_data_and_header(sumer_filepath, sumer_filename):
    """
    Function to get the header and data array of one spectrum. Also mask defective pixels and invert the order of the rows.
    """
    # header
    header_sumer_ = fits.getheader(sumer_filepath + sumer_filename)
    
    # data array
    data_sumer_ = fits.getdata(sumer_filepath + sumer_filename)
    data_sumer_masked_ = mask_defective_pixels(array_to_mask=data_sumer_, header_sumer=header_sumer_) # Mask defective pixels
    data_sumer_masked_flip_ = data_sumer_masked_[::-1,:] #Note:we flip the row order of the spectrum array
        
    return [header_sumer_, data_sumer_masked_flip_]

def SUMER_get_data_header_and_datauncertainties(sumer_filepath, sumer_filename, factor_fullspectrum, t_exp_sec):
    """
    Function to get the header, the data array, and the uncertainty array. Also mask defective pixels and invert the order of the rows.
    Uncertainties of the intensity for one pixel in one spectrum: DI_ij = sqrt(f_j*I_ij/t_exp), where:
        - f_j: factor intensity divided by rate (counts per second) for that wavelength. It is taken from SolarSoft (IDL). 
        - I_ij: Intensity of that pixel (i,j) od the data array.
        - t_exp: exposure time of the images (we assume all the images have the same exposure time.
    """
    # Get the header and data array of one spectrum. Also mask defective pixels and invert the order of the rows.
    header_sumer_, data_sumer_masked_flip_ = SUMER_get_data_and_header(sumer_filepath=sumer_filepath, sumer_filename=sumer_filename)
    
    # Uncertainty data arrays
    data_unc_sumer_masked_flip_ = np.sqrt(factor_fullspectrum * data_sumer_masked_flip_ / t_exp_sec) #Uncertainties of the intensity for one pixel in one spectrum: DI_ij = sqrt(f_j*I_ij/t_exp)
        
    return [header_sumer_, data_sumer_masked_flip_, data_unc_sumer_masked_flip_]


def radiance_line_1D(intensity_1darray_lineprofile, intensity_1darray_bckg, px_width_Ang):
    bckg_mean = np.mean(intensity_1darray_bckg) #[W/sr/m^2/Angstroem] Background level (mean)
    intensity_1darray_lineprofile_nobg = intensity_1darray_lineprofile - bckg_mean #[W/sr/m^2/Angstroem] 1D-array of intensities (of the line), background subtracted
    radiance_nobg = px_width_Ang * np.sum(intensity_1darray_lineprofile_nobg) #[W/sr/m^2] value of radiance (of the line), background subtracted
    return radiance_nobg #[W/sr/m^2] value of radiance (of the line), background subtracted

def radiance_line_2D(intensity_2darray_lineprofile, intensity_2darray_bckg, px_width_Ang):
    bckg_mean_1D = intensity_2darray_bckg.mean(axis=1) #[W/sr/m^2/Angstroem] Background level (mean)
    intensity_2darray_lineprofile_nobg = intensity_2darray_lineprofile - bckg_mean_1D[:, np.newaxis] #[W/sr/m^2/Angstroem] 2D-array of intensities (of the line), background subtracted
    radiance_1D_nobg = px_width_Ang * intensity_2darray_lineprofile_nobg.sum(axis=1) #[W/sr/m^2] value of radiance (of the line), background subtracted
    radiance_1D = px_width_Ang * intensity_2darray_lineprofile.sum(axis=1) #[W/sr/m^2] value of radiance (of the line), background subtracted
    return [radiance_1D_nobg, radiance_1D] #[W/sr/m^2]; [1D-array of radiance (of the line) with background subtracted, radiance_1D with background not subtracted] 


# same as function create_monochromatic_image(), but it subtracts the background and calculates the radiance (intensity*px_width_Ang)
def create_spectroheliogram(list_of_spectra, column_range, column_range_bckg, px_width_Ang):
    w0, w1 = column_range
    w0b, w1b = column_range_bckg
    spectroheliogram_NT, spectroheliogram_nobg_NT = [],[]
    for spectrum_i in list_of_spectra:
        # Crop the spectrum array in the wavelength range (column range) selected
        ## Data to study
        spectrum_crop_i = spectrum_i[:, w0:w1+1] 
        bckg_i = spectrum_i[:, w0b:w1b+1] # Background to subtract
        
        # Calculate the total radiance of the line for each spatial pixel (row of the spectrogram), background subtracted
        radiance_line_1D_nobg_i, radiance_line_1D_i = radiance_line_2D(intensity_2darray_lineprofile=spectrum_crop_i, intensity_2darray_bckg=bckg_i, px_width_Ang=px_width_Ang)
        
        # Create a 2D-array with the profiles of all the spectra to create the monochromatic image (or spectroheliogram)
        spectroheliogram_nobg_NT.append(radiance_line_1D_nobg_i) #background subtracted
        spectroheliogram_NT.append(radiance_line_1D_i) #background NOT subtracted
        
    # Transpose
    spectroheliogram = np.array(spectroheliogram_NT).T # Profile with background (background not subtracted)
    spectroheliogram_nobg = np.array(spectroheliogram_nobg_NT).T #Background subtracted
    
    return [spectroheliogram_nobg, spectroheliogram]


"""
## Example 1:
w_cal_range_suggest = [1547.5, 1549.0]
w_px_range, w_cal_range_pixcenter = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=w_cal_range_suggest, slope_cal=slope_cal, intercept_cal=intercept_cal)

w_cal_range_suggest_bckg = [1546.5, 1547.0]
w_px_range_bckg, w_cal_range_pixcenter_bckg = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=w_cal_range_suggest_bckg, slope_cal=slope_cal, intercept_cal=intercept_cal)

monochromatic_image, monochromatic_image_nobg, monochromatic_image_bg = create_monochromatic_image(list_of_spectra=datas_sumer, column_range=w_px_range, column_range_bckg=w_px_range_bckg)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
img = ax.imshow(monochromatic_image, cmap='hot', aspect='auto', norm=LogNorm(vmin=1e-1, vmax=1e1))
plt.show(block=False)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
img = ax.imshow(monochromatic_image_nobg, cmap='hot', aspect='auto', norm=LogNorm(vmin=1e-1, vmax=1e1))
plt.show(block=False)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
img = ax.imshow(monochromatic_image_bg, cmap='hot', aspect='auto', norm=LogNorm(vmin=1e-1, vmax=1e1))
plt.show(block=False)
"""


#######################################################################
#######################################################################
# SOHO/SUMER. Conversion between pixels and wavelength preliminary or calibrated.

def w_image_pixels_to_preliminary_wavelength(w_px_img, header_sumer): 
    """
    This function converts the pixel numbers of the x axis (wavelength direction) in the image to preliminary wavelength values (in Angstrom) according to the information of the fits header. 
    
    To perform this conversion we consider a straight line y=mx+b where:
    - y= preliminary wavelengths.
    - x= pixel number (or index, it's the same, because it starts at 0) of the wavelength dimension (x axis).
    - m= slope, which is given in the header with the keyword 'CDELT1'.
    - b= intercept, it is calculated with the reference values of x and y axis, given in the header with the keywords 'CRPIX1' and 'CRPIY1', respectively.
    """
    x = w_px_img
    m = header_sumer['CDELT1'] #slope # Axis increments along axis 1 (Angstrom)
    x_ref = header_sumer['CRPIX1'] # Reference pixel along axis 1 
    y_ref = header_sumer['CRVAL1'] # Value at reference pixel of axis 1 (Angstrom)   
    b = y_ref - m*x_ref
    w_preliminary = m*x+b
    return w_preliminary #[Angstrom]


def w_preliminary_wavelength_to_image_pixels(w_preliminary, header_sumer):
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
    x = w_preliminary
    mm = header_sumer['CDELT1'] #slope # Axis increments along axis 1 (Angstrom)
    x_ref = header_sumer['CRPIX1'] # Reference pixel along axis 1 
    y_ref = header_sumer['CRVAL1'] # Value at reference pixel of axis 1 (Angstrom)   
    bb = y_ref - mm*x_ref
    m = 1/mm
    b = -bb/mm
    w_px_img = m * x + b 
    return w_px_img #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(w_px) in case s_px is one number, or np.floor(w_px).astype(int) in case s_px is an array.


# Create functions for main axis conversion ("extent")
def create_extent_preliminary_wavelength(header_sumer):
    extent_x_left = w_image_pixels_to_preliminary_wavelength(w_px_img=0-0.5, header_sumer=header_sumer)
    extent_x_right = w_image_pixels_to_preliminary_wavelength(w_px_img=header_sumer['NAXIS1']-1+0.5, header_sumer=header_sumer)
    return [extent_x_left, extent_x_right]


"""
# Create functions for secondary y axis
def newaxis_x(w_px_img): return w_image_pixels_to_preliminary_wavelength(w_px_img=w_px_img, header_sumer=header_sumer)
def oldaxis_x(w_preliminary): return w_preliminary_wavelength_to_image_pixels(w_preliminary=w_preliminary, header_sumer=header_sumer)
"""

"""
## Example 1:
extent_x_left, extent_x_right = create_extent_preliminary_wavelength(header_sumer=header_sumer)
extent_preliminary_wavelength = [extent_x_left, extent_x_right, 0-0.5, header_sumer['NAXIS2']-1+0.5]

fig, ax = plt.subplots(figsize=(12, 6))
img = ax.imshow(data_sumer, extent=extent_preliminary_wavelength, cmap='Greys', norm=LogNorm(), aspect='auto')
plt.show(block=False)
"""

"""
## Example 2:
fig, ax = plt.subplots(figsize=(12, 8))
img = ax.imshow(data_sumer_flip, cmap='Greys', norm=LogNorm())
secax_x = ax.secondary_xaxis('top', functions=(newaxis_x, oldaxis_x), color='green') # set secondary y axis
secax_x.set_xlabel('Preliminary wavelength [Angstrom]', color='green') # set label of secondary y axis
plt.show(block=False)
"""


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



# Create functions for main axis conversion ("extent")
def create_extent_calibrated_wavelength_fullspectrum(header_sumer, slope_cal, intercept_cal):
    extent_x_left = w_image_pixels_to_calibrated_wavelength(w_px_img=0-0.5, slope_cal=slope_cal, intercept_cal=intercept_cal)
    extent_x_right = w_image_pixels_to_calibrated_wavelength(w_px_img=header_sumer['NAXIS1']-1+0.5, slope_cal=slope_cal, intercept_cal=intercept_cal)
    return [extent_x_left, extent_x_right]

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
"""
# Create functions for secondary y axis
def newaxis_x(w_px_img): return w_image_pixels_to_calibrated_wavelength(w_px_img=w_px_img, slope_cal=slope_cal, intercept_cal=intercept_cal)
def oldaxis_x(w_cal): return w_calibrated_wavelength_to_image_pixels(w_cal=w_cal, slope_cal=slope_cal, intercept_cal=intercept_cal)
"""

"""
## Example 1: get RANGE of pixels and wavelength centered in pixels
w_cal_range_suggest = [1547.5, 1549.0]
w_px_range, w_cal_range_pixcenter = w_range_pixel_from_wavelength_calibrated_fullspectrum(w_cal_range=w_cal_range_suggest, slope_cal=slope_cal, intercept_cal=intercept_cal)
"""

"""
## Example 2: entire x-axis (to plot PROFILES for example)
x_px_profile = np.arange(w_px_range[0], w_px_range[1]+1)
x_cal_profile = w_image_pixels_to_calibrated_wavelength(w_px_img=x_px_profile, slope_cal=slope_cal, intercept_cal=intercept_cal)
"""

"""
## Example 3: CROP x axis of 2D-array
spectrum2Darray_crop = spectrum2Darray[:, x_px_range[0]:x_px_range[1]+1] 
"""

"""
## Example 4: EXTENT
slope_cal = 0.04201 #(± 0.00022 Angstroms/pixel) for example
intercept_cal = 1530.450 #(± 0.066 Angstrom) for example
extent_x_left, extent_x_right = create_extent_calibrated_wavelength(header_sumer=header_sumer, slope_cal=slope_cal, intercept_cal=intercept_cal)
extent_calibrated_wavelength = [extent_x_left, extent_x_right, 0-0.5, header_sumer['NAXIS2']-1+0.5]
"""

"""
## Example 5: SECONDARY x axis
fig, ax = plt.subplots(figsize=(16, 6))
img = ax.imshow(data_sumer, cmap='Greys', norm=LogNorm(), aspect='auto')
secax_x = ax.secondary_xaxis('top', functions=(newaxis_x, oldaxis_x), color='green') # set secondary x axis
secax_x.set_xlabel('X detector pixels', color='green') # set label of secondary x axis
plt.show(block=False)
"""

#######################################################################
#######################################################################
# SOHO/SUMER. Change of coordinates between spatial dimension (Y axis of the spectrum) and (approximately) helioprojective latitude.

def s_image_pixels_to_helioprojective_latitude(s_px_img, header_sumer): 
    """
    This function converts the pixel numbers of the x axis (spatial direction) in the image to preliminary spatial direction values (in arcsec) according to the information of the fits header. 
    
    To perform this conversion we consider a straight line y=mx+b where:
    - y= preliminary spatial direction.
    - x= pixel number (or index, it's the same, because it starts at 0) of the spatial direction (y axis).
    - m= slope, which is given in the header with the keyword 'CDELT2'.
    - b= intercept, it is calculated with the reference values of x and y axis, given in the header with the keywords 'CRPIX2' and 'CRPIY2', respectively.
    """
    x = s_px_img
    m = sign_yaxis * header_sumer['CDELT2'] #slope # Axis increments along axis 2 (arcsec)
    x_ref = header_sumer['CRPIX2'] # Reference pixel along axis 2 
    y_ref = header_sumer['CRVAL2'] # Value at reference pixel of axis 2 (arcsec)   
    b = y_ref - m*x_ref
    s_HP = m*x+b
    return s_HP

def s_helioprojective_latitude_to_image_pixels(s_HP, header_sumer):
    """
    This function is the inverse of image_pixels_to_preliminary_space()
    
    To perform this conversion we consider a straight line y=mx+b where:
    - y= preliminary spatial direction.
    - x= pixel number (or indecreate_extent_helioprojective_latitude_fullspectrumx, it's the same, because it starts at 0) of the spatial direction (y axis).
    - mm= slope, which is given in the header with the keyword 'CDELT2'.
    - bb= intercept, it is calculated with the reference values of x and y axis, given in the header with the keywords 'CRPIX2' and 'CRPIY2', respectively.
    - m=1/mm = slope of the inverse of y=mx+b
    - b=-bb/mm = intercept of the inverse of y=mx+b
    """
    x = s_HP
    mm = sign_yaxis * header_sumer['CDELT2'] #slope # Axis increments along axis 2 (arcsec)
    x_ref = header_sumer['CRPIX2'] # Reference pixel along axis 2 
    y_ref = header_sumer['CRVAL2'] # Value at reference pixel of axis 2 (arcsec)   
    bb = y_ref - mm*x_ref
    m = 1/mm
    b = -bb/mm
    s_px_img = m * x + b 
    
    return s_px_img #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(s_px) in case s_px is one number, or np.floor(s_px).astype(int) in case s_px is an array.


# Create functions for main axis conversion ("extent")
def create_extent_helioprojective_latitude_fullspectrum(header_sumer):
    extent_y_bottom = s_image_pixels_to_helioprojective_latitude(s_px_img=header_sumer['NAXIS2']-1+0.5, header_sumer=header_sumer)
    extent_y_top = s_helioprojective_latitude_to_image_pixels(s_HP=0-0.5, header_sumer=header_sumer)
    return [extent_y_bottom, extent_y_top]

# Convert helioprojective latitude range to (closest) pixel range
def s_range_pixel_from_helioprojective_latitude(s_HPlat_range, header_sumer):
    # Calculate the pixel range (floats)
    s_px_range0_float = s_helioprojective_latitude_to_image_pixels(s_HP=s_HPlat_range[0], header_sumer=header_sumer)
    s_px_range1_float = s_helioprojective_latitude_to_image_pixels(s_HP=s_HPlat_range[1], header_sumer=header_sumer)
    
    # Round and convert to integer 
    s_px_range0 = int(np.round(s_px_range0_float))
    s_px_range1 = int(np.round(s_px_range1_float))
    s_px_range =  [s_px_range0,  s_px_range1]
    
    # Calculate the range of helioprojective latitude from the center of the pixels range
    s_HPlat_range0_pixcenter = s_image_pixels_to_helioprojective_latitude(s_px_img=float(s_px_range0), header_sumer=header_sumer)
    s_HPlat_range1_pixcenter = s_image_pixels_to_helioprojective_latitude(s_px_img=float(s_px_range1), header_sumer=header_sumer)
    s_HPlat_range_pixcenter = [s_HPlat_range0_pixcenter, s_HPlat_range1_pixcenter]
    
    return [s_px_range, s_HPlat_range_pixcenter]

"""
# Create functions for secondary y axis
def newaxis_y(s_px_img): return s_image_pixels_to_helioprojective(s_px_img=s_px_img, header_sumer=header_sumer)
def oldaxis_y(s_HP): return s_helioprojective_to_image_pixels(s_HP=s_HP, header_sumer=header_sumer)
"""

"""
## Example 1:
extent_y_bottom, extent_y_top = create_extent_helioprojective_latitude_fullspectrum(header_sumer=header_sumer)
extent_helioprojective_latitude = [0-0.5, header_sumer['NAXIS1']-1+0.5, extent_y_bottom, extent_y_top]

fig, ax = plt.subplots(figsize=(16, 6))
img = ax.imshow(data_sumer, extent=extent_helioprojective_latitude, cmap='Greys', norm=LogNorm())
plt.show(block=False)
"""

"""
## Example 2:
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(data_sumer, cmap='Greys', norm=LogNorm(), aspect='auto')
secax_y = ax.secondary_yaxis('right', functions=(newaxis_y, oldaxis_y), color='green') # set secondary y axis
secax_y.set_ylabel('Helioprojective latitude [arcsec]', color='green') # set label of secondary y axis
plt.show(block=False)
"""
"""
## Example 3:
HPlat_range_suggest_lineprofile = [-401.34, -150.65] 
s_px_range_lineprofile, s_HPlat_range_pixcenter_lineprofile = s_range_pixel_from_helioprojective_latitude(s_HPlat_range=HPlat_range_suggest_lineprofile, header_sumer=header_sumer)
"""


################################################################
# Improved versions

def SUMER_pixels_to_helioprojective_latitude_enhanced(pixel, header_sumer, wavelength_Ang, wavelength_order): 
    x = pixel
    m = sign_yaxis * spatial_scale_DetA(wavelength_Ang=wavelength_Ang, order=wavelength_order) #slope # Axis increments along axis 2 (arcsec)
    x_ref = header_sumer['CRPIX2'] # Reference pixel along axis 2 
    y_ref = header_sumer['CRVAL2'] # Value at reference pixel of axis 2 (arcsec)   
    b = y_ref - m*x_ref
    HP_lat = m*x+b
    return HP_lat

def SUMER_helioprojective_latitude_to_pixels_enhanced(HP_lat, header_sumer, wavelength_Ang, wavelength_order):
    x = HP_lat
    mm = sign_yaxis * spatial_scale_DetA(wavelength_Ang=wavelength_Ang, order=wavelength_order) #slope # Axis increments along axis 2 (arcsec)
    x_ref = header_sumer['CRPIX2'] # Reference pixel along axis 2 
    y_ref = header_sumer['CRVAL2'] # Value at reference pixel of axis 2 (arcsec)   
    bb = y_ref - mm*x_ref
    m = 1/mm
    b = -bb/mm
    pixel = m * x + b 
    return pixel #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(s_px) in case s_px is one number, or np.floor(s_px).astype(int) in case s_px is an array.


#######################################################################
#######################################################################
# SOHO/EIT and SUMER. Doppler shift due to solar rotation. 

# The difference of this function respect to the original is that here we exclude header_sumer (I think it is unnecessary)
def helioprojectiveSOHO_to_Stonyhurst_v2(x_HP, y_HP, header_eit): #(arcseconds, arcseconds, --, --) #Reference: SOHO's paper (Wilhelm 1995)
    # some parameters
    R_Sun_apparent = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius
    z_HP = np.sqrt(R_Sun_apparent**2 - (x_HP**2 + y_HP**2)) # [arcsec] zeta dimension (towards observer in helioprojective coordinates) in heliographic coordinates 
    B0_rad = (np.pi/180) * header_eit['SOLAR_B0'] #[rad] s/c tilt of solar North pole
    
    # Helioprojective to heliographic coordinates
    x_HG = x_HP #[arcsec] X dimension of heliographic coordinates
    y_HG = y_HP*np.cos(B0_rad) - z_HP*np.sin(B0_rad) #[arcsec] Y dimension of heliographic coordinates

    # Conversion to Heliographic Stonyhurst coordinates
    lat_HGS = np.arcsin(y_HG/R_Sun_apparent)
    lon_HGS = np.arcsin(x_HG/(R_Sun_apparent*np.cos(lat_HGS)))
    return [lon_HGS, lat_HGS] #[radians, radians]

def Stonyhurst_to_vLOS(lon_HGS, lat_HGS, header_eit): #(radians, radians, --, seconds)
    R_Sun_apparent = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius
    R_Sun = header_eit['RSUN_REF']/1000 #[km] assumed physical solar radius
    w_sidereal = 14.339 - 2.85*((np.sin(lat_HGS))**2 - (np.sin(15*np.pi/180))**2) #[deg/day] sidereal rotation frequency of the Sun for a certain latitude "lat_HGS" (due to the differential rotation, it depends on the latitude) Reference: SOHO's paper (Wilhelm 1995)
    w_Earth = 0.98561 #[deg/day] mean sidereal orbital velocity of the Earth (or SOHO because it is in L1) Reference: SOHO's paper (Wilhelm 1995)
    w_synodic = w_sidereal - w_Earth #[deg/day] synodic rotation frequency of a certain latitude "lat_HGS" in the Sun view from SOHO
    w_synodic_radsec = w_synodic * np.pi/180 /24/3600 #[rad/sec]
    
    R_Sun_lat = R_Sun * np.cos(lat_HGS) #distance from the South-North axis of the Sun to the surface at latitude lat_HGS
    v_synodic = R_Sun_lat * w_synodic_radsec #linear rotation velocity at a certain latitude of the Sun. As we want it for the redshift, we take into account the orbital speed of SOHO, so we use the synodic
    
    v_synodic_LOS = v_synodic * np.sin(lon_HGS) * np.cos(lat_HGS) #TODO: this is fine for SOHO but it is not accurate, so we can not use this for other spacecraft (I think)
    
    return v_synodic_LOS #[km/s]


def Doppler_shift_of_solar_rotation(x_HP, y_HP, header_eit, lamb_0):
    lon_HGS, lat_HGS = helioprojectiveSOHO_to_Stonyhurst(x_HP=x_HP, y_HP=y_HP, header_eit=header_eit)
    v_synodic_LOS = Stonyhurst_to_vLOS(lon_HGS=lon_HGS, lat_HGS=lat_HGS, header_eit=header_eit) #[km/s]
    c = 299792.4580 #[km/s] speed of light
    delta_lambda = lamb_0*(v_synodic_LOS/c) #Doppler velocity to wavelength shift (delta).
    return [v_synodic_LOS, delta_lambda] #[km/s, same units as lamb_0]

def Doppler_shift_of_solar_rotation_2Darray(HPlon_arr, HPlat_arr, header_eit, header_sumer, lamb_0, plot_speed='yes', plot_wlshift='yes'):
    Nrow = len(HPlat_arr)
    Ncol = len(HPlon_arr)
    delta_lambda_2darray = np.zeros([Nrow, Ncol])
    delta_v_2darray = np.zeros([Nrow, Ncol])
    for lon_idx, lon_ in enumerate(HPlon_arr):
        for lat_idx, lat_ in enumerate(HPlat_arr[::-1]):
            dv, dwl = Doppler_shift_of_solar_rotation(x_HP=lon_, y_HP=lat_, header_eit=header_eit, header_sumer=header_sumer, lamb_0=lamb_0)
            delta_lambda_2darray[lat_idx,lon_idx] = dwl
            delta_v_2darray[lat_idx,lon_idx] = dv
    
    if plot_speed=='yes' or plot_wlshift=='yes':
        # Compute aspect ratio of data range
        aspect_ratio = (HPlon_arr.max() - HPlon_arr.min()) / (HPlat_arr.max() - HPlat_arr.min())

        # Adjust figure size accordingly (keep width fixed, adjust height)
        fig_width = 12  # Keep this fixed
        fig_height = fig_width / aspect_ratio  # Adjust height based on data aspect
        fig_size=(fig_width, fig_height) # Use new figsize
    
    if plot_speed=='yes':
        # create symetric limits of the colors to ensure that 0 values are white, negatives in blue, and positives in red
        v_max_v = np.nanmax(np.abs(delta_v_2darray))  # Find max absolute value (different from NaNs) for symmetric limits
        v_min_v = -v_max_v  # Ensure zero is white
        print(v_min_v)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        img = ax.pcolormesh(HPlon_arr, HPlat_arr, delta_v_2darray, cmap='seismic', vmin=v_min_v, vmax=v_max_v)
        cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
        cbar = fig.colorbar(img, cax=cax)
        cbar.set_label(f'LOS rotation velocity [km/s]', fontsize=16)
        ax.set_title('LOS rotation velocity of the Sun', fontsize=18) 
        ax.set_xlabel('helioprojective longitude [arcsec]', color='black', fontsize=16)
        ax.set_ylabel('helioprojective latitude [arcsec]', color='black', fontsize=16)
        ax.set_aspect('equal')
        plt.show(block=False)
    
    if plot_wlshift=='yes':
        # create symetric limits of the colors to ensure that 0 values are white, negatives in blue, and positives in red
        v_max_lambda = np.nanmax(np.abs(delta_lambda_2darray)) # Find max absolute value (different from NaNs) for symmetric limits
        v_min_lambda = -v_max_lambda  # Ensure zero is white
        print(v_min_lambda)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        img = ax.pcolormesh(HPlon_arr, HPlat_arr, delta_lambda_2darray, cmap='seismic', vmin=v_min_lambda, vmax=v_max_lambda)
        cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
        cbar = fig.colorbar(img, cax=cax)
        cbar.set_label(f'Doppler shift [Angstrom]', fontsize=16)
        ax.set_title('Doppler shift in the Sun due to the rotation', fontsize=18) 
        ax.set_xlabel('helioprojective longitude [arcsec]', color='black', fontsize=16)
        ax.set_ylabel('helioprojective latitude [arcsec]', color='black', fontsize=16)
        ax.set_aspect('equal')
        plt.show(block=False)
    
    return [delta_v_2darray, delta_lambda_2darray]




#######################################################################
#######################################################################
# SOHO/EIT and SUMER. Solar rotation formulas. Reference: SOHO's paper (Wilhelm 1995)
def helioprojectiveSOHO_to_Stonyhurst(x_HP, y_HP, header_eit, header_sumer): #(arcseconds, arcseconds, --, --) #Reference: SOHO's paper (Wilhelm 1995)
    # some parameters
    R_Sun_apparent = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius
    z_HP = np.sqrt(R_Sun_apparent**2 - (x_HP**2 + y_HP**2)) # [arcsec] zeta dimension (towards observer in helioprojective coordinates) in heliographic coordinates 
    B0_rad = (np.pi/180) * header_sumer['SOLAR_B0'] #[rad] s/c tilt of solar North pole
    
    # Helioprojective to heliographic coordinates
    x_HG = x_HP #[arcsec] X dimension of heliographic coordinates
    y_HG = y_HP*np.cos(B0_rad) - z_HP*np.sin(B0_rad) #[arcsec] Y dimension of heliographic coordinates

    # Conversion to Heliographic Stonyhurst coordinates
    lat_HGS = np.arcsin(y_HG/R_Sun_apparent)
    lon_HGS = np.arcsin(x_HG/(R_Sun_apparent*np.cos(lat_HGS)))
    return [lon_HGS, lat_HGS] #[radians, radians]



def Stonyhurst_to_rotation_step(lon_HGS, lat_HGS, header_eit, time_interval): #(radians, radians, --, seconds)
    R_Sun_apparent = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius
    w_sidereal = 14.339 - 2.85*((np.sin(lat_HGS))**2 - (np.sin(15*np.pi/180))**2) #[deg/day] sidereal rotation frequency of the Sun (due to the differential rotation, it depends on the latitude (theta_rad))
    w_Earth = 0.98561 #[deg/day] mean sidereal orbital velocity of the Earth (or SOHO because it is in L1)
    w_synodic = w_sidereal - w_Earth #[deg/day] synodic rotation frequency of the Sun view from SOHO
    w_effective_DegPerDay = w_synodic * np.cos(lon_HGS) #[deg/day] effective angular rotation frequency
    w_effective = w_effective_DegPerDay *np.pi/180/86400 #[rad/sec] effective angular rotation frequency (in radians per second)

    step_HP = w_effective * R_Sun_apparent * np.cos(lat_HGS) * time_interval #[arcsec] 
    return step_HP #[arcsec] rotation step from East to West in helioprojective coordinates

def rotation_step_SOHO(x_HP, y_HP, header_eit, header_sumer, time_interval):
    lon_HGS, lat_HGS = helioprojectiveSOHO_to_Stonyhurst(x_HP=x_HP, y_HP=y_HP, header_eit=header_eit, header_sumer=header_sumer)
    step_HP = Stonyhurst_to_rotation_step(lon_HGS=lon_HGS, lat_HGS=lat_HGS, header_eit=header_eit, time_interval=time_interval)
    return step_HP #[arcsec]

def SUMERraster_substract_solar_rotation_HPlon_east(sumer_header_list, header_eit, closest_index):
    """
    Calculate the longitudes and latitudes of the spectroheliogram compensating the solar rotation, at the east (left) of the slit location when the EIT image is taken.
    Note: Here we consider that the slit is moving towards West (right) faster than the solar rotation.
    """
    I = closest_index
    #print('i:', I)
    header1 = sumer_header_list[I]
    lon1_HP = header1['CRVAL3']
    lat1_HP = header1['CRVAL2']
    longitudes_east_HP = [lon1_HP]
    latitudes_east_HP = [lat1_HP]
    for i in np.arange(1,I+1)[::-1]:
        #print('i:', i)
        # Call the header of the current spectrum (header1) and the following (header2)
        header1 = sumer_header_list[i]
        header2 = sumer_header_list[i-1]

        # Define the time step between the current spectrum and the following one
        date_start1 = dt.datetime.strptime(header1['DATE_OBS'], '%Y-%m-%dT%H:%M:%S.%f') #convert date in string format to datetime. We use the time when the current (1) observation starts
        date_start2 = dt.datetime.strptime(header2['DATE_OBS'], '%Y-%m-%dT%H:%M:%S.%f') #convert date in string format to datetime. We use the time when the following (2) observation starts
        Delta_t_sec = abs((date_start2-date_start1).total_seconds())

        # Calculate the shift in helioprojective latitude (arcseconds) of the solar surface due to its rotation during the time step between the current spectrum and the following one
        #shift_arcsec_rotation = apparent_displacement(latitude_deg=lon1_HGS, longitude_deg=lat1_HGS, R_Sun_km=695699.968, d_km=header_eit['DSUN_OBS']/1000, Delta_t_sec=Delta_t_sec)
        shift_arcsec_rotation = rotation_step_SOHO(x_HP=lon1_HP, y_HP=lat1_HP, header_eit=header_eit, header_sumer=header1, time_interval=Delta_t_sec)
        shift_arcsec_rotation_abs = abs(shift_arcsec_rotation)

        # Calculate the step of the slit in HP coordinates 
        shift_arcsec_slit = header2['CRVAL3'] - header1['CRVAL3'] #As the difference between the values at the reference pixels of axis 3 (reference pixels correspon to the center of the slit)
        shift_arcsec_slit_abs = abs(shift_arcsec_slit)

        # Calculate the step subtracting the solar rotation
        effective_shift_acsec = shift_arcsec_slit_abs - shift_arcsec_rotation_abs

        # Calculate the new coordinate (HP longitude)
        lon1_HP = lon1_HP - effective_shift_acsec #We will use it for the next iteration, so it is called "lon1_HP"
        lat1_HP = header2['CRVAL2'] #latitude of the following spectrum (2, not 1). We will use it for the next iteration, so it is called "lat1_HP"
        #print('lon1_HP east:', lon1_HP)
        
        # Save results in lists
        longitudes_east_HP.append(lon1_HP)
        latitudes_east_HP.append(lat1_HP)

    return [longitudes_east_HP, latitudes_east_HP]

def SUMERraster_substract_solar_rotation_HPlon_west(sumer_header_list, header_eit, closest_index):
    """
    Calculate the longitudes and latitudes of the spectroheliogram compensating the solar rotation, at the west (right) of the slit location when the EIT image is taken.
    Note: Here we consider that the slit is moving towards West (right) faster than the solar rotation.
    """
    I = closest_index
    header1 = sumer_header_list[I]
    lon1_HP = header1['CRVAL3']
    lat1_HP = header1['CRVAL2']
    longitudes_west_HP = [] #not include closest_index because it is already included in "east" function
    latitudes_west_HP = []
    NN = len(sumer_header_list)
    for i in np.arange(I,NN-1):
        #print('i:', i)
        # Call the header of the current spectrum (header1) and the following (header2)
        header1 = sumer_header_list[i]
        header2 = sumer_header_list[i+1]

        # Define the time step between the current spectrum and the following one
        date_start1 = dt.datetime.strptime(header1['DATE_OBS'], '%Y-%m-%dT%H:%M:%S.%f') #convert date in string format to datetime. We use the time when the current (1) observation starts
        date_start2 = dt.datetime.strptime(header2['DATE_OBS'], '%Y-%m-%dT%H:%M:%S.%f') #convert date in string format to datetime. We use the time when the following (2) observation starts
        Delta_t_sec = abs((date_start2-date_start1).total_seconds())

        # Calculate the shift in helioprojective latitude (arcseconds) of the solar surface due to its rotation during the time step between the current spectrum and the following one
        #shift_arcsec_rotation = apparent_displacement(latitude_deg=lon1_HGS, longitude_deg=lat1_HGS, R_Sun_km=695699.968, d_km=header_eit['DSUN_OBS']/1000, Delta_t_sec=Delta_t_sec)
        shift_arcsec_rotation = rotation_step_SOHO(x_HP=lon1_HP, y_HP=lat1_HP, header_eit=header_eit, header_sumer=header1, time_interval=Delta_t_sec)
        shift_arcsec_rotation_abs = abs(shift_arcsec_rotation)

        # Calculate the step of the slit in HP coordinates 
        shift_arcsec_slit = header2['CRVAL3'] - header1['CRVAL3'] #As the difference between the values at the reference pixels of axis 3 (reference pixels correspon to the center of the slit)
        shift_arcsec_slit_abs = abs(shift_arcsec_slit)

        # Calculate the step subtracting the solar rotation
        effective_shift_acsec = shift_arcsec_slit_abs - shift_arcsec_rotation_abs

        # Calculate the new coordinate (HP longitude)
        lon1_HP = lon1_HP + effective_shift_acsec #We will use it for the next iteration, so it is called "lon1_HP"
        lat1_HP = header2['CRVAL2'] #latitude of the following spectrum (2, not 1). We will use it for the next iteration, so it is called "lat1_HP"
        #print('lon1_HP west:', lon1_HP)
        
        # Save results in lists
        longitudes_west_HP.append(lon1_HP)
        latitudes_west_HP.append(lat1_HP)

    return [longitudes_west_HP, latitudes_west_HP]


def SUMERraster_substract_solar_rotation_HPlon(sumer_header_list, header_eit, closest_index):
    #Calculate the longitudes and latitudes of the spectroheliogram compensating the solar rotation, at the east and west of the slit location when the EIT image is taken
    longitudes_east_HP_reversed, latitudes_east_HP_reversed = SUMERraster_substract_solar_rotation_HPlon_east(sumer_header_list=sumer_header_list, header_eit=header_eit, closest_index=closest_index)
    longitudes_west_HP, latitudes_west_HP = SUMERraster_substract_solar_rotation_HPlon_west(sumer_header_list=sumer_header_list, header_eit=header_eit, closest_index=closest_index)
    
    # Reverse the order of longitudes and latitudes arrays so that they go from east to west and, thus, we can concatenate with the "west" arrays
    longitudes_east_HP = longitudes_east_HP_reversed[::-1]
    latitudes_east_HP = latitudes_east_HP_reversed[::-1]
    
    # Concatenate the longitudes and latitudes before (east) and after (west) the EIT image
    HPlon_raster_rotcomp = np.concatenate([longitudes_east_HP, longitudes_west_HP]) #latitudes, solar rotation compensated
    HPlat_raster_feference_pixel = np.concatenate([latitudes_east_HP, latitudes_west_HP])
    
    return [HPlon_raster_rotcomp, HPlat_raster_feference_pixel]

"""
## Example 1:
closest_index, closest_time_sumer, time_eit = closest_time_EIT_SUMER(sumer_header_list, eit_header)
HPlon_raster_rotcomp, HPlat_raster_feference_pixel = SUMERraster_substract_solar_rotation_HPlon(sumer_header_list=sumer_header_list, header_eit=header_eit, closest_index=closest_index)
"""
"""
## Example 2:
xx = HPlon_raster_rotcomp
yy = s_image_pixels_to_helioprojective_latitude(s_px_img=np.arange(slit_top_px,slit_bottom_px), header_sumer=header_sumer)
zz = monochromatic_image_nobg_HPlatcrop
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
img = ax.pcolormesh(xx,yy,zz, cmap='Greys_r', norm=LogNorm(vmin=1e-1, vmax=1e1))
cbar = fig.colorbar(img, ax=ax, pad=0.03)
ax.set_title(f'SOHO/SUMER spectroheliogram, range: {round(w_cal_range_pixcenter[0],2)} - {round(w_cal_range_pixcenter[1],2)} Angstrom')
ax.set_xlabel('Helioprojective longitude (arcsec)')
ax.set_ylabel('Helioprojective latitude (arcsec)')
plt.show(block=False)
"""


#######################################################################
#######################################################################
# 

def create_profile(spectrum_array, spectrum_array_unc, row_range, col_range):
    # Create profile full wavelength range of the spectrum
    N_cols = spectrum_array.shape[1]
    N_rows = row_range[1]-row_range[0]+1
    x_profile_fullspectrum = np.arange(0,N_cols)
    y_profile_fullspectrum = spectrum_array[row_range[0]:row_range[1]+1, :].mean(axis=0)
    x_unc_profile_fullspectrum = 0.5 * np.ones(N_cols)
    y_unc_square = (spectrum_array_unc[row_range[0]:row_range[1]+1, :])**2
    y_unc_profile_fullspectrum = 1/N_rows * np.sqrt(y_unc_square.sum(axis=0))
    
    # Create profile selected wavelength sub-range of the spectrum
    x_profile = x_profile_fullspectrum[col_range[0]:col_range[1]+1]
    y_profile = y_profile_fullspectrum[col_range[0]:col_range[1]+1]
    x_unc_profile = x_unc_profile_fullspectrum[col_range[0]:col_range[1]+1]
    y_unc_profile = y_unc_profile_fullspectrum[col_range[0]:col_range[1]+1]
    
    return [x_profile_fullspectrum, x_unc_profile_fullspectrum, y_profile_fullspectrum, y_unc_profile_fullspectrum, x_profile, x_unc_profile, y_profile, y_unc_profile]


def plot_full_spectrum_and_profile(spectrum_array, spectrum_array_unc, row_range, col_range):
    # Calculate profile
    x_profile_fullspectrum, x_unc_profile_fullspectrum, y_profile_fullspectrum, y_unc_profile_fullspectrum, x_profile, x_unc_profile, y_profile, y_unc_profile = create_profile(spectrum_array=spectrum_array, spectrum_array_unc=spectrum_array_unc, row_range=row_range, col_range=col_range)
    
    # Show averaged spectra
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2,1.2]})
    ## 2D-spectrum
    img = ax[0].imshow(spectrum_array, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.9, 0.39, 0.03, 0.56])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax[0].set_title(f'SOHO/SUMER, raster spectra averaged', fontsize=18) 
    ax[0].set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax[0].set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)
    ax[0].axvspan(col_range[0], col_range[1], color='blue', alpha=0.15, label=f'Columns range {col_range[0]}-{col_range[1]}') 
    ax[0].axhline(row_range[0], color='blue', linestyle='--', linewidth=0.8, label=f'Rows range {row_range[0]}-{row_range[1]}') 
    ax[0].axhline(row_range[1], color='blue', linestyle='--', linewidth=0.8) 
    ax[0].legend()
    ## Profile
    ax[1].errorbar(x=x_profile_fullspectrum, y=y_profile_fullspectrum, xerr=x_unc_profile_fullspectrum, yerr=y_unc_profile_fullspectrum, color='black', linewidth=0.8)
    #ax[1].set_title(f'SOHO/SUMER spectrum, raster spectra averaged, rows {row_range[0]}-{row_range[1]}', fontsize=18) 
    ax[1].set_xlabel('Wavelength dimension (pixels)', color='black', fontsize=16)
    ax[1].set_ylabel(f'Intensity [W/sr/m^2/Angstroem]', color='black', fontsize=16)
    ax[1].set_yscale('log')
    ax[1].set_xlim([x_profile_fullspectrum[0], x_profile_fullspectrum[-1]])
    ax[1].axvspan(col_range[0], col_range[1], color='blue', alpha=0.15) 
    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)
    plt.show(block=False)
    return [x_profile_fullspectrum, x_unc_profile_fullspectrum, y_profile_fullspectrum, y_unc_profile_fullspectrum, x_profile, x_unc_profile, y_profile, y_unc_profile]


def plot_subspectrum_profile(spectrum_array, spectrum_array_unc, row_range, col_range):
    # Calculate profile
    x_profile_fullspectrum, x_unc_profile_fullspectrum, y_profile_fullspectrum, y_unc_profile_fullspectrum, x_profile, x_unc_profile, y_profile, y_unc_profile = create_profile(spectrum_array=spectrum_array, spectrum_array_unc=spectrum_array_unc, row_range=row_range, col_range=col_range)
    
    # Show entire spectrum profile of one row
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.errorbar(x=x_profile, y=y_profile, xerr=x_unc_profile, yerr=y_unc_profile, color='blue', linewidth=0.8)
    ax.set_title(f'SOHO/SUMER spectrum, raster spectra averaged, rows {row_range[0]}-{row_range[1]}', fontsize=18) 
    ax.set_xlabel('Wavelength dimension (pixels)', color='black', fontsize=16)
    ax.set_ylabel(f'Intensity [W/sr/m^2/Angstroem]', color='black', fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)
    ax.set_xlim(col_range)
    ax.grid()
    plt.show(block=False)
    return [x_profile_fullspectrum, x_unc_profile_fullspectrum, y_profile_fullspectrum, y_unc_profile_fullspectrum, x_profile, x_unc_profile, y_profile, y_unc_profile]


def create_profile_1row(spectrum_array, spectrum_array_unc, row, col_range):
    # Create profile full wavelength range of the spectrum
    N_cols = spectrum_array.shape[1]
    x_profile_fullspectrum = np.arange(0,N_cols)
    y_profile_fullspectrum = spectrum_array[row, :]
    x_unc_profile_fullspectrum = 0.5 * np.ones(N_cols)
    y_unc_profile_fullspectrum = spectrum_array_unc[row, :]
    
    # Create profile selected wavelength sub-range of the spectrum
    x_profile = x_profile_fullspectrum[col_range[0]:col_range[1]+1]
    y_profile = y_profile_fullspectrum[col_range[0]:col_range[1]+1]
    x_unc_profile = x_unc_profile_fullspectrum[col_range[0]:col_range[1]+1]
    y_unc_profile = y_unc_profile_fullspectrum[col_range[0]:col_range[1]+1]
    
    return [x_profile_fullspectrum, x_unc_profile_fullspectrum, y_profile_fullspectrum, y_unc_profile_fullspectrum, x_profile, x_unc_profile, y_profile, y_unc_profile]


def plot_compare_subspectrum_profiles(spectrum_array, spectrum_array_unc, row_list, col_range):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    color_list = ['blue', 'red', 'green', 'orange', 'cyan', 'magenta', 'darkblue', 'purple']
    for r, row in enumerate(row_list):
        x_profile_fullspectrum, x_unc_profile_fullspectrum, y_profile_fullspectrum, y_unc_profile_fullspectrum, x_profile, x_unc_profile, y_profile, y_unc_profile = create_profile_1row(spectrum_array=spectrum_array, spectrum_array_unc=spectrum_array_unc, row=row, col_range=col_range)
        ax.errorbar(x=x_profile, y=y_profile, xerr=x_unc_profile, yerr=y_unc_profile, color=color_list[r], linewidth=0.8, label=f'{row}')
    ax.set_title(f'SOHO/SUMER spectrum, raster spectra averaged, comparison of profiles', fontsize=18) 
    ax.set_xlabel('Wavelength dimension (pixels)', color='black', fontsize=16)
    ax.set_ylabel(f'Intensity [W/sr/m^2/Angstroem]', color='black', fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)
    ax.set_xlim(col_range)
    ax.grid()
    ax.legend(title='Row index')
    plt.show(block=False)
    
def plot_locate_compare_subspectrum_profiles(spectrum_array, spectrum_array_unc, row_list, col_range):
    ## Locate the profiles in the spectrogram
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(raster_average, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.9, 0.39, 0.03, 0.56])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title(f'SOHO/SUMER, raster spectra averaged', fontsize=18) 
    ax.set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)
    ax.axvspan(col_range[0], col_range[1], color='blue', alpha=0.15, label=f'Columns range {col_range[0]}-{col_range[1]}') 
    
    color_list = ['blue', 'red', 'green', 'orange', 'cyan', 'magenta', 'darkblue', 'purple']
    for r, row in enumerate(row_list):
        ax.axhline(y=row, color=color_list[r], linestyle='--', linewidth=1.5, label=f'Row {row}')
    ax.legend()
    plt.show(block=False)
    
    
# Define function that executes all the analysis
def fit_lines_spectrum(spectrum_array, spectrum_array_unc, row_range, col_range, init_parameters, y_lims='no'):
    
    ## Create full spectrum profile and sub-range profile 
    x_profile_fullspectrum, x_unc_profile_fullspectrum, y_profile_fullspectrum, y_unc_profile_fullspectrum, x_profile, x_unc_profile, y_profile, y_unc_profile = create_profile(spectrum_array=spectrum_array, spectrum_array_unc=spectrum_array_unc, row_range=row_range, col_range=col_range)
    
    ## Execute the fitting
    multigauss_fit_results = fit_multi_gaussian_ODR(x_data=x_profile, y_data=y_profile, x_unc_data=x_unc_profile, y_unc_data=y_unc_profile, init_parameters=init_parameters)
    ## Plot the fit
    plot_fit_multi_gaussian_ODR(multigauss_fit_results, y_lims=y_lims)
    
    return multigauss_fit_results

def plot_spectrogram_and_rows_ranges(spectrum_array, row_range_list):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    img = ax.imshow(spectrum_array, cmap='Greys', aspect='auto', norm=LogNorm())
    cax = fig.add_axes([0.91, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(f'Av. intensity [W/sr/m^2/Angstroem]', fontsize=16)
    ax.set_title(f'SOHO/SUMER, raster spectra averaged', fontsize=18) 
    ax.set_xlabel('Wavelength direction (pixels)', color='black', fontsize=16)
    ax.set_ylabel('Spatial direction (pixels)', color='black', fontsize=16)
    color_rangelist = plt.get_cmap('tab20').colors + plt.get_cmap('tab10').colors[:9] # Generate 29 colors by combining 'tab20' and 'tab10' colormaps
    #color_rangelist = ['blue', 'red', 'green', 'orange', 'cyan', 'magenta', 'darkblue', 'purple', 'blue', 'red', 'green', 'orange', 'cyan', 'magenta', 'darkblue', 'purple', 'blue', 'red', 'green', 'orange', 'cyan', 'magenta', 'darkblue', 'purple', 'blue', 'red', 'green', 'orange', 'cyan', 'magenta', 'darkblue', 'purple', 'blue', 'red', 'green', 'orange', 'cyan', 'magenta', 'darkblue', 'purple', 'blue', 'red', 'green', 'orange', 'cyan', 'magenta', 'darkblue', 'purple']
    for i, row_range_i in enumerate(row_range_list):
        ax.axhspan(row_range_i[0], row_range_i[1], color=color_rangelist[i], alpha=0.15)#, label=f'Rows range {row_range_i[0]}-{row_range_i[1]}') 
    plt.show(block=False)
    
    
    
#######################################################################
#######################################################################
# 


def w_image_pixels_to_calibrated_wavelength_v2(w_px, w_unc_px, slope_cal, slope_unc_cal, intercept_cal, intercept_unc_cal): 
    """
    This function converts the pixel numbers of the x axis (wavelength direction) in the image to calibrated wavelength values (in Angstrom). 
    """
    x = w_px
    dx = w_unc_px
    m = slope_cal
    dm = slope_unc_cal
    b = intercept_cal
    db = intercept_unc_cal
    
    w_cal = m*x + b
    w_unc_cal = np.sqrt( (m*dx)**2 + (x*dm)**2 + db**2 )
    return [w_cal, w_unc_cal] #[Angstrom]


def w_calibrated_wavelength_to_image_pixels_v2(w_cal, w_unc_cal, slope_cal, slope_unc_cal, intercept_cal, intercept_unc_cal):
    """
    This function is the inverse of image_pixels_to_calibrated_wavelength()
    """
    x = w_cal
    dx = w_unc_cal
    m = slope_cal
    dm = slope_unc_cal
    b = intercept_cal
    db = intercept_unc_cal
    
    M = 1/m
    B = -b/m
    dM = abs(dm/M**2)
    dB = np.sqrt( (db/m)**2 + (b*dm/m**2)**2 )
    
    w_px = M*x + B
    w_unc_px = np.sqrt( (M*dx)**2 + (x*dM)**2 + dB**2 )
    
    return [w_px, w_unc_px] #Note: if you use this output as an index (or array of indices, depending on the input), it could make problems because it is not an integer, but it should be an integer.0, but you can solve it by doing int(w_px) in case s_px is one number, or np.floor(w_px).astype(int) in case s_px is an array.


# Convert calibrated wavelength range to (closest) pixel range
def w_range_pixel_from_wavelength_calibrated_v2(w_cal_lowest, w_unc_cal_lowest, w_px_length, slope_cal, slope_unc_cal, intercept_cal, intercept_unc_cal):
    # Range of pixels
    ## 1) Calculate the first pixel of the range (float) and its uncertainty
    w_px_start_float, w_unc_px_start_float = w_calibrated_wavelength_to_image_pixels_v2(w_cal=w_cal_lowest, w_unc_cal=w_unc_cal_lowest, slope_cal=slope_cal, slope_unc_cal=slope_unc_cal, intercept_cal=intercept_cal, intercept_unc_cal=intercept_unc_cal)
    w_px_range_float =  [w_px_start_float, w_px_start_float+w_px_length]
    w_unc_px_range_float = [w_unc_px_start_float, w_unc_px_start_float]
    ## 2) Round and convert to integer 
    w_px_start = int(np.round(w_px_start_float))
    w_px_range =  [w_px_start, w_px_start+w_px_length]
    ### Uncertainties
    if w_unc_px_start_float<0.5: w_unc_px_start = 0
    else: w_unc_px_start = max(0,w_unc_px_start_float)
    w_unc_px_range =  [w_unc_px_start, w_unc_px_start]
    
    # Calculate the range of wavelength calibrated from the center of the pixels range
    ## 1)
    w_cal_range0_pixcenter, w_unc_cal_range0_pixcenter = w_image_pixels_to_calibrated_wavelength_v2(w_px=float(w_px_range[0]), w_unc_px=w_unc_px_range[0], slope_cal=slope_cal, slope_unc_cal=slope_unc_cal, intercept_cal=intercept_cal, intercept_unc_cal=intercept_unc_cal)
    w_cal_range1_pixcenter, w_unc_cal_range1_pixcenter = w_image_pixels_to_calibrated_wavelength_v2(w_px=float(w_px_range[1]), w_unc_px=w_unc_px_range[1], slope_cal=slope_cal, slope_unc_cal=slope_unc_cal, intercept_cal=intercept_cal, intercept_unc_cal=intercept_unc_cal)
    w_cal_range_pixcenter = [w_cal_range0_pixcenter, w_cal_range1_pixcenter]
    w_unc_cal_range_pixcenter = [w_unc_cal_range0_pixcenter, w_unc_cal_range1_pixcenter]
    
    return [w_px_range, w_unc_px_range, w_cal_range_pixcenter, w_unc_cal_range_pixcenter, w_px_range_float, w_unc_px_range_float]



def crop_spectra_list(sumer_data_list, sumer_data_unc_list, slit_top_px, slit_bottom_px):
    sumer_data_list_croplat, sumer_data_unc_list_croplat = [],[]
    for i in range(len(sumer_data_list)):
        sumer_data_list_croplat.append(sumer_data_list[i][slit_top_px:slit_bottom_px+1, :])
        sumer_data_unc_list_croplat.append(sumer_data_unc_list[i][slit_top_px:slit_bottom_px+1, :])
    return [sumer_data_list_croplat, sumer_data_unc_list_croplat]


def create_spectroheliogram_allrows_1col(spectrum_array, spectrum_array_unc, w_cal_lowest, w_unc_cal_lowest, w_px_length, w_cal_lowest_bckg, w_unc_cal_lowest_bckg, w_px_length_bckg, slope_list, slope_unc_list, intercept_list, intercept_unc_list):
    
    average_intensity_line, average_intensity_unc_line, average_intensity_bckg, average_intensity_unc_bckg, average_intensity_line_nobg, average_intensity_unc_line_nobg = [],[],[],[],[],[]
    for i in range(len(slope_list)):
        # Slope and intercept for each row
        slope_cal = slope_list[i]
        slope_unc_cal = slope_unc_list[i]
        intercept_cal = intercept_list[i]
        intercept_unc_cal = intercept_unc_list[i]
        
        # Calculate the range of pixels and Angstrom in the wavelength direction for each row
        w_px_range, w_unc_px_range, w_cal_range_pixcenter, w_unc_cal_range_pixcenter, w_px_range_float, w_unc_px_range_float = w_range_pixel_from_wavelength_calibrated_v2(w_cal_lowest=w_cal_lowest, w_unc_cal_lowest=w_unc_cal_lowest, w_px_length=w_px_length, slope_cal=slope_cal, slope_unc_cal=slope_unc_cal, intercept_cal=intercept_cal, intercept_unc_cal=intercept_unc_cal)
        w_px_range_bckg, w_unc_px_range_bckg, w_cal_range_pixcenter_bckg, w_unc_cal_range_pixcenter_bckg, w_px_range_float_bckg, w_unc_px_range_float_bckg = w_range_pixel_from_wavelength_calibrated_v2(w_cal_lowest=w_cal_lowest_bckg, w_unc_cal_lowest=w_unc_cal_lowest, w_px_length=w_px_length_bckg, slope_cal=slope_cal, slope_unc_cal=slope_unc_cal, intercept_cal=intercept_cal, intercept_unc_cal=intercept_unc_cal)
        
        # Rows and columns indices (of the boundaries) for each row
        row = i# "i" if the spectra are cropped in latitude, otherwise it would be "row_list[i]"
        col0 = w_px_range_list[i][0]
        col1 = w_px_range_list[i][1]
        colbg0 = w_px_range_bckg_list[i][0]
        colbg1 = w_px_range_bckg_list[i][1]
        
                
        # Calculate profile of each row
        profile_intensity_line = spectrum_array[row, col0:col1+1]
        profile_intensity_unc_line = spectrum_array_unc[row, col0:col1+1]
        N_line = col1-col0+1
        profile_intensity_bckg = spectrum_array[row, colbg0:colbg1+1]
        profile_intensity_unc_bckg = spectrum_array_unc[row, colbg0:colbg1+1]
        N_bckg = colbg1-colbg0+1
        
        # Calculate the average of the profile of each row
        average_intensity_line_1row1col = np.mean(profile_intensity_line)
        average_intensity_unc_line_1row1col = (1/N_line) * np.sqrt(np.sum(profile_intensity_unc_line**2))
        average_intensity_bckg_1row1col = np.mean(profile_intensity_bckg)
        average_intensity_unc_bckg_1row1col = (1/N_bckg) * np.sqrt(np.sum(profile_intensity_unc_bckg**2))
        average_intensity_line_nobg_1row1col = average_intensity_line_1row1col - average_intensity_bckg_1row1col
        average_intensity_unc_line_nobg_1row1col = np.sqrt( average_intensity_unc_line_1row1col**2 + average_intensity_unc_bckg_1row1col**2 )
        ## Save in lists
        average_intensity_line.append(average_intensity_line_1row1col)
        average_intensity_unc_line.append(average_intensity_unc_line_1row1col)
        average_intensity_bckg.append(average_intensity_bckg_1row1col)
        average_intensity_unc_bckg.append(average_intensity_unc_bckg_1row1col)
        average_intensity_line_nobg.append(average_intensity_line_nobg_1row1col)
        average_intensity_unc_line_nobg.append(average_intensity_unc_line_nobg_1row1col)
        print(i)
        
    return [average_intensity_line, average_intensity_unc_line, average_intensity_bckg, average_intensity_unc_bckg, average_intensity_line_nobg, average_intensity_unc_line_nobg]


def create_spectroheliogram_allrows_allcols(spectra_list, spectra_unc_list, w_cal_lowest, w_unc_cal_lowest, w_px_length, w_cal_lowest_bckg, w_unc_cal_lowest_bckg, w_px_length_bckg, slope_list, slope_unc_list, intercept_list, intercept_unc_list):
    spectroheliogram_line_NT, spectroheliogram_unc_line_NT, spectroheliogram_bckg_NT, spectroheliogram_unc_bckg_NT, spectroheliogram_line_nobg_NT, spectroheliogram_unc_line_nobg_NT  = [],[],[],[],[],[]
    for j in range(len(spectra_list)):
        spectrum_array = spectra_list[j]
        spectrum_array_unc = spectra_unc_list[j]
        average_intensity_line_1col, average_intensity_unc_line_1col, average_intensity_bckg_1col, average_intensity_unc_bckg_1col, average_intensity_line_nobg_1col, average_intensity_unc_line_nobg_1col = create_spectroheliogram_allrows_1col(spectrum_array, spectrum_array_unc, w_cal_lowest, w_unc_cal_lowest, w_px_length, w_cal_lowest_bckg, w_unc_cal_lowest_bckg, w_px_length_bckg, slope_list, slope_unc_list, intercept_list, intercept_unc_list)
        spectroheliogram_line_NT.append(average_intensity_line_1col)
        #spectroheliogram_unc_line_NT.append(average_intensity_unc_line_1col)
        spectroheliogram_bckg_NT.append(average_intensity_bckg_1col)
        #spectroheliogram_unc_bckg_NT.append(average_intensity_unc_bckg_1col)
        spectroheliogram_line_nobg_NT.append(average_intensity_line_nobg_1col)
        #spectroheliogram_unc_line_nobg_NT.append(average_intensity_unc_line_nobg_1col)
        
        # Convert to array and transpose
        ## Line with background (background not subtracted)
        spectroheliogram_line = np.array(spectroheliogram_line_NT).T
        #spectroheliogram_unc_line = np.array(spectroheliogram_unc_line_NT).T
        ## Background 
        spectroheliogram_bckg = np.array(spectroheliogram_bckg_NT).T
        #spectroheliogram_unc_bckg = np.array(spectroheliogram_unc_bckg_NT).T
        ## Line without background (background subtracted)
        spectroheliogram_line_nobg = np.array(spectroheliogram_line_nobg_NT).T
        #spectroheliogram_unc_line_nobg = np.array(spectroheliogram_unc_line_nobg_NT).T
        
    #return [spectroheliogram_line, spectroheliogram_unc_line, spectroheliogram_bckg, spectroheliogram_unc_bckg, spectroheliogram_line_nobg, spectroheliogram_unc_line_nobg]
    return [spectroheliogram_line, spectroheliogram_bckg, spectroheliogram_line_nobg]

        
def get_profiles(spectrumindices_latitudeindices, spectra_list, spectra_unc_list, w_cal_lowest, w_unc_cal_lowest, w_px_length, slope_list, slope_unc_list, intercept_list, intercept_unc_list):
    profile_intensity_line_sum = np.zeros(w_px_length+1)
    profile_intensity_unc_line_sumsq = np.zeros(w_px_length+1)
    N_pixels = len(spectrumindices_latitudeindices)
    for sp, lt in spectrumindices_latitudeindices:
        # Slope and intercept for each row
        slope_cal = slope_list[i]
        slope_unc_cal = slope_unc_list[i]
        intercept_cal = intercept_list[i]
        intercept_unc_cal = intercept_unc_list[i]
        
        # Calculate the range of pixels and Angstrom in the wavelength direction for each row
        w_px_range, w_unc_px_range, w_cal_range_pixcenter, w_unc_cal_range_pixcenter, w_px_range_float, w_unc_px_range_float = w_range_pixel_from_wavelength_calibrated_v2(w_cal_lowest=w_cal_lowest, w_unc_cal_lowest=w_unc_cal_lowest, w_px_length=w_px_length, slope_cal=slope_cal, slope_unc_cal=slope_unc_cal, intercept_cal=intercept_cal, intercept_unc_cal=intercept_unc_cal)
        
        # Rows and columns indices (of the boundaries) for each row
        row = row_list[i]
        col0 = w_px_range_list[i][0]
        col1 = w_px_range_list[i][1]
                
        # Calculate profile of each row
        profile_intensity_line = spectrum_array[row, col0:col1+1]
        profile_intensity_unc_line = spectrum_array_unc[row, col0:col1+1]
        
        # Calculate the average of all the profiles
        profile_intensity_line_sum = profile_intensity_line_sum + profile_intensity_line
        profile_intensity_unc_line_sumsq = profile_intensity_unc_line_sumsq + profile_intensity_unc_line**2
    
    profile_wavelength_line = w_image_pixels_to_calibrated_wavelength(w_px_img=np.arange(col0, col1+1), slope_cal=slope_cal, intercept_cal=intercept_cal) #From the last iteration
    profile_intensity_line_average = (1/N_pixels) * profile_intensity_line_sum
    profile_intensity_unc_line_average = (1/N_pixels) * np.sqrt(profile_intensity_unc_line_sumsq)
    
    return [profile_wavelength_line, profile_intensity_line_average, profile_intensity_unc_line_average]


def average_profiles_spectroheliogram_range_intensities(proportion_range, w_px_range, spectroheliogram_array, slit_top_px, sumer_data_list, slope_cal, intercept_cal):
    lower_bound = proportion_range[0] * np.max(spectroheliogram_array)
    upper_bound = proportion_range[1] * np.max(spectroheliogram_array)
    rowcol_inside_range = np.argwhere((spectroheliogram_array>=lower_bound) & (spectroheliogram_array<=upper_bound))

    wpx0, wpx1 = min(w_px_range), max(w_px_range)
    wcal_lineprofile = w_image_pixels_to_calibrated_wavelength(w_px_img=np.arange(wpx0,wpx1+1), slope_cal=slope_cal, intercept_cal=intercept_cal)
    intensity_lineprofile_list = []
    for ii in range(len(rowcol_inside_range)):
        row_i = rowcol_inside_range[ii][0] + slit_top_px
        col_i = rowcol_inside_range[ii][1]
        data_sumer_i = sumer_data_list[col_i]
        intensity_lineprofile_i = data_sumer_i[row_i, wpx0:wpx1+1]
        intensity_lineprofile_list.append(intensity_lineprofile_i)

    # Calculate the total radiance of the line for each spatial pixel (row of the spectrogram), background subtracted
    #radiance_line_1D_nobg_i, radiance_line_1D_i = radiance_line_2D(intensity_2darray_lineprofile=spectrum_crop_i, intensity_2darray_bckg=bckg_i, px_width_Ang=px_width_Ang)
    print(len(intensity_lineprofile_list))
    intensity_lineprofile_average = np.mean(intensity_lineprofile_list, axis=0)
    return [wcal_lineprofile, intensity_lineprofile_average]


######################################################################
######################################################################

def interpolate_pixel_intensities(w_cal_range, N_interp, slope_cal, intercept_cal, pixel_index, pixel_intensity, show_figure='no'):
    # Convert pixel indices to wavelength
    pixel_centerwavelength = w_image_pixels_to_calibrated_wavelength(w_px_img=pixel_index, slope_cal=slope_cal, intercept_cal=intercept_cal)
    
    # Create an interpolation function
    interp_func = interp1d(pixel_centerwavelength, pixel_intensity, kind='linear')

    # interpolate values
    interp_wavelength = np.linspace(w_cal_range[0], w_cal_range[1], N_interp)
    interp_intensity = interp_func(interp_wavelength)
    
    if show_figure=='yes':
        # Generate new x values for smooth curve
        interp_xcurve = np.linspace(min(pixel_centerwavelength), max(pixel_centerwavelength), 100)  # More points for smooth line
        interp_ycurve = interp_func(interp_xcurve)  # Interpolated y values
        
        # Wavelengths corresponding to the edges of the pixels
        pixel_boundarywavelength = w_image_pixels_to_calibrated_wavelength(w_px_img=np.concatenate([pixel_index-0.5, [pixel_index[-1]+0.5]]), slope_cal=slope_cal, intercept_cal=intercept_cal)

        def newaxis_x(w_cal): return w_calibrated_wavelength_to_image_pixels(w_cal=w_cal, slope_cal=slope_cal, intercept_cal=intercept_cal)
        def oldaxis_x(w_px): return w_image_pixels_to_calibrated_wavelength(w_px_img=w_px, slope_cal=slope_cal, intercept_cal=intercept_cal)


        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,6))
        ax.scatter(pixel_centerwavelength, pixel_intensity, color='red', label='Intensities of every pixel (center)') # Plot the original data points
        for pbw in pixel_boundarywavelength:
            ax.axvline(pbw, color='red', linewidth=0.5)
        ax.plot([],[], color='red', linewidth=0.5, label='Edges of the pixels')
        ax.plot(interp_xcurve, interp_ycurve, linestyle='--', color='blue', label='Linear interpolation') # Plot the interpolated line
        ax.scatter(interp_wavelength, interp_intensity, color='blue', label='Interpolated points')
        ax.axvspan(interp_wavelength[0], interp_wavelength[-1], color='blue', linestyle=':', alpha=0.15, label='Boundaries of the range\n'f'{interp_wavelength[0]} - {interp_wavelength[-1]} Ang')
        ax.set_xlabel('Wavelength calibrated (Angstrom)')
        ax.set_ylabel('Intensity')
        ax.set_title('Linear Interpolation')
        secax = ax.secondary_xaxis('top', functions=(newaxis_x, oldaxis_x))
        secax.set_xlabel("Pixel index")
        ax.legend()
        plt.show()
    
    return [interp_wavelength, interp_intensity]



def interpolate_pixel_intensities_average(w_cal_range, slope_cal, intercept_cal, pixel_column, pixel_intensity, show_figure='no'):

    # rename some variables
    column_centerpixel = np.array(pixel_column)
    intensity_centerpixel = np.array(pixel_intensity)
    wavelength_range = w_cal_range

    halfpixel_wavelength = 0.5*slope_cal

    # Convert pixel columns to wavelength
    wavelength_centerpixel = w_image_pixels_to_calibrated_wavelength(w_px_img=column_centerpixel, slope_cal=slope_cal, intercept_cal=intercept_cal)
    wavelength_edgepixel = np.concatenate([wavelength_centerpixel-halfpixel_wavelength, [wavelength_centerpixel[-1]+halfpixel_wavelength]])

    # Create an interpolation function
    interp_func = interp1d(wavelength_centerpixel, intensity_centerpixel, kind='linear')

    # Inside the wavelength range (full pixels)
    index_inrange = np.where((wavelength_centerpixel-halfpixel_wavelength > wavelength_range[0]) & (wavelength_centerpixel+halfpixel_wavelength < wavelength_range[1]))[0]
    wavelength_centerpixel_inrange = wavelength_centerpixel[index_inrange]
    intensity_centerpixel_inrange = interp_func(wavelength_centerpixel_inrange)

    # Adding the partial pixels of the edge of the range
    wavelength_analysis_left = np.mean([wavelength_range[0], wavelength_centerpixel_inrange[0]-halfpixel_wavelength])
    wavelength_analysis_right = np.mean([wavelength_range[1], wavelength_centerpixel_inrange[-1]+halfpixel_wavelength])
    wavelength_analysis = np.concatenate([[wavelength_analysis_left], wavelength_centerpixel_inrange, [wavelength_analysis_right]])
    intensity_analysis = interp_func(wavelength_analysis)

    # Delta pixels (of the range)
    wavelength_deltapixel_inrange_entirepixels = slope_cal * np.ones(len(wavelength_centerpixel_inrange))
    wavelength_deltapixel_left = wavelength_range[0] - (wavelength_centerpixel_inrange[0]-halfpixel_wavelength)
    wavelength_deltapixel_right = (wavelength_centerpixel_inrange[-1]+halfpixel_wavelength) - wavelength_range[1]
    wavelength_deltapixel_inrange = np.concatenate([[wavelength_deltapixel_left], wavelength_deltapixel_inrange_entirepixels, [wavelength_deltapixel_right]])

    # Return: weighted mean of the intensity
    ia, w = intensity_analysis, wavelength_deltapixel_inrange
    intensity_weighted_mean = np.sum(w*ia) / np.sum(w)
    
    
    if show_figure=='yes':
        
        
        # Generate new x values for smooth curve
        interp_xcurve = np.linspace(min(wavelength_centerpixel), max(wavelength_centerpixel), 100)  # More points for smooth line
        interp_ycurve = interp_func(interp_xcurve)  # Interpolated y values

        
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,6))

        # Original data and interpolation
        ax.scatter(wavelength_centerpixel, pixel_intensity, color='black', label='Intensities of every pixel (center) (original data)') # Plot the original data points
        ax.plot(interp_xcurve, interp_ycurve, linestyle='--', color='black', linewidth=0.8, label='Linear interpolation') # Plot the interpolated line

        # Edges of the pixels
        for we in wavelength_edgepixel:
            ax.axvline(we, color='black', linewidth=0.5)
        ax.plot([],[], color='black', linewidth=0.5, label='Edges of the pixels')

        # Wavelength range (w_cal_range)
        ax.axvspan(w_cal_range[0], w_cal_range[1], color='blue', linestyle=':', alpha=0.15, label='Range to analyze\n'f'{w_cal_range[0]} - {w_cal_range[1]} Ang')

        # Inside the wavelength range (full pixels)
        index_inrange = np.where((wavelength_centerpixel-halfpixel_wavelength > wavelength_range[0]) & (wavelength_centerpixel+halfpixel_wavelength < wavelength_range[1]))[0]
        ax.scatter(wavelength_centerpixel_inrange, intensity_centerpixel_inrange, marker='+', s=100, color='green', label='Entire pixels inside the range (original data)') # Plot the original data points

        # Adding the partial pixels of the edge of the range
        ax.scatter(wavelength_analysis, intensity_analysis, color='red', marker='s', s=20, label='"Artificial" pixels at the edges of the range') # Plot the original data points

        # Delta pixel
        xerr = wavelength_deltapixel_inrange/2
        ax.errorbar(x=wavelength_analysis, y=intensity_analysis, xerr=xerr, color='red', linewidth=0, elinewidth=1.2)

        # Horizontal line as weighted mean
        ax.axhline(y=intensity_weighted_mean, color='blue', linestyle='--', label=f'Weighted mean of the intensity: {intensity_weighted_mean}')

        ax.set_xlabel('Wavelength calibrated (Angstrom)')
        ax.set_ylabel('Intensity')
        ax.set_title('Linear Interpolation')
        ax.legend()
        plt.show()
    
    
    return intensity_weighted_mean



######################################################################
######################################################################
# Code of Antoine Dolliou


from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.wcs import WCS
import numpy as np
import astropy.units as u
from scipy.ndimage import map_coordinates
from matplotlib.colors import LogNorm



def ang2pipi(ang):
    """ put angle between ]-180, +180] deg """
    pi = u.Quantity(180, 'deg')
    return - ((- ang + pi) % (2 * pi) - pi)


def interpol2d(image, x, y, fill, order, dst=None):
    """"
    taken from Frederic interpol2d function
    """
    bad = np.logical_or(x == np.nan, y == np.nan)
    x = np.where(bad, -1, x)
    y = np.where(bad, -1, y)

    coords = np.stack((y.ravel(), x.ravel()), axis=0)
    if dst is None:
        dst = np.empty(x.shape, dtype=image.dtype)
    map_coordinates(image, coords, order=order, mode='constant', cval=fill, output=dst.ravel(), prefilter=False)

    return dst


def build_regular_grid(longitude, latitude, lonlims=None, latlims=None):
    x = np.abs((longitude[0, 1] - longitude[0, 0]).to("deg").value)
    y = np.abs((latitude[0, 1] - latitude[0, 0]).to("deg").value)
    dlon = np.sqrt(x ** 2 + y ** 2)

    x = np.abs((longitude[1, 0] - longitude[0, 0]).to("deg").value)
    y = np.abs((latitude[1, 0] - latitude[0, 0]).to("deg").value)
    dlat = np.sqrt(x ** 2 + y ** 2)

    longitude1D = np.arange(np.min(ang2pipi(longitude).to(u.deg).value),
                            np.max(ang2pipi(longitude).to(u.deg).value), dlon)
    latitude1D = np.arange(np.min(ang2pipi(latitude).to(u.deg).value),
                           np.max(ang2pipi(latitude).to(u.deg).value), dlat)
    if (lonlims is not None) or (latlims is not None):
        longitude1D = longitude1D[(longitude1D > ang2pipi(lonlims[0]).to("deg").value) &
                                  (longitude1D < ang2pipi(lonlims[1]).to("deg").value)]
        latitude1D = latitude1D[(latitude1D > ang2pipi(latlims[0]).to("deg").value) &
                                (latitude1D < ang2pipi(latlims[1]).to("deg").value)]
    longitude_grid, latitude_grid = np.meshgrid(longitude1D, latitude1D)

    longitude_grid = longitude_grid * u.deg
    latitude_grid = latitude_grid * u.deg
    dlon = dlon * u.deg
    dlat = dlat * u.deg
    return longitude_grid, latitude_grid, dlon, dlat



def plot_EIT_map(filepath_eit, savefig='no'):

    if __name__ == "__main__":
        
        with fits.open(filepath_eit) as hdul:
            hdu = hdul[-1]
            data = hdu.data
            header = hdu.header

            w = WCS(hdu.header)

            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot()
            im = ax.imshow(data, interpolation='none', origin='lower', norm=LogNorm(), cmap='Greys_r')
            cbar = fig.colorbar(im, ax=ax)
            fig.savefig("test2.pdf")
            # header["CROTA"] = 10
            w = WCS(hdu.header)
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            lon, lat  = w.pixel_to_world(x, y)


            long, latg, dlon, dlat  = build_regular_grid(lon, lat)
            long = long.to("arcsec")
            latg = latg.to("arcsec")
            dlon = dlon.to("arcsec").value
            dlat = dlat.to("arcsec").value

            xg, yg = w.world_to_pixel(long, latg)
            data_interp = interpol2d(data, x=xg, y=yg, order=3, fill = np.nan)
            long = long.to("arcsec").value
            latg = latg.to("arcsec").value

            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot()
            im = ax.imshow(data_interp, interpolation='none', origin='lower', norm=LogNorm(), cmap='Greys_r',
                           extent=(long[0, 0] - 0.5*dlon, long[-1, -1] + 0.5 * dlon,
                                   latg[0, 0] - 0.5*dlat, latg[-1, -1] + 0.5 * dlat),)
            cbar = fig.colorbar(im, ax=ax)
            ax.set_xlabel("Solar-X")
            ax.set_ylabel("Solar-Y")
            if savefig!='no': fig.savefig("savefig.pdf")
            
        return data_interp

"""
data_interp = plot_EIT_map(filepath_eit='../../data/soho/eit/SOHO_EIT_195_19991107T042103_L1.fits', savefig='no')

"""



#######################################################################
#######################################################################
# 



