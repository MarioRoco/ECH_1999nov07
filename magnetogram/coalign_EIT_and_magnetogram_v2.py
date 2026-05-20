
##############################################################
# Import packages, variables, functions...

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



##############################################################
# 

#filename_list = ['991106B1',  '991106H1',  '991106M1',  '991106V1',  '991106D1',  '991106I1',  '991106Q1']
filename_mag = '991106M1'

header_mag = fits.getheader('data/'+filename_mag)
data_mag = fits.getdata('data/'+filename_mag)
data_mag = data_mag * header_mag['BSCALE'] + header_mag['BZERO'] #header: BSCALE  =       1.0000000000E0  /  REAL = TAPE*BSCALE + BZERO
data_mag_inverse = data_mag[::-1]



#filename_eit = 'SOHO_EIT_171_19991106T190704_L1.fits'
filename_eit = 'SOHO_EIT_195_19991106T170227_L1.fits'
#filename_eit = 'SOHO_EIT_195_19991106T171142_L1.fits'
#filename_eit = 'SOHO_EIT_195_19991106T180438_L1.fits'
#filename_eit = 'SOHO_EIT_195_19991106T181327_L1.fits'
#filename_eit = 'SOHO_EIT_195_19991107T034617_L1.fits'

header_eit = fits.getheader('data/'+filename_eit)
data_eit = fits.getdata('data/'+filename_eit)
data_eit_inverse = data_eit[::-1]



print('###')
print('Magnetogram:', header_mag['UTDATE'], "---", header_mag['UTSTART'], '[UT]')
print('EIT:        ', header_eit['DATE-OBS'], '[UTC]')


##############################################################
# EIT map in coordinates

Nx = header_eit['NAXIS1']
Ny = header_eit['NAXIS2']
x_eit_arcsec = X__pixel_to_HP(x_px=np.arange(Nx), header_eit=header_eit)
y_eit_arcsec = Y__pixel_to_HP(y_px=np.arange(Ny), header_eit=header_eit)


### EIT full Sun, coordinates instead of pixels
fig, ax = plt.subplots(figsize=(9, 8.25))
ax.pcolormesh(x_eit_arcsec, y_eit_arcsec, data_eit_inverse, norm=LogNorm(), cmap='Greys_r', shading='auto')
ax.axis('equal')
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=17)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=17)
ax.set_xlim([-1000,1000])
ax.set_ylim([-1000,1000])
plt.show(block=False)


##############################################################
# Magnetogram in coordinates

header_mag['CTYPE1'] = 'HPLN-TAN'
header_mag['CTYPE2'] = 'HPLT-TAN'
header_mag['CUNIT1'] = 'arcsec'
header_mag['CUNIT2'] = 'arcsec'
header_mag['CDELT1'] = header_mag['SCALE']
header_mag['CDELT2'] = header_mag['SCALE']
header_mag['CRPIX1'] = header_mag['E_XCEN']
header_mag['CRPIX2'] = header_mag['E_YCEN']
header_mag['CRVAL1'] = 0.0
header_mag['CRVAL2'] = 0.0


Nx = header_mag['NAXIS1']
Ny = header_mag['NAXIS2']
x_mag_arcsec = X__pixel_to_HP(x_px=np.arange(Nx), header_eit=header_mag)
y_mag_arcsec = Y__pixel_to_HP(y_px=np.arange(Ny), header_eit=header_mag)

### Magnetogram full Sun, coordinates instead of pixels
v_max = 10.
fig, ax = plt.subplots(figsize=(9, 8.25))
ax.pcolormesh(x_mag_arcsec, y_mag_arcsec, data_mag_inverse, vmin=-v_max, vmax=v_max)
ax.axis('equal')
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=17)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=17)
ax.set_xlim([-1000,1000])
ax.set_ylim([-1000,1000])
plt.show(block=False)

##############################################################
# 

#contour_intensity_eit_fullsun = 135. #110.
contour_intensity_eit_fullsun = 132.58588


color_contours = 'red'

# Extent to convert the axes to helioprojective units (arcseconds)
extent_eit_fullsun_HP = helioprojective_extent_EIT(header_eit=header_eit)


### Magnetogram full Sun, coordinates instead of pixels
v_max = 10.
fig, ax = plt.subplots(figsize=(9, 8.25))
ax.pcolormesh(x_mag_arcsec, y_mag_arcsec, data_mag_inverse, vmin=-v_max, vmax=v_max)
ax.axis('equal')
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=17)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=17)
# Contours
from skimage import measure
contours = measure.find_contours(data_eit_inverse[::-1], level=contour_intensity_eit_fullsun) # Find contours at a given level
solar_center = (0,0)  # Helioprojective (x, y) center of the Sun
solar_radius = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius   
largest_contour = None
max_points = 0
for contour in contours:
    # Convert image coordinates to helioprojective coordinates
    x_contour = np.interp(contour[:, 1], [0, data_eit_inverse.shape[1]], [extent_eit_fullsun_HP[0], extent_eit_fullsun_HP[1]])
    y_contour = np.interp(contour[:, 0], [0, data_eit_inverse.shape[0]], [extent_eit_fullsun_HP[2], extent_eit_fullsun_HP[3]])
    # Filter out contour points outside the solar disk
    distances = np.sqrt((x_contour - solar_center[0])**2 + (y_contour - solar_center[1])**2)
    inside_mask = distances < solar_radius
    # Only plot if contour has at least some points inside solar disk
    if np.any(inside_mask):
        x_inside = x_contour[inside_mask]
        y_inside = y_contour[inside_mask]
        ax.plot(x_inside, y_inside, color=color_contours, linewidth=1.5)
ax.plot([],[], color=color_contours, linewidth=1.5, label=f'Contours {contour_intensity_eit_fullsun} DN/s')
ax.set_xlim([-1000,1000])
ax.set_ylim([-1000,1000])
plt.show(block=False)




### EIT full Sun, coordinates instead of pixels
fig, ax = plt.subplots(figsize=(9, 8.25))
ax.pcolormesh(x_eit_arcsec, y_eit_arcsec, data_eit_inverse, norm=LogNorm(), cmap='Greys_r', shading='auto')
ax.axis('equal')
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=17)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=17)
# Contours
from skimage import measure
contours = measure.find_contours(data_eit_inverse[::-1], level=contour_intensity_eit_fullsun) # Find contours at a given level
solar_center = (0,0)  # Helioprojective (x, y) center of the Sun
solar_radius = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius   
largest_contour = None
max_points = 0
for contour in contours:
    # Convert image coordinates to helioprojective coordinates
    x_contour = np.interp(contour[:, 1], [0, data_eit_inverse.shape[1]], [extent_eit_fullsun_HP[0], extent_eit_fullsun_HP[1]])
    y_contour = np.interp(contour[:, 0], [0, data_eit_inverse.shape[0]], [extent_eit_fullsun_HP[2], extent_eit_fullsun_HP[3]])
    # Filter out contour points outside the solar disk
    distances = np.sqrt((x_contour - solar_center[0])**2 + (y_contour - solar_center[1])**2)
    inside_mask = distances < solar_radius
    # Only plot if contour has at least some points inside solar disk
    if np.any(inside_mask):
        x_inside = x_contour[inside_mask]
        y_inside = y_contour[inside_mask]
        ax.plot(x_inside, y_inside, color=color_contours, linewidth=1.5)
ax.plot([],[], color=color_contours, linewidth=1.5, label=f'Contours {contour_intensity_eit_fullsun} DN/s')
ax.set_xlim([-1000,1000])
ax.set_ylim([-1000,1000])
plt.show(block=False)









### EIT full Sun, coordinates instead of pixels
fig, ax = plt.subplots(figsize=(9, 8.25))
ax.pcolormesh(x_eit_arcsec, y_eit_arcsec, data_eit_inverse, norm=LogNorm(), cmap='Greys_r', shading='auto')
ax.axis('equal')
ax.set_title(f'Magnetogram {header_eit["DATE-OBS"]} UTC', fontsize=20)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=17)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=17)
# Contours
from skimage import measure
contours = measure.find_contours(data_eit_inverse[::-1], level=contour_intensity_eit_fullsun) # Find contours at a given level
solar_center = (0,0)  # Helioprojective (x, y) center of the Sun
solar_radius = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius   
largest_contour = None
max_points = 0
for contour in contours:
    # Convert image coordinates to helioprojective coordinates
    x_contour = np.interp(contour[:, 1], [0, data_eit_inverse.shape[1]], [extent_eit_fullsun_HP[0], extent_eit_fullsun_HP[1]])
    y_contour = np.interp(contour[:, 0], [0, data_eit_inverse.shape[0]], [extent_eit_fullsun_HP[2], extent_eit_fullsun_HP[3]])
    # Filter out contour points outside the solar disk
    distances = np.sqrt((x_contour - solar_center[0])**2 + (y_contour - solar_center[1])**2)
    inside_mask = distances < solar_radius
    x_inside = x_contour[inside_mask]
    y_inside = y_contour[inside_mask]
    # Keep only the largest contour (or contours above a size threshold)
    if len(x_inside) > max_points:
        largest_contour = (x_inside, y_inside)
        max_points = len(x_inside)
# Plot the largest contour only
if largest_contour is not None:
    ax.plot(largest_contour[0], largest_contour[1], color=color_contours, linewidth=1.5)
ax.plot([],[], color=color_contours, linewidth=1.5, label=f'Contours {contour_intensity_eit_fullsun} DN/s')
ax.set_xlim([-1000,1000])
ax.set_ylim([-1000,1000])
plt.show(block=False)






### Magnetogram full Sun, coordinates instead of pixels
v_max = 100.
fig, ax = plt.subplots(figsize=(11, 12))
img = ax.pcolormesh(x_mag_arcsec, y_mag_arcsec, data_mag_inverse, vmin=-v_max, vmax=v_max, cmap='Greys_r')
#cax = fig.add_axes([0.92, 0.11, 0.03, 0.77])  # [left, bottom, width, height]
#cbar = fig.colorbar(img, ax=ax, cax=cax, pad=0.01)
#cbar.set_label(f'Magnetic field (Gauss)', fontsize=16)
cax = fig.add_axes([0.88, 0.11, 0.025, 0.77])
cbar = fig.colorbar(img, ax=ax, cax=cax, pad=0.01)
cbar.set_label('Magnetic field (Gauss)', fontsize=16, labelpad=12)
ax.axis('equal')
ax.set_title(f'Magnetogram {header_mag["UTDATE"]} - {header_mag["UTSTART"]} UT', fontsize=20)
ax.set_xlabel('Helioprojective longitude (arcsec)', fontsize=17)
ax.set_ylabel('Helioprojective latitude (arcsec)', fontsize=17)
# Contours
from skimage import measure
contours = measure.find_contours(data_eit_inverse[::-1], level=contour_intensity_eit_fullsun) # Find contours at a given level
solar_center = (0,0)  # Helioprojective (x, y) center of the Sun
solar_radius = header_eit['RSUN_OBS'] #[arcsec] apparent photospheric solar radius   
largest_contour = None
max_points = 0
for contour in contours:
    # Convert image coordinates to helioprojective coordinates
    x_contour = np.interp(contour[:, 1], [0, data_eit_inverse.shape[1]], [extent_eit_fullsun_HP[0], extent_eit_fullsun_HP[1]])
    y_contour = np.interp(contour[:, 0], [0, data_eit_inverse.shape[0]], [extent_eit_fullsun_HP[2], extent_eit_fullsun_HP[3]])
    # Filter out contour points outside the solar disk
    distances = np.sqrt((x_contour - solar_center[0])**2 + (y_contour - solar_center[1])**2)
    inside_mask = distances < solar_radius
    x_inside = x_contour[inside_mask]
    y_inside = y_contour[inside_mask]
    # Keep only the largest contour (or contours above a size threshold)
    if len(x_inside) > max_points:
        largest_contour = (x_inside, y_inside)
        max_points = len(x_inside)
# Plot the largest contour only
if largest_contour is not None:
    ax.plot(largest_contour[0], largest_contour[1], color=color_contours, linewidth=1.5)
ax.plot([],[], color=color_contours, linewidth=1.5, label=f'Contours {contour_intensity_eit_fullsun} DN/s')
ax.set_xlim([-1000,1000])
ax.set_ylim([-1000,1000])
plt.show(block=False)









