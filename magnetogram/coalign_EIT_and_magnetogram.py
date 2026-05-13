
##############################################################
# Import packages, variables, functions...

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datetime as dt
from astropy.io import fits


##############################################################
# 

#filename_list = ['991106B1',  '991106H1',  '991106M1',  '991106V1',  '991106D1',  '991106I1',  '991106Q1']
filename_mag = '991106M1'

header_mag = fits.getheader('data/'+filename_mag)
data_mag = fits.getdata('data/'+filename_mag)
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
# 

#Magnetogram
xcen_mag, ycen_mag = header_mag['E_XCEN'], header_mag['E_YCEN'] #Center of the solar disk
xcorr, ycorr = -9, 0
xcen_mag_corr, ycen_mag_corr = header_mag['E_XCEN'], header_mag['E_YCEN'] #Center of the solar disk, corrected
D_sun_mag = 148290000000. #[m] (from Stellarium)
R_sun_mag = header_eit['RSUN_REF'] #[m] (from EIT header)
R_sun_mag_arcsec = np.arctan(R_sun_mag/D_sun_mag) *180.*3600./np.pi
R_sun_mag_px = R_sun_mag_arcsec / header_mag['SCALE'] #Radius of the solar disk in pixels in the magnetogram


#EIT
R_sun_eit_arcsec = header_eit['RSUN_ARC']
R_sun_eit_px = R_sun_eit_arcsec / header_eit['CDELT1']

##############################################################
# 

### EIT full Sun
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,8.25))
ax.imshow(data_eit_inverse, norm=LogNorm(), cmap='Greys_r')
ax.axis('equal') # Ensures equal scaling of axis x and y
ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
plt.show(block=False)


### Magnetogram full Sun
v_max = 500.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,8.25))
#ax.imshow(data_mag_inverse)#, norm=LogNorm(vmin=v_min_eit, vmax=v_max_eit), cmap='Greys_r', extent=extent_eit_sumer)
#ax.imshow(data_mag_inverse, norm=LogNorm(), cmap='Greys_r')
ax.imshow(data_mag_inverse, vmin=-v_max, vmax=v_max)
ax.axis('equal') # Ensures equal scaling of axis x and y
ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
plt.show(block=False)



### Magnetogram full Sun, more contrast
v_max = 10.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,8.25))
#ax.imshow(data_mag_inverse)#, norm=LogNorm(vmin=v_min_eit, vmax=v_max_eit), cmap='Greys_r', extent=extent_eit_sumer)
#ax.imshow(data_mag_inverse, norm=LogNorm(), cmap='Greys_r')
ax.imshow(data_mag_inverse, vmin=-v_max, vmax=v_max)
ax.axis('equal') # Ensures equal scaling of axis x and y
ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
plt.show(block=False)



### Magnetogram full Sun, more contrast
v_max = 10.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,8.25))
#ax.imshow(data_mag_inverse)#, norm=LogNorm(vmin=v_min_eit, vmax=v_max_eit), cmap='Greys_r', extent=extent_eit_sumer)
#ax.imshow(data_mag_inverse, norm=LogNorm(), cmap='Greys_r')
ax.imshow(data_mag_inverse, vmin=-v_max, vmax=v_max)
ax.axis('equal') # Ensures equal scaling of axis x and y
ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
# Add circumference
from matplotlib.patches import Circle
circle = Circle((xcen_mag_corr, ycen_mag_corr), R_sun_mag_px, fill=False, edgecolor='red', linewidth=2)
ax.scatter(xcen_mag_corr, ycen_mag_corr, color='red', marker='.', s=20)
ax.add_patch(circle)
plt.show(block=False)




"""
#EIT
SOLAR_B0=              3.90000 / [deg] s/c tilt of solar North pole             
RSUN_ARC=        978.687006670 / [arcsec] apparent photospheric solar radius    
RSUN_OBS=        978.687006670 / [arcsec] apparent photospheric solar radius    
RSUN_REF=            695699968 / [m] assumed physical solar radius              
DSUN_OBS=        146622302387. / [m], s/c distance from Sun       
PC1_1   =       0.999997425984 / WCS coordinate transformation matrix           
PC1_2   =     0.00226892598004 / WCS coordinate transformation matrix           
PC2_1   =    -0.00226892598004 / WCS coordinate transformation matrix           
PC2_2   =       0.999997425984 / WCS coordinate transformation matrix           
CDELT1  =        5.25400000000 / [arcsec] pixel scale along axis 1              
CDELT2  =        5.25400000000 / [arcsec] pixel scale along axis 2              
CRVAL1  =        16.0758119752 / [arcsec] value of reference pixel along axis 1 
CRVAL2  =       -12.2257862968 / [arcsec] value of reference pixel along axis 2 
CRPIX1  =        256.500000000 / [pixel] reference pixel location along axis 1  
CRPIX2  =        256.500000000 / [pixel] reference pixel location along axis 1  
CROTA   =      -0.129999995232 / [deg] rotation angle      


# Magnetogram
SCALE   =               1.1483          / ARCSECONDS PER PIXEL
B_ZERO  =             3.824015                                                  
L_ZERO  =             16.66748                                                  
E_XCEN  =             894.6695                                                  
E_YCEN  =             888.9087                                                  
E_XSMD  =             853.4315                                                  
E_YSMD  =             851.6743                     
"""


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




import sunpy.map
from reproject import reproject_interp
import matplotlib.pyplot as plt

# Load FITS files
#map1 = sunpy.map.Map('data/'+filename_mag)
#map2 = sunpy.map.Map('data/'+filename_eit)


map1 = sunpy.map.Map(data_mag, header_mag)
map2 = sunpy.map.Map(data_eit, header_eit)

# Reproject map2 to match map1's coordinate system and shape
array, footprint = reproject_interp(map2, map1.wcs, shape_out=map1.data.shape)

# Create a new SunPy map from reprojected data
map2_aligned = sunpy.map.Map(array, map1.meta)

# Plot comparison
fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection=map1)
map1.plot(axes=ax1)
ax1.set_title("Reference")

ax2 = fig.add_subplot(122, projection=map2_aligned)
map2_aligned.plot(axes=ax2)
ax2.set_title("Aligned")

plt.show()
