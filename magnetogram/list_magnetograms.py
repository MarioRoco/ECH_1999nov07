
##############################################################
# Import packages, variables, functions...

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datetime as dt
from astropy.io import fits


##############################################################
# 

file_index = 2

filename_list = ['991106B1',  '991106H1',  '991106M1',  '991106V1',  '991106D1',  '991106I1',  '991106Q1', '991108B1', '991108I1', '991108V1', '991108H1', '991108Q1', '991108D1', '991108M1']

  

data_list, header_list = [],[]
for filename_i in filename_list:
	data_list.append(fits.getdata('data/'+filename_i))
	header_i = fits.getheader('data/'+filename_i)
	header_list.append(header_i)
	print('###')
	print(header_i['UTDATE'], "---", header_i['UTSTART'])

data_mg = data_list[file_index]
header_mg = header_list[file_index]




filename_eit_list = ['SOHO_EIT_171_19991106T190704_L1.fits', 'SOHO_EIT_195_19991106T170227_L1.fits', 'SOHO_EIT_195_19991106T171142_L1.fits', 'SOHO_EIT_195_19991106T180438_L1.fits', 'SOHO_EIT_195_19991106T181327_L1.fits', 'SOHO_EIT_195_19991107T034617_L1.fits']


file_eit_index = 1

filename_eit_list = filename_eit_list[file_eit_index]

data_eit = fits.getdata('data/'+filename_eit_list)
header_eit = fits.getheader('data/'+filename_eit_list)



### Inverse
data_eit_inverse = data_eit[::-1]
data_mg_inverse = data_mg[::-1]

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
#ax.imshow(data_mg_inverse)#, norm=LogNorm(vmin=v_min_eit, vmax=v_max_eit), cmap='Greys_r', extent=extent_eit_sumer)
#ax.imshow(data_mg_inverse, norm=LogNorm(), cmap='Greys_r')
ax.imshow(data_mg_inverse, vmin=-v_max, vmax=v_max)
ax.axis('equal') # Ensures equal scaling of axis x and y
ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
plt.show(block=False)



### Magnetogram full Sun, more contrast
v_max = 10.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,8.25))
#ax.imshow(data_mg_inverse)#, norm=LogNorm(vmin=v_min_eit, vmax=v_max_eit), cmap='Greys_r', extent=extent_eit_sumer)
#ax.imshow(data_mg_inverse, norm=LogNorm(), cmap='Greys_r')
ax.imshow(data_mg_inverse, vmin=-v_max, vmax=v_max)
ax.axis('equal') # Ensures equal scaling of axis x and y
ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
plt.show(block=False)


