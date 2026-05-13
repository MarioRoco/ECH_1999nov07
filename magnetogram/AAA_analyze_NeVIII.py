
##############################################################
# Import packages, variables, functions...

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datetime as dt
from astropy.io import fits


##############################################################
# 

m1 = '991108B1'    
m2 = '991108I1'    
m3 = '991108V1'
m4 = '991108H1'    
m5 = '991108Q1'    
m6 = '991108D1'    
m7 = '991108M1'   


filename = m2
data_mg = fits.getdata('data/'+filename)
header_mg = fits.getheader('data/'+filename)


##############################################################
# 


#Show the EIT image cropped and the spectroheliogram in the same figure
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,8.25))
#ax.imshow(data_mg)#, norm=LogNorm(vmin=v_min_eit, vmax=v_max_eit), cmap='Greys_r', extent=extent_eit_sumer)
#ax.imshow(data_mg, norm=LogNorm(), cmap='Greys_r')
ax.imshow(data_mg)
ax.axis('equal') # Ensures equal scaling of axis x and y
ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
plt.show(block=False)




data_mg_crop = data_mg[1000:1300+1, 950:1400+1]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,8.25))
#ax.imshow(data_mg)#, norm=LogNorm(vmin=v_min_eit, vmax=v_max_eit), cmap='Greys_r', extent=extent_eit_sumer)
#ax.imshow(data_mg, norm=LogNorm(), cmap='Greys_r')
ax.imshow(data_mg_crop)
ax.axis('equal') # Ensures equal scaling of axis x and y
ax.set_xlabel('Helioprojective longitude (arcsec). Rot. compensated', fontsize=17)
fig.supylabel('Helioprojective latitude (arcsec)', fontsize=17)
plt.show(block=False)

