import numpy as np

NeVIII_theoretical_wavelength_dic = {}
NeVIII_theoretical_wavelength_dic['SUMER_atlas'] = [770.425, np.nan]
#NeVIII_theoretical_wavelength_dic['Fawcett_1961'] = [770.42, 0.03]
#NeVIII_theoretical_wavelength_dic['Bockasten_1963'] = [770.409, 0.018] # The original error is 0.005 Angstrom, but according to Peter and Judge 1999, a more realistic error estimate is 0.018 Angstrom. 
NeVIII_theoretical_wavelength_dic['Peter_1998'] = [770.425, 0.010]
NeVIII_theoretical_wavelength_dic['Dammasch_1999'] = [770.428, 0.003]
NeVIII_theoretical_wavelength_dic['Peter_and_Judge_1999'] = [770.428, 0.007]
NeVIII_theoretical_wavelength_dic['Kelly_database'] = [770.409, np.nan]


NeVIII_theoretical_wavelength_color_dic = {}
NeVIII_theoretical_wavelength_color_dic['SUMER_atlas'] = 'blue'
#NeVIII_theoretical_wavelength_color_dic['Fawcett_1961'] = 'red'
#NeVIII_theoretical_wavelength_color_dic['Bockasten_1963'] = 'green'
NeVIII_theoretical_wavelength_color_dic['Peter_1998'] = 'orange'
NeVIII_theoretical_wavelength_color_dic['Dammasch_1999'] = 'cyan'
NeVIII_theoretical_wavelength_color_dic['Peter_and_Judge_1999'] = 'magenta'
NeVIII_theoretical_wavelength_color_dic['Kelly_database'] = 'brown'




    
"""   
In next lines I was trying to guess what is the most correct way to calculate the uncertainty of the rest wavelength in km/s

In [2]: vkms_doppler_unc(lamb=1540.856, lamb_unc=0.014, lamb_0=1540.856, lamb_0_
   ...: unc=0.014)
Out[2]: np.float64(3.852136630555179)

In [3]: vkms_doppler_unc(lamb=1540.856/2., lamb_unc=0.014/2., lamb_0=1540.856/2.
   ...: , lamb_0_unc=0.014/2.)
Out[3]: np.float64(3.852136630555179)

In [4]: vkms_doppler_unc(lamb=1540.856+0.014, lamb_unc=0., lamb_0=1540.856, lamb
   ...: _0_unc=0.014)
Out[4]: np.float64(2.723896682238359)

In [5]: vkms_doppler_unc(lamb=1540.856+0.014, lamb_unc=0.014, lamb_0=1540.856, l
   ...: amb_0_unc=0.014)
Out[5]: np.float64(3.852154130579622)

In [6]: 299792.4580*0.014/1540.856
Out[6]: 2.723871933522665
"""


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

rest_wavelength_label = 'Peter_and_Judge_1999' #'SUMER_atlas', 'Peter_1998', 'Dammasch_1999', 'Peter_and_Judge_1999', 'Kelly_database'
lam_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][0] #Angstrom
lam_unc_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][1] #Angstrom

# Probably one of the next options is the most correct way to convert the uncertainty on the rest wavelength in wavelength units to Doppler shift velocity
#v_unc_0 = vkms_doppler_unc(lamb=lam_0+lam_unc_0, lamb_unc=0.0, lamb_0=lam_0, lamb_0_unc=lam_unc_0) 
v_unc_0 = vkms_doppler(lamb=lam_0+lam_unc_0, lamb_0=lam_0)






"""
Dammasch et al. 1999 (https://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1999ESASP.446..263D&defaultprint=YES&filetype=.pdf) say "It is fundamental to determine the rest wavelength of these lines(Ne VIII, Mg X, Fe XII), which -for reasons of the high ionization stage of the ions- are hard to measure in the laboratory with high accuracy." Also they say "The Ne VIII line formed at 630 000 K has shown strong outflow velocities of the Ne 7+  ion in coronal holes, but only a small blue shift in quiet Sun regions."

Rest wavelength of Ne VIII line according to different sources:

- SUMER Atlas (https://soho.nascom.nasa.gov/hotshots/2001_07_30/sumer_atlas.html): 1540.85 Angstrom in second order, so 1540.85/2 = 770.425 Angstrom
- H. Peter 1998 (https://iopscience.iop.org/article/10.1086/307102/pdf):
    - Fawcett, Jones, & Wilson (1961) and Bockasten, Hallin, & Hughes (1963): who found 770.42  ± 0.03 and 770.409  ±  0.005 Angstrom. For this it seems reasonable that the true rest wavelength of Ne VIII is 770.425  ±  0.010 Angstrom.
    - Dammasch et al. (1999): 770.428  ±  0.003 Angstrom
    - Kelly database (https://lweb.cfa.harvard.edu/ampcgi/kelly.pl): 770.409 Angstrom
- PhD thesis of Lidong Xia (https://www.mps.mpg.de/phd/theses/equatorial-coronal-holes-and-their-relation-to-the-high-speed-solar-wind-streams.pdf), page 17 (37/211 of the PDF):
    - Dammasch et al. 1999 (https://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1999ESASP.446..263D&defaultprint=YES&filetype=.pdf): 770.428  ±  0.003 Angstrom
    - Peter and Judge 1999 (https://iopscience.iop.org/article/10.1086/307672/pdf): 770.428  ±  0.007 Angstrom
- Bockasten, Halling, and Hughes (1963): 770.409  ±  0.005 Angstrom, but according to Peter and Judge 1999 (in page 10/19 of the PDF), a more realistic error estimate is 0.018 Angstrom.
"""

