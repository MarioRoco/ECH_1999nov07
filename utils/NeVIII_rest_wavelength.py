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

