




######################################################
########################### wavelength array
lam_sumer_cropNeVIII_list = []
v_sumer_cropNeVIII_list = []
ev_sumer_cropNeVIII_list = []
########################### uncorrected
lam_peak_uncorrected_list = []
elam_peak_uncorrected_list = []
v_peak_uncorrected_list = []
ev_peak_uncorrected_list = []
rad_peak_uncorrected_list = []
erad_peak_uncorrected_list = []
rad_sumer_cropNeVIII_uncorrected_list = []
erad_sumer_cropNeVIII_uncorrected_list = []
########################### corrected
lam_peak_corrected_list = []
elam_peak_corrected_list = []
v_peak_corrected_list = []
ev_peak_corrected_list = []
rad_peak_corrected_list = []
erad_peak_corrected_list = []
rad_sumer_cropNeVIII_corrected_list = []
erad_sumer_cropNeVIII_corrected_list = []
########################### EIT
line_label_list = []
eit_wavelength_list = []
eit_time_list = []
rad_integrated_eit_range_list = [] #W/sr/m^2
rad_integrated_eit_mean_list = [] #W/sr/m^2
erad_integrated_eit_list = [] #W/sr/m^2
range_percentage_eit_list_list = []
percentage_contour_eit_list = []
threshold_value_type_eit_list = []
########################### SUMER
rad_integrated_sumer_range_list = [] #W/sr/m^2
rad_integrated_sumer_mean_list = []  = []#W/sr/m^2
erad_integrated_sumer_list = [] #W/sr/m^2
range_percentage_sumer_list = []
threshold_value_type_sumer_list = []
###########################
rest_wavelength_label_list = []
###########################
######################################################





######################################################
########################### wavelength array
lam_sumer_cropNeVIII_list.append( [1540.2007532100001, 1540.2428312700001, 1540.2849093300001, 1540.32698739, 1540.36906545, 1540.41114351, 1540.45322157, 1540.49529963, 1540.53737769, 1540.57945575, 1540.62153381, 1540.66361187, 1540.70568993, 1540.74776799, 1540.78984605, 1540.83192411, 1540.87400217, 1540.91608023, 1540.95815829, 1541.00023635, 1541.04231441, 1541.08439247, 1541.12647053, 1541.16854859, 1541.21062665, 1541.25270471, 1541.29478277, 1541.33686083, 1541.37893889, 1541.42101695] )
v_sumer_cropNeVIII_list.append( [-127.4863100579608, -119.2995067257417, -111.1127033935226, -102.9259000613035, -94.7390967290844, -86.5522933968653, -78.3654900646462, -70.1786867324271, -61.99188340020801, -53.80508006798891, -45.618276735769804, -37.43147340355071, -29.244670071331615, -21.057866739112512, -12.871063406893418, -4.684260074674318, 3.502543257544779, 11.689346589763877, 19.876149921982975, 28.06295325420207, 36.24975658642117, 44.43655991864026, 52.62336325085936, 60.81016658307846, 68.99696991529757, 77.18377324751665, 85.37057657973575, 93.55737991195485, 101.74418324417395, 109.93098657639305] )
ev_sumer_cropNeVIII_list.append( [3.839660663046047, 3.8430755512384684, 3.846501850804727, 3.8499395312777587, 3.8533885621981985, 3.856848913115249, 3.8603205535875302, 3.863803453183932, 3.8672975814844537, 3.8708029080810378, 3.874319402578395, 3.8778470345948253, 3.8813857737630304, 3.884935589730912, 3.8884964521623773, 3.8920683307381165, 3.895651195156399, 3.8992450151338325, 3.9028497604061436, 3.9064654007289277, 3.9100919058784083, 3.9137292456521737, 3.917377389869926, 3.9210363083742004, 3.924705971031093, 3.928386347730979, 3.932077408389215, 3.9357791229468453, 3.9394914613712912, 3.943214393657041] )
########################### uncorrected
lam_peak_uncorrected_list.append( 1540.830775529024 )
elam_peak_uncorrected_list.append( 0.00045032973563565365 )
v_peak_uncorrected_list.append( -4.907730609229314 )
ev_peak_uncorrected_list.append( 2.7252361663697413 )
rad_peak_uncorrected_list.append( 1.0867287018714045 )
erad_peak_uncorrected_list.append( 0.0012441634638142348 )
rad_sumer_cropNeVIII_uncorrected_list.append( [0.17630779926705314, 0.17766806207974858, 0.18144765889969863, 0.18319040302343598, 0.17834690765354153, 0.1794125216315943, 0.19720248784357125, 0.23047368054072145, 0.2779300668580518, 0.3401884596179289, 0.4365511033120229, 0.5843056670098818, 0.7607406875937165, 0.9254563272101307, 1.0484170659747942, 1.0866985314413566, 1.0439959125952005, 0.960646468554621, 0.8254983372828775, 0.6331960892546378, 0.4461950460575748, 0.3163573665704448, 0.2421746988087394, 0.2043606503760423, 0.1920778013308431, 0.2043559724527568, 0.23087359323313988, 0.2430807466648607, 0.22729874500954156, 0.20580953471620728] )
erad_sumer_cropNeVIII_uncorrected_list.append( [0.0004911238136146272, 0.0004932748271558639, 0.0004993920020872913, 0.0005019996304144685, 0.0004944968346498219, 0.0004966007359680804, 0.0005224856093921882, 0.000566406509216023, 0.0006237355616329567, 0.0006912240736858648, 0.0007853206759018755, 0.0009119166554391511, 0.0010420401177614344, 0.001149078738137979, 0.0012233078314336956, 0.0012439400444762647, 0.001216943960887602, 0.001167206459288435, 0.001081003486486059, 0.0009430868580356349, 0.0007889933550455703, 0.000665050457934942, 0.0005842553358739159, 0.0005386492361425038, 0.0005236992646008977, 0.0005426337166143852, 0.0005791644039176655, 0.0005942758350243281, 0.0005725154185503924, 0.0005449690179596191] )
########################### corrected
lam_peak_corrected_list.append( 1540.8455250450409 )
elam_peak_corrected_list.append( 0.0005473276255806982 )
v_peak_corrected_list.append( -2.038031129864631 )
ev_peak_corrected_list.append( 2.7259342264045627 )
rad_peak_corrected_list.append( 0.8765746447580457 )
erad_peak_corrected_list.append( 0.0012864047247651853 )
rad_sumer_cropNeVIII_corrected_list.append( [0.01107098198878595, 0.01037760991520048, 0.009415737774585486, 0.009115193316118736, 0.011290032140611639, 0.020104990992334537, 0.03513621049222207, 0.057833257121178094, 0.09584418876431067, 0.15575438779871734, 0.24907009253141382, 0.37798606563694753, 0.5235898654338345, 0.6647757027290122, 0.7886293853106876, 0.8713269666067456, 0.8535696605049828, 0.7538147611724787, 0.5904936891838191, 0.416554219374195, 0.26607283282639793, 0.1553961949063031, 0.08433866143612548, 0.045469063021682576, 0.027175027432563725, 0.022228606617447355, 0.022030824793917214, 0.020195527694419763, 0.013095344804252368, 0.0098050949652978] )
erad_sumer_cropNeVIII_corrected_list.append( [0.0004937163619423853, 0.000495920518771234, 0.0005021552865446593, 0.0005048141461424547, 0.000497128677736436, 0.000498984554083772, 0.0005248308443649291, 0.0005688615885266845, 0.0006262160563245766, 0.0006935212344080003, 0.0007874106409021645, 0.0009140966530465119, 0.0010445606304367921, 0.0011518405563864305, 0.0012258847052536244, 0.0012456823416094143, 0.0012183364238242836, 0.0011689189059945304, 0.001083389625472004, 0.0009454109177824403, 0.000790913712090214, 0.0006668695014889599, 0.0005862456404112054, 0.0005408363125215004, 0.0005261215188774282, 0.000545484428558709, 0.0005826749266083016, 0.0005981716851171351, 0.0005762505006594692, 0.0005482552749218318] )
########################### EIT
line_label_list.append( "NeVIII" )
eit_wavelength_list.append( 195 )
eit_time_list.append( "late" )
rad_integrated_eit_range_list.append([ 170.08305 , 3646.112 ]) #W/sr/m^2
rad_integrated_eit_mean_list.append( 1908.0975 ) #W/sr/m^2
erad_integrated_eit_list.append( 1738.0145 ) #W/sr/m^2
range_percentage_eit_list_list.append( [90.0, np.float32(1929.3519)] )
percentage_contour_eit_list.append( 90.0 )
threshold_value_type_eit_list.append( "mean" )
########################### SUMER
rad_integrated_sumer_range_list.append([ 0.21821112232935017 , 0.43642224465870033 ]) #W/sr/m^2
rad_integrated_sumer_mean_list.append( 0.32731668349402526 ) #W/sr/m^2
erad_integrated_sumer_list.append( 0.10910556116467508 ) #W/sr/m^2
range_percentage_sumer_list.append( [50.0, 100.0] )
threshold_value_type_sumer_list.append( "mean" )
###########################
rest_wavelength_label_list.append( "Peter_and_Judge_1999" )
###########################
######################################################






## Rest wavelength used
rest_wavelength_label = rest_wavelength_label_list[0] #'Peter_and_Judge_1999', 'SUMER_atlas', 'Peter_1998', 'Dammasch_1999', 'Kelly_database'
lam_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][0] #Angstrom
lam_unc_0 = 2.*NeVIII_theoretical_wavelength_dic[rest_wavelength_label][1] #Angstrom
print('Rest wavelength Ne VIII (2nd order):', lam_0, r'$\pm$', lam_unc_0, '\u212B')
v_unc_0 = vkms_doppler_unc(lamb=lam_0, lamb_unc=lam_unc_0, lamb_0=lam_0, lamb_0_unc=lam_unc_0) # uncertainty of the rest wavelength in km/s



# Convert some lists to Numpy arrays
## x, dx, y, dy = rad_peak_uncorrected, erad_peak_uncorrected, lam_peak_uncorrected, elam_peak_uncorrected
lam_peak_uncorrected = np.array(lam_peak_uncorrected)
rad_peak_uncorrected = np.array(rad_peak_uncorrected)
elam_peak_uncorrected = np.array(elam_peak_uncorrected)
erad_peak_uncorrected = np.array(erad_peak_uncorrected)
## x, dx, y, dy = rad_peak_corrected, erad_peak_corrected, lam_peak_corrected, elam_peak_corrected
lam_peak_corrected = np.array(lam_peak_corrected)
rad_peak_corrected = np.array(rad_peak_corrected)
elam_peak_corrected = np.array(elam_peak_corrected)
erad_peak_corrected = np.array(erad_peak_corrected)
## x, dx = rad_integrated_mean, erad_integrated
rad_integrated_mean = np.array(rad_integrated_mean)
erad_integrated = np.array(erad_integrated)


# Calculate Doppler velocity from the wavelength 
## y, dy = v_peak_original, ev_peak_original
v_peak_uncorrected = vkms_doppler(lamb=lam_peak_uncorrected, lamb_0=lamb_0_Ang)
ev_peak_uncorrected = vkms_doppler_unc(lamb=lam_peak_uncorrected, lamb_unc=elam_peak_uncorrected, lamb_0=lamb_0_Ang, lamb_0_unc=lamb_unc_0_Ang)
## y, dy = v_peak_clean, ev_peak_clean
v_peak_corrected = vkms_doppler(lamb=lam_peak_corrected, lamb_0=lamb_0_Ang)
ev_peak_corrected = vkms_doppler_unc(lamb=lam_peak_corrected, lamb_unc=elam_peak_corrected, lamb_0=lamb_0_Ang, lamb_0_unc=lamb_unc_0_Ang)






x_original, dx_original, y_original, dy_original = rad_peak_original, erad_peak_original, lam_peak_original, elam_peak_original
x_clean, dx_clean, y_clean, dy_clean = rad_peak_clean, erad_peak_clean, lam_peak_clean, elam_peak_clean
x_integrated, dx_integrated = rad_integrated_mean, erad_integrated



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=x_original, xerr=dx_original, y=y_original, yerr=dy_original, color='red', linewidth=0., elinewidth=1.0, marker='.', label='SUMER original')
ax.errorbar(x=x_clean, xerr=dx_clean, y=y_clean, yerr=dy_clean, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER clean (HRTS subtracted)')
ax.set_title(f'SOHO/SUMER, profile averaged', fontsize=18) 
ax.set_xlabel(r'Radiance of the peak (W/sr/m$^2$/''\u212B)', color='black', fontsize=16)
ax.set_ylabel('Wavelength (\u212B)', color='black', fontsize=16)
for i, label_i in enumerate(list(NeVIII_theoretical_wavelength_dic)):
    wl_, dwl_ = NeVIII_theoretical_wavelength_dic[label_i]
    wl_, dwl_ = 2.*wl_, 2.*dwl_
    color_ = NeVIII_theoretical_wavelength_color_dic[label_i]
    ax.axhline(y=wl_, color=color_, linewidth=1.2, linestyle='--')#, label=label_i) 
    ax.axhspan(wl_-dwl_, wl_+dwl_, color=color_, alpha=0.05)
ax.legend()
plt.show(block=False)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=x_integrated, xerr=dx_integrated, y=y_original, yerr=dy_original, color='red', linewidth=0., elinewidth=1.0, marker='.', label='SUMER original')
ax.errorbar(x=x_integrated, xerr=dx_integrated, y=y_clean, yerr=dy_clean, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER clean (HRTS subtracted)')
ax.set_title(f'SOHO/SUMER, profile averaged', fontsize=18) 
ax.set_xlabel(r'Radiance of the peak (W/sr/m$^2$/''\u212B)', color='black', fontsize=16)
ax.set_ylabel('Wavelength (\u212B)', color='black', fontsize=16)
for i, label_i in enumerate(list(NeVIII_theoretical_wavelength_dic)):
    wl_, dwl_ = NeVIII_theoretical_wavelength_dic[label_i]
    wl_, dwl_ = 2.*wl_, 2.*dwl_
    color_ = NeVIII_theoretical_wavelength_color_dic[label_i]
    ax.axhline(y=wl_, color=color_, linewidth=1.2, linestyle='--')#, label=label_i) 
    ax.axhspan(wl_-dwl_, wl_+dwl_, color=color_, alpha=0.05)
ax.legend()
plt.show(block=False)







x_original, dx_original, y_original, dy_original = rad_peak_original, erad_peak_original, v_peak_original, ev_peak_original
x_clean, dx_clean, y_clean, dy_clean = rad_peak_clean, erad_peak_clean, v_peak_clean, ev_peak_clean
x_integrated, dx_integrated = rad_integrated_mean, erad_integrated


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=x_original, xerr=dx_original, y=y_original, yerr=dy_original, color='red', linewidth=0., elinewidth=1.0, marker='.', label='SUMER original')
ax.errorbar(x=x_clean, xerr=dx_clean, y=y_clean, yerr=dy_clean, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER clean (HRTS subtracted)')
ax.set_title(f'SOHO/SUMER, profile averaged', fontsize=18) 
ax.set_xlabel(r'Mean intensity of the pixel list (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0.0, color=color_, linewidth=1.2, linestyle='--', label=label_line_theo) 
ax.axhspan(0.0-v_unc_0_Ang, 0.0+v_unc_0_Ang, color=color_, alpha=0.05)
ax.legend()
plt.show(block=False)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
ax.errorbar(x=x_integrated, xerr=dx_integrated, y=y_original, yerr=dy_original, color='red', linewidth=0., elinewidth=1.0, marker='.', label='SUMER original')
ax.errorbar(x=x_integrated, xerr=dx_integrated, y=y_clean, yerr=dy_clean, color='blue', linewidth=0., elinewidth=1.0, marker='.', label='SUMER clean (HRTS subtracted)')
ax.set_title(f'SOHO/SUMER, profile averaged', fontsize=18) 
ax.set_xlabel(r'Mean intensity of the pixel list (W/sr/m$^2$)', color='black', fontsize=16)
ax.set_ylabel('Doppler shift (km/s)', color='black', fontsize=16)
ax.axhline(y=0.0, color=color_, linewidth=1.2, linestyle='--', label=label_line_theo) 
ax.axhspan(0.0-v_unc_0_Ang, 0.0+v_unc_0_Ang, color=color_, alpha=0.05)
ax.legend()
plt.show(block=False)
