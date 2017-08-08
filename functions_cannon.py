# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:19:04 2016

@author: eilers
"""

import numpy as np
import pickle
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from scipy import ndimage
from astropy.io import ascii
import errno

# -------------------------------------------------------------------------------
# functions for continuum normalization
# -------------------------------------------------------------------------------

def gaussian_weight_matrix(wl, L):
    """ Matrix of Gaussian weights 
    
    Parameters
    ----------
    wl: numpy ndarray pixel wavelength values
    L: float width of Gaussian 
    
    Returns
    ------
    Weight matrix
    """
    return np.exp(-0.5*(wl[:,None]-wl[None,:])**2/L**2)
    
    
# -------------------------------------------------------------------------------    

def find_cont_gaussian_smooth(wl, fluxes, ivars, w):
    """ Returns the weighted mean block of spectra
   
   Parameters
    ----------
    wl: numpy ndarray wavelength vector
    flux: numpy ndarray block of flux values
    ivar: numpy ndarray block of ivar values
    w: Gaussian weights 
    
    Returns
    -------
    smoothed_fluxes: numpy ndarray block of smoothed flux values, mean spectra
    """
    bot = np.dot(ivars, w.T)
    top = np.dot(fluxes*ivars, w.T)
    bad = bot == 0
    cont = np.zeros(top.shape)
    cont[~bad] = top[~bad] / bot[~bad]
    return cont
    
# -------------------------------------------------------------------------------      

def cont_norm_gaussian_smooth(dataset, L):
    """ Continuum normalize by dividing by a Gaussian-weighted smoothed spectrum
    
    Parameters
    ----------
    dataset: Dataset, the dataset to continuum normalize
    L: float, the width of the Gaussian used for weighting
    
    Returns
    -------
    dataset: Dataset, updated dataset
    """
    
    print("Gaussian smoothing the entire dataset...")
    weights = gaussian_weight_matrix(dataset.wl, L)
    
    print("Gaussian smoothing the training set")
    cont = find_cont_gaussian_smooth(dataset.wl, dataset.tr_flux, dataset.tr_ivar, weights)
    
    norm_tr_flux, norm_tr_ivar = cont_norm(dataset.tr_flux, dataset.tr_ivar, cont)
    
    print("Gaussian smoothing the test set")
    cont = find_cont_gaussian_smooth(dataset.wl, dataset.test_flux, dataset.test_ivar, weights)
    
    norm_test_flux, norm_test_ivar = cont_norm(dataset.test_flux, dataset.test_ivar, cont)
    
    return norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar 
    
# -------------------------------------------------------------------------------
    
def cont_norm(fluxes, ivars, cont):
    """ Continuum-normalize a continuous segment of spectra.
    
    Parameters
    ----------
    fluxes: numpy ndarray
    pixel intensities
    ivars: numpy ndarray
    inverse variances, parallel to fluxes
    contmask: boolean mask True indicates that pixel is continuum
   
    Returns
    -------
    norm_fluxes: numpy ndarray normalized pixel intensities
    norm_ivars: numpy ndarray rescaled inverse variances
    """
    
    norm_fluxes = np.ones(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    bad = cont == 0.
    norm_fluxes = np.ones(fluxes.shape)
    norm_fluxes[~bad] = fluxes[~bad] / cont[~bad]
    norm_ivars = cont**2 * ivars
    return norm_fluxes, norm_ivars
        
# -------------------------------------------------------------------------------
# function for microturbulence
# -------------------------------------------------------------------------------  
        
def vmic_vmac(teff, logg, feh):
    
    if logg <= 4.2 or teff >= 5500.:
        ai1, ai2, ai3 = 1.1, 1.e-4, 4.e-7
        aa1, ba1, ca1 = 1.5, 0, 1.e-6
    if logg > 4.2 and teff < 5500:
        ai1, ai2, ai3 = 1.1, 1.6e-4, 0
        aa1, ba1, ca1 = 1.5, .2e-3, 0   
        
    teff0 = 5500.
    # logg0 = 4.
    
    vmic = ai1 + ai2 * (teff-teff0) + ai3 * (teff-teff0)**2
    vmac = 3. * (aa1 + ba1 * (teff-teff0) + ca1 * (teff-teff0)**2)
    
    return vmic, vmac
        
# -------------------------------------------------------------------------------
# creating file to download apogee spectra
# -------------------------------------------------------------------------------    
        
def DownloadData(table):

#    all_plates = []
#    all_ids = []
#    
#    for i in range(len(table)):
#        plate = table[i]['col1']
#        index = table[i]['col2']
#        all_plates.append(plate)
#        all_ids.append(index)

    all_ids = table['APOGEE_ID']
    all_plates = table['LOCATION_ID']
    
    f = open('apogee_data/download_apogee_application2.txt', 'w')    
    for i in range(len(table)):
        if all_plates[i] > 1: 
            print >>f, 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/' + str(all_plates[i]) + '/aspcapStar-r6-l30e.2-' + all_ids[i] + '.fits'
        else:
            print >>f, 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/apo1m/hip/apStar-r6-' + all_ids[i] + '.fits'
    f.close()    

    return 
    
def DownloadDataJohanna(indices, plates):

    all_plates = []
    all_ids = []
    
    for i in range(len(plates)):
        plate = int(plates[i])
        index = indices[i]
        all_plates.append(plate)
        all_ids.append(index)
    
    f = open('apogee_data/download_apogee_johanna.txt', 'w')    
    for i in range(len(plates)):
        if all_plates[i] > 1:
            print 'here!'
            print >>f, 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/' + str(all_plates[i]) + '/aspcapStar-r6-l30e.2-' + all_ids[i] + '.fits'
        else:
            print >>f, 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/hip/aspcapStar-r6-l30e.2-' + all_ids[i] + '.fits'
    f.close()    

    return 

def DownloadDataDiane(table):
    
    f = open('apogee_data/download_apogee_diane.txt', 'w')    
    for i in range(len(table)):
        print >>f, 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/apo1m/hip/apStar-r6-' + table['2MASS'][i].strip() + '.fits'
    f.close()    

    return

    
# -------------------------------------------------------------------------------
# creating file to download apogee spectra for Kepler data
# -------------------------------------------------------------------------------    
        
def DownloadDataKepler(kepler_data):

    all_plates = []
    all_ids = []
    
    for i in range(len(kepler_data)):
        kepler_loc = kepler_data[i]['LOC_ID'] # plate?
        kepler_ids = kepler_data[i]['2MASS_ID']
        all_plates.append(kepler_loc)
        all_ids.append(kepler_ids)
    
    f = open('apogee_data/download_apogee_kepler.txt', 'w')    
    for i in range(len(kepler_data)):
        print >>f, 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/' + all_plates[i] + '/aspcapStar-r6-l30e.2-' + all_ids[i] + '.fits'
    f.close()    

    return

# -------------------------------------------------------------------------------
# Compiling the necessary labels in a new file
# -------------------------------------------------------------------------------        
        
def CompileLabels(data_labels, table, ids): 

    xx = data_labels[1].data   
    
    arr = np.zeros((len(table)*10)).reshape(len(table), 10)
    xx_new = Table(arr, names=('TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR', 'FE_H', 'FE_H_ERR', 'ALPHA_M', 'ALPHA_M_ERR', 'SNR', 'LOCATION_ID')) 
    
    id_str = Column(ids, name='ID')
    xx_new.add_column(id_str, index=0)
    
    #clusters = [[] for i in range(len(ids))]
    
    for i in range(len(table)): 
        xx_subset = xx[np.where(ids[i] == xx['APOGEE_ID'])]
        if len(xx_subset) == 0:
            print('object not found! {0}, {1}'.format(i, ids[i]))
        else: 
            if len(xx_subset) > 1:
                xx_subset = xx_subset[xx_subset['COMMISS'] == 0]
            print i, str(ids[i]), xx_subset['LOCATION_ID']
            xx_new[i] = (ids[i], xx_subset['TEFF'][0], xx_subset['TEFF_ERR'][0], xx_subset['LOGG'][0], xx_subset['LOGG_ERR'][0], xx_subset['FE_H'][0], xx_subset['FE_H_ERR'][0], 
                        xx_subset['ALPHA_M'][0], xx_subset['ALPHA_M_ERR'][0], xx_subset['SNR'][0], xx_subset['LOCATION_ID'][0])
            #clusters[i] = xx_subset['FIELD'][0]
                        
#    cluster_str = Column(clusters, name='CLUSTER') 
#    xx_new.add_column(cluster_str, index=10)
                        
    # needs to be saved as np.array not as table!
    all_labels = []
    for i in range(1, len(xx_new[0])):
        all_labels.append(np.array(xx_new.columns.values()[i]))
    all_labels = np.array(all_labels)
    
    f = open('training_set/train_labels_johanna_new.pickle', 'w')
    pickle.dump(all_labels.T, f)
    f.close()
    
#    f = open('training_set/clusters.pickle', 'w')
#    pickle.dump(clusters, f)
#    f.close()
    
    return 
    
# -------------------------------------------------------------------------------
# Compiling the necessary labels in a new file for 19 labels!
# -------------------------------------------------------------------------------        
        
def CompileLabels19(data_labels, table, ids): 

    xx = data_labels[1].data   
    
    arr = np.zeros((len(table)*47)).reshape(len(table), 47)
    xx_new = Table(arr, names=('TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR', 'FE_H', 'FE_H_ERR', 'ALPHA_M', 'ALPHA_M_ERR', 
                               'C_FE', 'C_FE_ERR', 'N_FE', 'N_FE_ERR', 'O_FE', 'O_FE_ERR', 'NA_FE', 'NA_FE_ERR', 
                               'MG_FE', 'MG_FE_ERR', 'AL_FE', 'AL_FE_ERR', 'SI_FE', 'SI_FE_ERR', 'S_FE', 'S_FE_ERR', 
                               'K_FE', 'K_FE_ERR', 'CA_FE', 'CA_FE_ERR', 'TI_FE', 'TI_FE_ERR', 'V_FE', 'V_FE_ERR', 
                               'MN_FE', 'MN_FE_ERR', 'NI_FE', 'NI_FE_ERR', 'P_FE', 'P_FE_ERR', 'CR_FE', 'CR_FE_ERR', 
                               'CO_FE', 'CO_FE_ERR', 'CU_FE', 'CU_FE_ERR', 'RP_FE', 'RB_FE_ERR', 'SNR')) 
    
    id_str = Column(ids, name='ID')
    xx_new.add_column(id_str, index=0)
    
    clusters = [[] for i in range(len(ids))]
    
    for i in range(len(table)): 
        xx_subset = xx[np.where(ids[i] == xx['APOGEE_ID'])]
        if len(xx_subset) == 0:
            print('object not found! {0}, {1}'.format(i, ids[i]))
        else: 
            if len(xx_subset) > 1:
                xx_subset = xx_subset[xx_subset['COMMISS'] == 0]
            xx_new[i] = (ids[i], xx_subset['TEFF'], xx_subset['TEFF_ERR'], xx_subset['LOGG'], xx_subset['LOGG_ERR'], xx_subset['FE_H'], xx_subset['FE_H_ERR'], 
                        xx_subset['ALPHA_M'], xx_subset['ALPHA_M_ERR'], xx_subset['C_FE'], xx_subset['C_FE_ERR'], xx_subset['N_FE'], xx_subset['N_FE_ERR'], 
                        xx_subset['O_FE'], xx_subset['O_FE_ERR'], xx_subset['NA_FE'], xx_subset['NA_FE_ERR'], xx_subset['MG_FE'], xx_subset['MG_FE_ERR'], 
                        xx_subset['AL_FE'], xx_subset['AL_FE_ERR'], xx_subset['SI_FE'], xx_subset['SI_FE_ERR'], xx_subset['S_FE'], xx_subset['S_FE_ERR'], 
                        xx_subset['K_FE'], xx_subset['K_FE_ERR'], xx_subset['CA_FE'], xx_subset['CA_FE_ERR'], xx_subset['TI_FE'], xx_subset['TI_FE_ERR'], 
                        xx_subset['V_FE'], xx_subset['V_FE_ERR'], xx_subset['MN_FE'], xx_subset['MN_FE_ERR'], xx_subset['NI_FE'], xx_subset['NI_FE_ERR'], 
                        xx_subset['P_FE'], xx_subset['P_FE_ERR'], xx_subset['CR_FE'], xx_subset['CR_FE_ERR'], xx_subset['CO_FE'], xx_subset['CO_FE_ERR'], 
                        xx_subset['CU_FE'], xx_subset['CU_FE_ERR'], xx_subset['RB_FE'], xx_subset['RB_FE_ERR'], xx_subset['SNR'])
            clusters[i] = xx_subset['FIELD'][0]
                        
#    cluster_str = Column(clusters, name='CLUSTER') 
#    xx_new.add_column(cluster_str, index=10)
                        
    # needs to be saved as np.array not as table!
    all_labels = []
    for i in range(1, len(xx_new[0])):
        all_labels.append(np.array(xx_new.columns.values()[i]))
    all_labels = np.array(all_labels)
    
    f = open('training_set/train_labels_19.pickle', 'w')
    pickle.dump(all_labels.T, f)
    f.close()
    
#    f = open('training_set/clusters.pickle', 'w')
#    pickle.dump(clusters, f)
#    f.close()
    
    return 
    
# -------------------------------------------------------------------------------
# Compiling the necessary labels in a new file with Kepler data
# -------------------------------------------------------------------------------        
        
def CompileLabelsKepler(data_labels, table, kepler_data, ids): 

    xx = data_labels[1].data   
    
    arr = np.zeros((len(ids)*10)).reshape(len(ids), 10)
    xx_new = Table(arr, names=('TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR', 'FE_H', 'FE_H_ERR', 'ALPHA_M', 'ALPHA_M_ERR', 'SNR', 'KEPLER')) 
    
    id_str = Column(ids, name='ID')
    xx_new.add_column(id_str, index=0)
        
    # first cluster stars...
    for i in range(len(table)): 
        xx_subset = xx[np.where(ids[i] == xx['APOGEE_ID'])]
        if len(xx_subset) == 0:
            print('object not found! {0}, {1}'.format(i, ids[i]))
        else: 
            if len(xx_subset) > 1:
                xx_subset = xx_subset[xx_subset['COMMISS'] == 0]
            print i, ids[i], xx_subset['TEFF'], xx_subset['TEFF_ERR'], xx_subset['LOGG'], xx_subset['LOGG_ERR'], xx_subset['M_H'], xx_subset['M_H_ERR'], xx_subset['ALPHA_M'], xx_subset['ALPHA_M_ERR'], xx_subset['SNR']
            xx_new[i] = (ids[i], xx_subset['TEFF'], xx_subset['TEFF_ERR'], xx_subset['LOGG'], xx_subset['LOGG_ERR'], xx_subset['M_H'], xx_subset['M_H_ERR'], 
                        xx_subset['ALPHA_M'], xx_subset['ALPHA_M_ERR'], xx_subset['SNR'], 0)

    # ... then Kepler stars    
    for i in range(len(table), len(table)+len(kepler_data)): 
        
        xx_subset = xx[np.where(ids[i] == xx['APOGEE_ID'])]
        kepler_index = np.where(ids[i] == kepler_data['2MASS_ID'])[0][0]
        
        if len(xx_subset) == 0:
            print('object not found! {0}, {1}'.format(i, ids[i]))
        else: 
            if len(xx_subset) > 1:
                xx_subset = xx_subset[xx_subset['COMMISS'] == 0]
            if len(xx_subset) > 1: # for some reason some stars still have two entries...
                xx_subset = xx_subset[0]
            print i, ids[i], xx_subset['TEFF'], xx_subset['TEFF_ERR'], kepler_data[kepler_index]['LOGG_VSA_SYD'], 0.5*(kepler_data[kepler_index]['LOGG_VSA_SYD_PERR']+kepler_data[kepler_index]['LOGG_VSA_SYD_MERR']), xx_subset['FE_H'], xx_subset['FE_H_ERR'], xx_subset['ALPHA_M'], xx_subset['ALPHA_M_ERR'], xx_subset['SNR']
#            xx_new[i] = (ids[i], xx_subset['TEFF'], xx_subset['TEFF_ERR'], kepler_data[kepler_index]['LOGG_VSA_SYD'], 0.5*(kepler_data[kepler_index]['LOGG_VSA_SYD_PERR']+kepler_data[kepler_index]['LOGG_VSA_SYD_MERR']), xx_subset['FE_H'], xx_subset['FE_H_ERR'], 
#                        xx_subset['ALPHA_M'], xx_subset['ALPHA_M_ERR'], xx_subset['SNR'], 1)  
            xx_new[i] = (ids[i], xx_subset['TEFF'], xx_subset['TEFF_ERR'], xx_subset['LOGG'], xx_subset['LOGG_ERR'], xx_subset['FE_H'], xx_subset['FE_H_ERR'], 
                        xx_subset['ALPHA_M'], xx_subset['ALPHA_M_ERR'], xx_subset['SNR'], 1)  
                        
    # needs to be saved as np.array not as table!
    all_labels = []
    for i in range(1, len(xx_new[0])):
        all_labels.append(np.array(xx_new.columns.values()[i]))
    all_labels = np.array(all_labels)
    
    f = open('training_set/train_labels_kepler_apogeedata.pickle', 'w')
    pickle.dump(all_labels.T, f)
    f.close()
    
    return 

# -------------------------------------------------------------------------------
# Compiling the necessary labels in a new file with APOGEE data for Diane's sample
# -------------------------------------------------------------------------------        
        
def CompileLabelsDiane(data_labels, table): #, kepler_data): 
    
    xx = data_labels[1].data 
    kmag = []
    e_kmag = []
    snr = []
    ind_not_found = []
    age = []
    e_age = []
                    
    for i in range(len(table)): 
        xx_subset = xx[np.where(table['2MASS'][i].strip() == xx['APOGEE_ID'].strip())]
        if len(xx_subset) == 0:
            print('object not found! {0}, {1}'.format(i, table['2MASS'][i]))
            ind_not_found.append(i)
        else: 
            if len(xx_subset) > 1:
                xx_subset = xx_subset[xx_subset['COMMISS'] == 0]
            kmag.append(xx_subset['K'][0])
            e_kmag.append(xx_subset['K_ERR'][0])
            snr.append(xx_subset['SNR'][0])

#            kepler_id = np.where(table['2MASS'][i].strip() == kepler_data['2MASS_ID'])       
#            if len(kepler_id) == 0:
#                print('object not found! {0}, {1}'.format(i, table['2MASS'][i]))
#                ind_not_found.append(i)
#            else: 
#                age.append(kepler_data[kepler_id]['AGE_VSA_SYD'])
#                e_age.append(0.5*(kepler_data[kepler_id]['AGE_VSA_SYD_PERR']+kepler_data[kepler_id]['AGE_VSA_SYD_MERR']))
    
    table.remove_rows(ind_not_found)
                
    k = Column(np.array(kmag), name='kmag')
    e_k = Column(np.array(e_kmag), name='e_kmag')
    age = Column(np.array(age), name='age')
    e_age = Column(np.array(e_age), name='e_age')
    sn = Column(np.array(snr), name='snr', dtype=float)    
    table.add_column(k, index=10)
    table.add_column(e_k, index=11)
    table.add_column(age, index=12)
    table.add_column(e_age, index=13)
    table.add_column(sn, index=14)
                
    ascii.write(table, 'training_set/train_labels_diane.txt')                
    
    return 
    
# -------------------------------------------------------------------------------
# Compiling the necessary labels in a new file with APOGEE data for Diane's sample and the cluster stars
# -------------------------------------------------------------------------------        
        
def CompileLabelsDianeCluster(data_labels, table): 
    
    xx = data_labels[1].data 
    kmag = []
    e_kmag = []
    snr = []
    ind_not_found = []
    #age = []
    #e_age = []
    c = []
    n = []
    e_c = []
    e_n = []
                    
    for i in range(len(table)): 
        xx_subset = xx[np.where(table['2MASS'][i].strip() == xx['APOGEE_ID'].strip())]
        if len(xx_subset) == 0:
            print('object not found! {0}, {1}'.format(i, table['2MASS'][i]))
            ind_not_found.append(i)
        else: 
            if len(xx_subset) > 1:
                xx_subset = xx_subset[xx_subset['COMMISS'] == 0]
            kmag.append(xx_subset['K'][0])
            e_kmag.append(xx_subset['K_ERR'][0])
            snr.append(xx_subset['SNR'][0])
            c.append(xx_subset['C_FE'][0])
            e_c.append(xx_subset['C_FE_ERR'][0])
            n.append(xx_subset['N_FE'][0])
            e_n.append(xx_subset['N_FE_ERR'][0])
            if table.mask['teff'][i] == True:
                table['teff'][i] = xx_subset['TEFF']
                table['e_teff'][i] = xx_subset['TEFF_ERR']
                table['logg'][i] = xx_subset['LOGG']
                table['e_logg'][i] = xx_subset['LOGG_ERR']
                table['feh'][i] = xx_subset['FE_H']
                table['e_feh'][i] = xx_subset['FE_H_ERR']
                table['afe'][i] = xx_subset['ALPHA_M']
                table['e_afe'][i] = xx_subset['ALPHA_M_ERR']
                table['par'][i] = 0.0
                table['e_par'][i] = 1000.
    
    table.remove_rows(ind_not_found)
                
    k = Column(np.array(kmag), name='kmag', dtype=float)
    e_k = Column(np.array(e_kmag), name='e_kmag', dtype=float)
    #age = Column(np.array(age), name='age')
    #e_age = Column(np.array(e_age), name='e_age')
    cfe = Column(np.array(c), name='cfe', dtype=float)
    e_cfe = Column(np.array(e_c), name='e_cfe', dtype=float)
    nfe = Column(np.array(n), name='nfe', dtype=float)
    e_nfe = Column(np.array(e_n), name='e_nfe', dtype=float)
    sn = Column(np.array(snr), name='snr', dtype=float)    
    table.add_column(k, index=9)
    table.add_column(e_k, index=10)
    #table.add_column(age, index=11)
    #table.add_column(e_age, index=12)
    table.add_column(cfe, index=11)
    table.add_column(e_cfe, index=12)
    table.add_column(nfe, index=13)
    table.add_column(e_nfe, index=14)
    table.add_column(sn, index=15)
                
    ascii.write(table, 'training_set/train_labels_diane_cluster.txt')                
    
    return 
            
# -------------------------------------------------------------------------------
# Compiling all spectra in a new file and normalizing the spectra
# -------------------------------------------------------------------------------        
        
def LoadDataAndNormalize(ids, kepler, diane):
    
    length = len(ids)

    all_flux = np.zeros((length, 8575))
    all_sigma = np.zeros((length, 8575))
    all_wave = np.zeros((length, 8575))
    
    # read in APOGEE spectra
    for i in range(len(ids)): 
        
        print('Loading spectra {}...'.format(i))
        if diane == True: 
            hdulist = fits.open('apogee_data/apStar-r6-{0}.fits'.format(str(ids[i])))
        else:
            hdulist = fits.open('apogee_data/aspcapStar-r6-l30e.2-{0}.fits'.format(str(ids[i])))        
        flux = hdulist[1].data
        sigma = hdulist[2].data
        header = hdulist[1].header
        start_wl = header['CRVAL1']
        diff_wl = header['CDELT1']
        val = diff_wl * (len(flux)) + start_wl
        wl_full_log = np.arange(start_wl, val, diff_wl)
        wl_full = 10**wl_full_log
        all_wave[i] = wl_full        
        all_flux[i] = flux
        all_sigma[i] = sigma
        
    data = np.array([all_wave, all_flux, all_sigma])
    
    data_norm, continuum = NormalizeData(data.T)
    
    if kepler == True:
        f = open('training_set/all_data_norm_kepler_f.pickle', 'w')
    else:        
        f = open('training_set/all_data_norm_application.pickle', 'w')
    pickle.dump(data_norm, f)
    f.close()     
    
    return data_norm, continuum
    
# -------------------------------------------------------------------------------
# Compiling all spectra in a new file and normalizing the spectra
# -------------------------------------------------------------------------------        
        
def LoadDataAndNormalizeNew(ids):
    
    length = len(ids)

    all_flux = np.zeros((length, 8575))
    all_sigma = np.zeros((length, 8575))
    all_wave = np.zeros((length, 8575))
    
    ind_not_found = np.ones((len(ids),), dtype = bool)
    xx = []
    
    # read in APOGEE spectra
    for i in range(len(ids)): 
        
        print('Loading spectra {}...'.format(i))
        try:
            hdulist = fits.open('apogee_data/aspcapStar-r6-l30e.2-{0}.fits'.format(str(ids[i])))    
        except IOError as e:
            if e.errno == errno.ENOENT:
                try:
                    hdulist = fits.open('apogee_data/apStar-r6-{0}.fits'.format(str(ids[i])))
                except IOError as e:
                    if e.errno == errno.ENOENT:
                        ind_not_found[i] = False
                        xx.append(i)
                        print('{} NOT FOUND!!'.format(i))
        flux = hdulist[1].data
        sigma = hdulist[2].data
        if len(flux) < 8000:
            flux = flux[0]
            sigma = sigma[0]
        foo = np.median(flux)
        if foo > 1.5: # MAGIC HACK HOGG
            flux /= foo
            sigma /= foo
        header = hdulist[1].header
        start_wl = header['CRVAL1']
        diff_wl = header['CDELT1']
        val = diff_wl * (len(flux)) + start_wl
        wl_full_log = np.arange(start_wl, val, diff_wl)
        wl_full = 10**wl_full_log
        all_wave[i] = wl_full        
        all_flux[i] = flux
        all_sigma[i] = sigma
        
    data = np.array([all_wave, all_flux, all_sigma])
    data = data[:, ind_not_found, :]
    
    data_norm, continuum = NormalizeData(data.T)
    
    f = open('training_set/all_data_norm_application4.pickle', 'w')
    pickle.dump(data_norm, f)
    f.close()     
    
    return data_norm, continuum, ind_not_found


# -------------------------------------------------------------------------------
# Compiling all spectra in a new file and normalizing the spectra (Diane's data)
# -------------------------------------------------------------------------------        
        
def LoadDataAndNormalizeDiane(ids, diane):
    
    length = len(ids)

    all_flux = np.zeros((length, 8575))
    all_sigma = np.zeros((length, 8575))
    all_wave = np.zeros((length, 8575))
    
    # read in APOGEE spectra
    for i in range(len(ids)): 
        
        print('Loading spectra {}...'.format(i))
        if diane == True: 
            hdulist = fits.open('apogee_data/apStar-r6-{0}.fits'.format(str(ids[i])))
        else:
            hdulist = fits.open('apogee_data/aspcapStar-r6-l30e.2-{0}.fits'.format(str(ids[i]))) 
        if len(hdulist[1].data) < 8575: 
            flux = hdulist[1].data[0]
            sigma = hdulist[2].data[0]
            print('something is weird...', i, ids[i])
        else:
            flux = hdulist[1].data
            sigma = hdulist[2].data
        header = hdulist[1].header
        start_wl = header['CRVAL1']
        diff_wl = header['CDELT1']
        val = diff_wl * (len(flux)) + start_wl
        wl_full_log = np.arange(start_wl, val, diff_wl)
        wl_full = [10**aval for aval in wl_full_log]
        all_wave[i] = wl_full        
        all_flux[i] = flux
        all_sigma[i] = sigma
        
    data = np.array([all_wave, all_flux, all_sigma])
    
    data_norm, continuum = NormalizeData(data.T)
    
    f = open('training_set/all_data_norm_diane.pickle', 'w')
    pickle.dump(data_norm, f)
    f.close()     
    
    return data_norm, continuum
              
# -------------------------------------------------------------------------------
# Normalizing the spectra
# -------------------------------------------------------------------------------       
        
def NormalizeData(dataall):
        
    Nlambda, Nstar, foo = dataall.shape
    
    pixlist = np.loadtxt('training_set/pixtest8_dr13.txt', usecols = (0,), unpack = 1)
    pixlist = map(int, pixlist)
    LARGE  = 1.                                                          # magic LARGE sigma value
   
    continuum = np.zeros((Nlambda, Nstar))
    dataall_flat = np.ones((Nlambda, Nstar, 3))
    for jj in range(Nstar):
        bad_a = np.logical_or(np.isnan(dataall[:, jj, 1]), np.isinf(dataall[:,jj, 1]))
        bad_b = np.logical_or(dataall[:, jj, 2] <= 0., np.isnan(dataall[:, jj, 2]))
        bad = np.logical_or(np.logical_or(bad_a, bad_b), np.isinf(dataall[:, jj, 2]))
        dataall[bad, jj, 1] = 1.
        dataall[bad, jj, 2] = LARGE
        var_array = LARGE**2 + np.zeros(len(dataall)) 
        var_array[pixlist] = 0.000
        
        bad = dataall_flat[bad, jj, 2] > LARGE
        dataall_flat[bad, jj, 1] = 1.
        dataall_flat[bad, jj, 2] = LARGE
        
        take1 = np.logical_and(dataall[:,jj,0] > 15150, dataall[:,jj,0] < 15800)
        take2 = np.logical_and(dataall[:,jj,0] > 15890, dataall[:,jj,0] < 16430)
        take3 = np.logical_and(dataall[:,jj,0] > 16490, dataall[:,jj,0] < 16950)
        ivar = 1. / ((dataall[:, jj, 2] ** 2) + var_array) 
        fit1 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take1,jj,0], y=dataall[take1,jj,1], w=ivar[take1], deg=2) # 2 or 3 is good for all, 2 only a few points better in temp 
        fit2 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take2,jj,0], y=dataall[take2,jj,1], w=ivar[take2], deg=2)
        fit3 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take3,jj,0], y=dataall[take3,jj,1], w=ivar[take3], deg=2)
        continuum[take1, jj] = fit1(dataall[take1, jj, 0])
        continuum[take2, jj] = fit2(dataall[take2, jj, 0])
        continuum[take3, jj] = fit3(dataall[take3, jj, 0])
        dataall_flat[:, jj, 0] = 1.0 * dataall[:, jj, 0]
        dataall_flat[take1, jj, 1] = dataall[take1,jj,1]/fit1(dataall[take1, 0, 0])
        dataall_flat[take2, jj, 1] = dataall[take2,jj,1]/fit2(dataall[take2, 0, 0]) 
        dataall_flat[take3, jj, 1] = dataall[take3,jj,1]/fit3(dataall[take3, 0, 0]) 
        dataall_flat[take1, jj, 2] = dataall[take1,jj,2]/fit1(dataall[take1, 0, 0]) 
        dataall_flat[take2, jj, 2] = dataall[take2,jj,2]/fit2(dataall[take2, 0, 0]) 
        dataall_flat[take3, jj, 2] = dataall[take3,jj,2]/fit3(dataall[take3, 0, 0]) 
        
    for jj in range(Nstar):
        print "continuum_normalize_tcsh working on star", jj
        bad_a = np.logical_not(np.isfinite(dataall_flat[:, jj, 1]))
        bad_a = np.logical_or(bad_a, dataall_flat[:, jj, 2] <= 0.)
        bad_a = np.logical_or(bad_a, np.logical_not(np.isfinite(dataall_flat[:, jj, 2])))
        bad_a = np.logical_or(bad_a, dataall_flat[:, jj, 2] > 1.)                    # magic 1.
        # grow the mask
        bad = np.logical_or(bad_a, np.insert(bad_a, 0, False, 0)[0:-1])
        bad = np.logical_or(bad, np.insert(bad_a, len(bad_a), False)[1:])
        dataall_flat[bad, jj, 1] = 1.
        dataall_flat[bad, jj, 2] = LARGE
            
    return dataall_flat, continuum

# -------------------------------------------------------------------------------
# Calculating chi2
# -------------------------------------------------------------------------------         
    
def CalculateChi2(labels, fluxes, ivars, coeffs, scatters, train_labels):
    
    nlabels = labels.shape[1]
    nstars = labels.shape[0]
    
    pivots, scales = get_pivots_and_scales(train_labels)
    
    #print pivots, scales
    
    # specialized to second-order model
    zero_term = np.ones((nstars, 1))
    linear_terms = (labels - pivots[None, :]) / scales[None, :]
    quadratic_terms = np.array([np.outer(m, m)[np.triu_indices(nlabels)] for m in (linear_terms)])
    lvec = np.hstack((zero_term, linear_terms, quadratic_terms))

    model = np.dot(coeffs, lvec.T)
        
    chi2 = np.sum((fluxes - model.T)**2 * ivars/ (1. + ivars * scatters**2), axis = 1)
    all_chi2 = (fluxes - model.T)**2 * ivars/ (1. + ivars * scatters**2)

    return chi2, all_chi2


def get_pivots_and_scales(label_vals):
    '''
    function scales the labels 
    '''
    qs = np.nanpercentile(label_vals, (2.5, 50, 97.5), axis=0)
    pivots = qs[1]
    scales = (qs[2] - qs[0])/4.
    
    return pivots, scales    
     
    
# -------------------------------------------------------------------------------
# xxx
# -------------------------------------------------------------------------------     
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   