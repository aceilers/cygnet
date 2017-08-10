#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 20:10:52 2017

@author: eilers
"""

import numpy as np
from astropy.table import Table, hstack
import os.path
import subprocess
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import pickle

from functions_cannon import NormalizeData

# -------------------------------------------------------------------------------
# download data
# -------------------------------------------------------------------------------

substr = 'l31c'
subsubstr = '2'
fn = 'allStar-' + substr + '.' + subsubstr + '.fits'
url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/' + substr \
    + '/' + substr + '.' + subsubstr + '/' + fn
destination = './data/' + fn
cmd = 'wget ' + url + ' -O ' + destination
if not os.path.isfile(destination):
    subprocess.call(cmd, shell = True) # warning: security
print("opening " + destination)
apogee_table = fits.open(destination)
apogee_data = apogee_table[1].data                         
apogee_data = apogee_data[apogee_data['DEC'] > -90.0]

 # download TGAS data    
fn = 'stacked_tgas.fits'
url = 'http://s3.adrian.pw/' + fn
destination = './data/'+ fn
cmd = 'wget ' + url + ' -O ' + destination
if not os.path.isfile(destination):
    subprocess.call(cmd, shell = True)
print("opening " + destination)
tgas_table = fits.open(destination)
tgas_data = tgas_table[1].data
                      
                      
# -------------------------------------------------------------------------------
# cut in logg <= 2:
# -------------------------------------------------------------------------------

apogee_data = apogee_data[np.logical_and(apogee_data['LOGG'] <= 2., apogee_data['LOGG'] >= 0)]
                          
# -------------------------------------------------------------------------------
# match TGAS and APOGEE
# -------------------------------------------------------------------------------

apogee_cat = SkyCoord(ra=apogee_data['RA']*u.degree, dec=apogee_data['DEC']*u.degree)
tgas_cat = SkyCoord(ra=tgas_data['RA']*u.degree, dec=tgas_data['DEC']*u.degree)

id_tgas, id_apogee, d2d, d3d = apogee_cat.search_around_sky(tgas_cat, 0.001*u.degree)

tgas_data = tgas_data[id_tgas]
apogee_data = apogee_data[id_apogee]

print('matched entries: {}'.format(len(tgas_data)))

# -------------------------------------------------------------------------------
# get APOGEE spectra
# -------------------------------------------------------------------------------

found = np.ones_like(id_tgas, dtype=bool)

for i, (fn2, loc) in enumerate(zip(apogee_data['FILE'], apogee_data['LOCATION_ID'])):
    fn = fn2.replace('apStar-r8', 'aspcapStar-r8-l31c.2')
    urlbase = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/' + substr \
    + '/' + substr + '.' + subsubstr + '/' + str(loc) + '/'
    url = urlbase + fn
    url2 = urlbase + fn2
    destination = './data/spectra/' + fn
    destination2 = './data/spectra/' + fn2
    if not (os.path.isfile(destination) or os.path.isfile(destination2)):
        try:
            cmd = 'wget ' + url + ' -O ' + destination
            print cmd
            subprocess.call(cmd, shell = True)
        except:
            try:
                cmd = 'wget ' + url2 + ' -O ' + destination2
                print cmd
                subprocess.call(cmd, shell = True)
            except:
                found[i] = False
                     
tgas_data = tgas_data[found]                     
apogee_data = apogee_data[found]

# remove missing files (code above doesn't work...)
found = np.ones_like(id_tgas, dtype=bool)
destination = './data/spectra/'
for i in range(len(apogee_data['FILE'])):
    print i
    entry = apogee_data['FILE'][i]
    entry = entry.replace('apStar-r8', 'aspcapStar-r8-l31c.2')
    try:
        hdulist = fits.open(destination + entry)
    except:
        print("not found!")
        found[i] = False

tgas_data = tgas_data[found]                     
apogee_data = apogee_data[found]    

print('spectra found for: {}'.format(len(tgas_data)))
 
# -------------------------------------------------------------------------------
# normalize spectra: functions
# -------------------------------------------------------------------------------

def LoadAndNormalizeData(file_spectra, file_name, destination):
    
    all_flux = np.zeros((len(file_spectra), 8575))
    all_sigma = np.zeros((len(file_spectra), 8575))
    all_wave = np.zeros((len(file_spectra), 8575))
    
    i=0
    for entry in file_spectra:
        print i
        entry = entry.replace('apStar-r8', 'aspcapStar-r8-l31c.2')
        hdulist = fits.open(destination + entry)
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
        i += 1
        
    data = np.array([all_wave, all_flux, all_sigma])
    data_norm, continuum = NormalizeData(data.T)
    
    f = open('data/' + file_name, 'w')
    pickle.dump(data_norm, f)
    f.close()
    
    return data_norm, continuum

def NormalizeData(dataall):
        
    Nlambda, Nstar, foo = dataall.shape
    
    pixlist = np.loadtxt('data/pixtest8_dr13.txt', usecols = (0,), unpack = 1)
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
# normalize spectra
# -------------------------------------------------------------------------------

file_name = 'apogee_spectra_norm.pickle'
destination = './data/' + file_name
if not os.path.isfile(destination):
    data_norm, continuum = LoadAndNormalizeData(apogee_data['FILE'], file_name, destination = './data/spectra/')

# -------------------------------------------------------------------------------
# save files!
# -------------------------------------------------------------------------------

apogee_data = Table(apogee_data)
tgas_data = Table(tgas_data)
training_labels = hstack([apogee_data, tgas_data])

f = open('data/training_labels_apogee_tgas.pickle', 'w')
pickle.dump(training_labels, f)
f.close()




'''#data = Table.read('training_set/feuillet2016.txt', format='ascii', data_start=0)
#par = Table.read('training_set/twom_par_best.txt', format='ascii', data_start=0)
#hip = Table.read('training_set/twom_par_hip.txt', format='ascii', data_start=0)
#
#data.rename_column('col1', 'APOGEE_ID')
#data.remove_column('col2')
#data.rename_column('col3', 'TEFF')
#data.rename_column('col4', 'TEFF_ERR')
#data.rename_column('col5', 'LOGG')
#data.rename_column('col6', 'LOGG_ERR')
#data.rename_column('col7', 'FE_H')
#data.rename_column('col8', 'FE_H_ERR')
#data.rename_column('col9', 'ALPHA_FE')
#data.rename_column('col10', 'ALPHA_FE_ERR')
#data.remove_column('col11')
#data.remove_column('col12')
#data.remove_column('col13')
#data.remove_column('col14')
#data.remove_column('col15')
#data.remove_column('col16')
#data.remove_column('col17')   # Bayesian Ages from Feuillet et al. 2016
#data.remove_column('col18') # Bayesian Ages from Feuillet et al. 2016
#data.remove_column('col19')
#data.remove_column('col20')
#
#parallax = []
#parallax_err = []
#ind_not_found = []
#for i in range(len(data)):
#    for j in range(len(par)):
#        found = 0
#        if data['APOGEE_ID'][i].strip() == par['col1'][j].strip():
#            parallax.append(par['col2'][j])
#            parallax_err.append(par['col3'][j])
#            found = 1
#            break
#        elif j == len(par)-1 and found == 0:
#            for k in range(len(hip)): 
#                if data['APOGEE_ID'][i].strip() == hip['col1'][k].strip():
#                    parallax.append(par['col2'][j])
#                    parallax_err.append(par['col3'][j])
#                    found = 1
#                    # print(i, 'found in hip!')
#                    break
#                elif k == len(hip)-1 and found == 0: 
#                    print(i, 'NOT FOUND!', data['APOGEE_ID'][i])
#                    ind_not_found.append(i)
#data.remove_rows(ind_not_found)
#
#par = Column(np.array(parallax), name='PAR', dtype = float)
#e_par = Column(np.array(parallax_err) * np.array(parallax), name='PAR_ERR', dtype = float)
#data.add_column(par, index=9)
#data.add_column(e_par, index=10)

# -------------------------------------------------------------------------------
# adding APOGEE data...
# -------------------------------------------------------------------------------

#data_labels = fits.open('training_set/allStar-l30e.2.fits')
#table = data_labels[1].data
#
## -------------------------------------------------------------------------------
## cuts in logg anf SNR
## -------------------------------------------------------------------------------
#
#xx = np.logical_and(table['LOGG'] <= 2., table['LOGG'] >= -10.)
#table_cuts = table[xx]
## make sure measurements for Teff and [Fe/H] exist...
#xx = np.logical_and(table_cuts['FE_H'] >= -9000, table_cuts['TEFF'] > -9000.)
#table_cuts = table_cuts[xx]
## table_cuts = table_cuts[table_cuts['SNR'] >= 500]
#
#ids = Column(np.array(table_cuts['APOGEE_ID']), name = 'APOGEE_ID')
#logg = Column(np.array(table_cuts['LOGG']), name = 'LOGG')
#logg_err = Column(np.array(table_cuts['LOGG_ERR']), name = 'LOGG_ERR')
#teff = Column(np.array(table_cuts['TEFF']), name = 'TEFF')
#teff_err = Column(np.array(table_cuts['TEFF_ERR']), name = 'TEFF_ERR')
#feh = Column(np.array(table_cuts['FE_H']), name = 'FE_H')
#feh_err = Column(np.array(table_cuts['FE_H_ERR']), name = 'FE_H_ERR')
#afe = Column(np.array(table_cuts['ALPHA_M']), name = 'ALPHA_FE')
#afe_err = Column(np.array(table_cuts['ALPHA_M_ERR']), name = 'ALPHA_FE_ERR')
#cfe = Column(np.array(table_cuts['C_FE']), name = 'C_FE')
#cfe_err = Column(np.array(table_cuts['C_FE_ERR']), name = 'C_FE_ERR')
#nfe = Column(np.array(table_cuts['N_FE']), name = 'N_FE')
#nfe_err = Column(np.array(table_cuts['N_FE_ERR']), name = 'N_FE_ERR')
#kmag = Column(np.array(table_cuts['K']), name = 'KMAG')
#kmag_err = Column(np.array(table_cuts['K_ERR']), name = 'KMAG_ERR')
#snr = Column(np.array(table_cuts['SNR']), name = 'SNR')
#
#apogee_selection = Table()
#apogee_selection.add_columns([ids, teff, teff_err, logg, logg_err, feh, feh_err, afe, afe_err, cfe, cfe_err, nfe, nfe_err, kmag, kmag_err, snr])
#
## randomly choose 1000 entries
#x = np.arange(len(apogee_selection))
#np.random.shuffle(x)
#apogee_selection_cut = apogee_selection[x[:1000]]
#table_cuts2 = table_cuts[x[:1000]]
#
## DownloadData(table_cuts2)
#
#data_cuts = data[np.where(data['LOGG'] <= 2.)]
#
#kmag = []
#e_kmag = []
#snr = []
#cfe = []
#e_cfe = []
#nfe = []
#e_nfe = []
#for i in range(len(data_cuts)):
#    xx = table[np.where(data_cuts['APOGEE_ID'][i] == table['APOGEE_ID'])]
#    kmag.append(xx['K'][0])
#    e_kmag.append(xx['K_ERR'][0])
#    snr.append(xx['SNR'][0])
#    cfe.append(xx['C_FE'][0])
#    e_cfe.append(xx['C_FE_ERR'][0])
#    nfe.append(xx['N_FE'][0])
#    e_nfe.append(xx['N_FE_ERR'][0])
#    
#kmag = Column(np.array(kmag), name = 'KMAG', dtype = float)
#e_kmag = Column(np.array(e_kmag), name = 'KMAG_ERR', dtype = float) 
#cfe = Column(np.array(cfe), name = 'C_FE', dtype = float)
#e_cfe = Column(np.array(e_cfe), name = 'C_FE_ERR', dtype = float)
#nfe = Column(np.array(nfe), name = 'N_FE', dtype = float)
#e_nfe = Column(np.array(e_nfe), name = 'N_FE_ERR', dtype = float) 
#snr = Column(np.array(snr), name = 'SNR', dtype = float)  
#data_cuts.add_column(kmag, index = 9)
#data_cuts.add_column(e_kmag, index = 10)
#data_cuts.add_column(snr, index = 11)
#data_cuts.add_column(cfe, index = 11)
#data_cuts.add_column(e_cfe, index = 11)
#data_cuts.add_column(nfe, index = 11)
#data_cuts.add_column(e_nfe, index = 11)
#    
#train_labels = join(data_cuts, apogee_selection_cut, join_type = 'outer')
#
## ascii.write(train_labels, 'training_set/train_labels_apogee_tgas2.txt')


#spectra, continua, ind_not_found = LoadDataAndNormalizeNew(train_labels['APOGEE_ID'])


ind_not_found = np.ones((len(train_labels),), dtype = bool)
# ind_not_found[[115, 1479, 1558, 1583, 1879, 2143]] = False
ind_not_found[[324]] = False
train_labels = train_labels[ind_not_found]


#f = open('training_set/all_data_norm_application1.pickle', 'r')    
#spectra1 = pickle.load(f)
#f.close()
#f = open('training_set/all_data_norm_application2.pickle', 'r')    
#spectra2 = pickle.load(f)
#f.close()
#f = open('training_set/all_data_norm_application3.pickle', 'r')    
#spectra3 = pickle.load(f)
#f.close()
#
#spectra = np.concatenate((spectra1, spectra2, spectra3), axis = 1)

# M = m - 5(log d - 1)
# sigma_M = np.sqrt(sigma_m^2 + sigma_d^2/d^2)
# parallax = 1/d
# parallaxes given in milli arcsec


Q = []
err_Q = []
K_mag_abs = []
err_K_mag_abs = []
for i in range(len(train_labels)):
    if train_labels['PAR'][i] != '--':
        par = float(train_labels['PAR'][i])
        par_err = float(train_labels['PAR_ERR'][i])
        K_mag_abs.append(train_labels['KMAG'][i] - 5. * (np.log10(1000./par) - 1.)) # assumes parallaxes is in mas
        err_K_mag_abs.append(np.sqrt(train_labels['KMAG_ERR'][i]**2 + 25./(np.log(10)**2) * par_err**2 / par**2))
        Q.append(10**(0.2*train_labels['KMAG'][i]) * par/100.) # assumes parallaxes is in mas
        err_Q.append(par_err * 10**(0.2*train_labels['KMAG'][i])/100.)
    else:
        K_mag_abs.append(np.nan)
        err_K_mag_abs.append(np.nan)   
        Q.append(np.nan)
        err_Q.append(np.nan)
K_mag_abs = Column(K_mag_abs, name = 'KMAG_ABS')
err_K_mag_abs = Column(err_K_mag_abs, name = 'KMAG_ABS_ERR')
train_labels.add_column(K_mag_abs, index = 12)
train_labels.add_column(err_K_mag_abs, index = 13)

Q = Column(Q, name = 'Q')
err_Q = Column(err_Q, name = 'Q_ERR')
train_labels.add_column(Q, index = 14)
train_labels.add_column(err_Q, index = 15)



seed = 3
np.random.seed(seed)

all_ids = np.arange(len(train_labels))
ids_tgas = all_ids[np.where(train_labels['KMAG_ABS_ERR'] < 1000.)]

ids_bool = np.ones((len(train_labels),), dtype = bool)
ids_bool[ids_tgas] = False
        
# remove two stars that have no [C_FE] measurement        
ids_bool[(213, 997),] = False
# remove stars that have no [N_FE] measurement        
ids_bool[( 341,  454,  575,  592, 1037), ] = False
        
ids_apogee = all_ids[ids_bool]
np.random.shuffle(ids_apogee)

n_apogee = 0 # out of 3071 available
input_ids_apogee = ids_apogee[:n_apogee]

input_ids = np.hstack((ids_tgas, input_ids_apogee))'''













