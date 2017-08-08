#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 20:10:52 2017

@author: eilers
"""

import numpy as np
from astropy.table import Table
import os.path
import subprocess
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

# -------------------------------------------------------------------------------

""" 
TO DO:
    1. get APOGEE spectra
    2. run Cannon normalization and build input data file
    3. write file
"""

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
for q in range(16):
    fn = 'TgasSource_000-000-{:03d}.fits'.format(q)
    url = 'http://cdn.gea.esac.esa.int/Gaia/tgas_source/fits/' + fn
    destination = './data/' + fn
    cmd = 'wget ' + url + ' -O ' + destination
    if not os.path.isfile(destination):
        subprocess.call(cmd, shell = True)
    print("opening " + destination)
    tgas_table = fits.open(destination)
    if q == 0:
        tgas_data = tgas_table[1].data
    else:
        tgas_data = np.append(tgas_data, tgas_table[1].data)
    
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

for fn, loc in zip(apogee_data['FILE'], apogee_data['LOCATION_ID']):
    url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/' + substr \
    + '/' + substr + '.' + subsubstr + '/' + str(loc) + '/' + fn
    print url
    destination = './data/spectra/' + fn
    cmd = 'wget ' + url + ' -O ' + destination
    if not os.path.isfile(destination):
        subprocess.call(cmd, shell = True)

# -------------------------------------------------------------------------------
# normalize spectra
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# data quality check (remove -9999.9 etc.)
# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------
# calculate K_MAG_ABS and Q
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# save files!
# -------------------------------------------------------------------------------




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













