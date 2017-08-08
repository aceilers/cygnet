# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:08:44 2017

@author: eilers
"""

"""
This file is part of the cygnet project.
Copyright 2017 Anna-Christina Eilers.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import seaborn as sns
import pickle
from datetime import datetime
from astropy.table import Column


from functions_cannon import DownloadData, CompileLabelsDianeCluster, LoadDataAndNormalizeNew, DownloadDataKepler, CompileLabelsKepler, get_pivots_and_scales, DownloadDataJohanna

from matplotlib import rc
rc('text', usetex=False)
rc('font', family='serif')

lsize = 14
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", "orange"]
colors = sns.xkcd_palette(colors)
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'axes.edgecolor':'black', 'xtick.direction': 'in', 'ytick.direction': 'in'})
sns.set_style("ticks")
matplotlib.rcParams['ytick.labelsize'] = lsize
matplotlib.rcParams['xtick.labelsize'] = lsize

# -------------------------------------------------------------------------------
# functions
# -------------------------------------------------------------------------------

def Plot1to1(expectations, tr_label_input_new, K, labels, Nlabels):
    
    for i, l in enumerate(labels):
        orig = tr_label_input_new[:, i]
        cannon = expectations[:, i]
        scatter = np.round(np.std(orig-cannon), 5)
        bias = np.round(np.mean(orig-cannon), 5)    
        xx = [-10000, 10000]
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(orig, cannon, color=colors[-2], label=' bias = {0} \n scatter = {1}'.format(bias, scatter), marker = 'o')
        plt.plot(xx, xx, color=colors[2], linestyle='--')
        plt.xlabel(r'reference labels {}'.format(latex[l]), size=lsize)
        plt.ylabel(r'inferred values {}'.format(latex[l]), size=lsize)
        plt.tick_params(axis=u'both', direction='in', which='both')
        plt.xlim(plot_limits[l])
        plt.ylim(plot_limits[l])
        plt.tight_layout()
        plt.legend(loc=2, fontsize=14, frameon=True)
        plt.title('K = {}'.format(K), fontsize=lsize)
        plt.savefig('hmf/1to1_{0}_{1}.pdf'.format(l, K))
        plt.close() 

def cross_validate(data, ivar, K):
    N, D = data.shape
    expectations = np.zeros((N, D))
    for loo_index in range(N):
        expectations[loo_index, :] = validate(data, ivar, K, loo_index)
    return expectations

def validate(data, ivar, K, loo_index):
    N, D = data.shape
    indices = np.arange(N)
    train = indices[indices != loo_index]
    mu, A, G = HMF(data[train, :], ivar[train, :], K)
    expectation = HMF_test(mu, G, (data[loo_index, :])[None, :], (ivar[loo_index, :])[None, :])
    return expectation 
    
def HMF_test(mu, G, newdata, newivar):
    N, D = newdata.shape
    assert newivar.shape == (N, D)
    data = newdata - mu[None, :]
    A = HMF_astep(data, newivar, G, regularize=False)
    return mu + np.dot(A, G)

def HMF(inputdata, ivar, K):
    """
    Bugs:
        - not commented
        - should be part of a class/object
    """
    
    N, D = inputdata.shape
    assert ivar.shape == (N, D)
    tiny = 1e-6 # magic

    # don't infer the mean; just hard-estimate it and subtract
    mu = HMF_mean(inputdata, ivar)
    data = inputdata - mu[None, :]
    A, G = HMF_initialize(data, ivar, K)
    
    chisq = np.Inf
    converged = False
    while not converged:
        
        print(chisq)
        
        A = HMF_astep(data, ivar, G)
        G = HMF_gstep(data, ivar, A)

        cc = HMF_chisq(data, ivar, A, G)
        if cc > chisq:
            raise ValueError("chi-squared got worse!")
        if cc > chisq - tiny: # dumb
            converged = True
        chisq = cc
        
    return mu, A, G

def HMF_mean(data, ivar):
    return np.sum(ivar * data, axis=0) / np.sum(ivar, axis=0)

def HMF_initialize(data, ivar, K):    
    u, s, v = np.linalg.svd(data, full_matrices = False)
    A = u[:, :K]
    G = v[:K, :]    
    return A, G

def HMF_astep(data, ivar, G, regularize=True):
    """
    svd trick needs to be checked!
    """
    N, D = data.shape
    K, DD = G.shape
    assert D == DD    
    A = np.zeros((N, K))
    
    for i in range(N):
        G_matrix = np.dot(G, (ivar[i, :])[:, None] * G.T)
        F_vector = np.dot(G, ivar[i,:] * data[i,:])
        A[i, :] = np.linalg.solve(G_matrix, F_vector)

    if regularize:
        u, s, v = np.linalg.svd(A, full_matrices = False)
        A = np.dot(u, v)
    
    return A

def HMF_gstep(data, ivar, A):
    """
    currently not regularizing this, ought to have degeneracies broken!
    """
    N, D = data.shape
    NN, K = A.shape
    assert N == NN     
    
    G = np.zeros((K, D))
    
    for j in range(D):
        A_matrix = np.dot(A.T, (ivar[:, j])[:, None] * A)
        F_vector = np.dot(A.T, ivar[:, j] * data[:, j])    
        G[:, j] = np.linalg.solve(A_matrix, F_vector)
    
    return G

def HMF_chisq(data, ivar, A, G):    
    resid = data - np.dot(A, G)    
    return np.sum(resid * ivar * resid)


def DataExpectation(mu, A, G, Nlabels):
    
    expected_data = mu[None, :] + np.dot(A, G)
    
    expected_labels = expected_data[:, :Nlabels]
    expected_spectra = expected_data[:, Nlabels:]
    
    return expected_spectra, expected_labels

# -------------------------------------------------------------------------------
# prepare data
# -------------------------------------------------------------------------------

#data = Table.read('training_set/feuillet2016.txt', format='ascii', data_start=0)
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


# -------------------------------------------------------------------------------
# loading training labels and spectra
# -------------------------------------------------------------------------------

train_labels = Table.read('training_set/train_labels_apogee_tgas2.txt', format='ascii', header_start = 0)

#spectra, continua, ind_not_found = LoadDataAndNormalizeNew(train_labels['APOGEE_ID'])

ind_not_found = np.ones((len(train_labels),), dtype = bool)
# ind_not_found[[115, 1479, 1558, 1583, 1879, 2143]] = False
ind_not_found[[324]] = False
train_labels = train_labels[ind_not_found]

print 'loading normalized spectra...'
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

f = open('training_set/all_data_norm_application4.pickle', 'r')    
spectra = pickle.load(f)
f.close()


wl = spectra[:, 0, 0]
fluxes = spectra[:, :, 1].T
ivars = (1./(spectra[:, :, 2]**2)).T 

# -------------------------------------------------------------------------------
# Calulcate absolute Kmag
# -------------------------------------------------------------------------------

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

#plt.errorbar(train_labels['LOGG'], train_labels['KMAG_ABS'], yerr = train_labels['KMAG_ABS_ERR'], xerr = train_labels['LOGG_ERR'], fmt = 'o')
#plt.ylabel(r'$K_{\rm mag, abs}$')
#plt.xlabel('$\log g$')
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.savefig('apogee_tgas/k_vs_logg_giants.pdf')
#plt.close()

# -------------------------------------------------------------------------------
# xxx
# -------------------------------------------------------------------------------

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

input_ids = np.hstack((ids_tgas, input_ids_apogee))

latex = {}
latex["TEFF"] = r"$T_{\rm eff}$"
latex["LOGG"] = r"$\log g$"
latex["FE_H"] = r"$\rm [Fe/H]$"
latex["ALPHA_FE"] = r"$[\alpha/\rm Fe]$"
latex["KMAG_ABS"] = r"$M_K$"
latex["C_FE"] = r"$\rm [C/Fe]$"
latex["N_FE"] = r"$\rm [N/Fe]$"
latex["WL_TEFF"] = r"$\lambda_{T_{\rm eff}}$"
latex["WL_LOGG"] = r"$\lambda_{\log g}$"
latex["WL_FE_H"] = r"$\lambda_{\rm [Fe/H]}$"
latex["WL_ALPHA_FE"] = r"$\lambda_{[\alpha/\rm Fe]}$"
latex["Q"] = r"$Q$"

# -------------------------------------------------------------------------------
# 
# -------------------------------------------------------------------------------

def make_label_input(labels, train_labels, input_ids):
    tr_label_input = np.array([train_labels[x] for x in labels]).T
    tr_label_input = tr_label_input[input_ids, :]
    tr_delta_input = np.array([train_labels[x+'_ERR'] for x in labels]).T
    tr_delta_input = tr_delta_input[input_ids, :]    
    return tr_label_input, tr_delta_input


labels = ['TEFF', 'FE_H', 'LOGG', 'ALPHA_FE', 'Q', 'N_FE', 'C_FE']
latex_labels = [latex[l] for l in labels]
tr_label_input, tr_delta_input = make_label_input(labels, train_labels, input_ids)

tr_label_input_orig = 1.0 * tr_label_input
tr_delta_input_orig = 1.0 * tr_delta_input


## for previously missing labels, last 46 stars without Fe/H
#f = open('apogee_tgas2/more_labels/test_labels_5lab_66diane_50apogee_predicted_new_deriv_old_seed3.pickle', 'r')
#test_labels_prev = pickle.load(f)
#f.close()
#test_labels_prev = test_labels_prev[:, 1:]
#
#for i in range(len(ids_tgas), len(tr_label_input)):
#    tr_label_input[i, -1] = test_labels_prev[i, -1]
#    tr_delta_input[i, -1] = 3 # magic!

ids = train_labels['APOGEE_ID'][input_ids]

fluxes_tot = fluxes[input_ids, :]   #np.vstack((fluxes_johanna, fluxes))[all_ids, :]
ivars_tot = ivars[input_ids, :]     #np.vstack((ivars_johanna, ivars))[all_ids, :]



tr_label_input_new = np.concatenate((tr_label_input, fluxes_tot), axis=1)
tr_ivar_input_new = np.concatenate((1./(tr_delta_input**2), ivars_tot), axis=1)


# -------------------------------------------------------------------------------
# new...
# -------------------------------------------------------------------------------


#data, data_err = PrepareData(fluxes_tot, ivars_tot, tr_label_input, tr_delta_input, mean=True)

data = 1.0 * tr_label_input_new
ivar = 1.0 * tr_ivar_input_new


K = 3

# i: number of objects, j: number of pixels (8575), k: number of basis components
# best guess for g_jk and a_ik? 
aa = datetime.now()
mu, A, G = HMF(data, ivar, K)
bb = datetime.now()
cc = bb - aa
mins = (cc.seconds + cc.microseconds/1000000.)/60.
print('{} mins. '.format(round(mins, 2)))

Nlabels = 7

labels = ['TEFF', 'FE_H', 'LOGG', 'ALPHA_FE', 'Q', 'N_FE', 'C_FE']
plot_limits = {}
plot_limits['TEFF'] = (3000, 7000)
plot_limits['FE_H'] = (-2.5, 1)
plot_limits['LOGG'] = (0, 2.5)
plot_limits['ALPHA_FE'] = (-.2, .6)
plot_limits['Q'] = (0, .5)
plot_limits['N_FE'] = (-.2, .6)
plot_limits['C_FE'] = (-.2, .3)


# expected_data, expected_labels = DataExpectation(mu, A, G, Nlabels)

for i in range(10):
    K += 2
    expectations = cross_validate(data, ivar, K)
    Plot1to1(expectations, tr_label_input_new, K, labels, Nlabels)

# -------------------------------------------------------------------------------'''


