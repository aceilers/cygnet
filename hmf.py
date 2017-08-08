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
        print(loo_index)
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
# loading training labels and spectra
# -------------------------------------------------------------------------------

train_labels = Table.read('training_set/train_labels_apogee_tgas2.txt', format='ascii', header_start = 0)

print 'loading normalized spectra...'
f = open('training_set/all_data_norm_application4.pickle', 'r')    
spectra = pickle.load(f)
f.close()

wl = spectra[:, 0, 0]
fluxes = spectra[:, :, 1].T
ivars = (1./(spectra[:, :, 2]**2)).T 

# -------------------------------------------------------------------------------
# latex
# -------------------------------------------------------------------------------

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


