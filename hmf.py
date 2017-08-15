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
import corner

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

def Plot1to1(expectations, tr_label_input_new, tr_ivar_input, K, labels, Nlabels, name):
    
    for i, l in enumerate(labels):
        orig = tr_label_input_new[:, i]
        cannon = expectations[:, i]
        scatter = np.round(np.std(orig-cannon), 5)
        bias = np.round(np.mean(orig-cannon), 5)  
        chi2 = np.round(np.sum((orig-cannon)**2 * tr_ivar_input[:, i]), 2)
        fig = plt.figure(figsize=(7, 6))
        cmap = 'viridis'
        plt.scatter(orig, cannon, c=tr_ivar_input[:, i], label=' bias = {0} \n scatter = {1}'.format(bias, scatter), marker = 'o', cmap = cmap, vmin = np.percentile(tr_ivar_input[:, i], 2.5), vmax = np.percentile(tr_ivar_input[:, i], 97.5))
        plt.plot(plot_limits[l], plot_limits[l], color=colors[2], linestyle='--')
        plt.colorbar()
        plt.xlabel(r'reference labels {}'.format(latex[l]), size=lsize)
        plt.ylabel(r'inferred values {}'.format(latex[l]), size=lsize)
        plt.tick_params(axis=u'both', direction='in', which='both')
        plt.xlim(plot_limits[l])
        plt.ylim(plot_limits[l])
        plt.tight_layout()
        plt.legend(loc=2, fontsize=14, frameon=True)
        plt.title('K = {0}, $\chi^2 = {1}$'.format(K, chi2), fontsize=lsize)
        plt.savefig('plots/1to1_{0}_K{1}_{2}.pdf'.format(l, K, name))
        plt.close() 

def cross_validate(data, ivar, K, d, folds):
    """
    data: [N, D] array of data values
    ivar: [N, D] array of inverse variance values
    K: dimensionality of the latent space
    d: index of the dimension where we CARE ABOUT THE X-VAL
    folds: number of folds to do (as in "k-fold cross-validation")
    """
    N, D = data.shape
    subset = np.random.randint(folds, size=N)
    expectations = np.zeros((N, D))
    for ss in range(folds):
        leave_out = (subset == ss)
        print('starting fold: ', ss, np.sum(leave_out))
        expectations[leave_out, :] = validate(data, ivar, K, d, leave_out)
    return expectations

def validate(data, ivar, K, d, leave_out):
    """
    see `cross_validate()` for input descriptions
    """
    N, D = data.shape
    print(np.sum(~leave_out))
    mu, A, G = HMF_train(data[~leave_out, :], ivar[~leave_out, :], K)
    xx = data[leave_out, :]
    yy = ivar[leave_out, :]
    xx[:, d] = np.median(data[:, d])  # replace the dimension you care about
    yy[:, d] = 0. * ivar[leave_out, d] # silence the dimension you care about
    expectation = HMF_test(mu, G, (xx), (yy))
    return expectation

def HMF_test(mu, G, newdata, ivar):
    """
    This function repeats some functionality in `HMF_astep()`. That sucks, but
    I don't want to mess with `HMF_astep()`
    """
    K, DD = G.shape
    N, D = newdata.shape
    assert D == DD    
    assert ivar.shape == (N, D)
    data = newdata - mu[None, :]
    A = np.zeros((N, K))
    for i in range(N):
        G_matrix = np.dot(G, (ivar[i, :])[:, None] * G.T) + np.eye(K) # prior
        F_vector = np.dot(G, ivar[i, :] * data[i, :])
        A[i, :] = np.linalg.solve(G_matrix, F_vector)
    A = HMF_astep(data, newivar, G)
    return mu + np.dot(A, G)

def HMF_train(inputdata, ivar, K, A = None, G = None):
    """
    Bugs:
        - not commented
        - should be part of a class/object
    """
    
    N, D = inputdata.shape
    assert ivar.shape == (N, D)
    tiny = 1e-5 # magic

    # don't infer the mean; just hard-estimate it and subtract
    mu = HMF_mean(inputdata, ivar)
    data = inputdata - mu[None, :]
    if A is None:
        A, G = HMF_initialize(data, ivar, K)
    
    chisq = np.Inf
    converged = False
    while not converged:
        
        print(chisq)
        
        A = HMF_astep(data, ivar, G)
        G = HMF_gstep(data, ivar, A)

        cc = HMF_chisq(data, ivar, A, G)
        if cc > chisq:
            print("chi-squared got worse!")
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

def HMF_astep(data, ivar, G):
    """
    svd trick needs to be checked!
    """
    N, D = data.shape
    K, DD = G.shape
    assert D == DD    
    A = np.zeros((N, K))
    
    for i in range(N):
        G_matrix = np.dot(G, (ivar[i, :])[:, None] * G.T) + np.eye(K) # prior
        F_vector = np.dot(G, ivar[i, :] * data[i, :])
        A[i, :] = np.linalg.solve(G_matrix, F_vector)

    # now post-process A to have unit variance
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
    """
    needs to be synchronized with the prior in `HMF_astep()`
    """
    resid = data - np.dot(A, G)
    return np.sum(resid * ivar * resid) + np.sum(A * A)

def DataExpectation(mu, A, G, Nlabels):
    
    expected_data = mu[None, :] + np.dot(A, G)
    
    expected_labels = expected_data[:, :Nlabels]
    expected_spectra = expected_data[:, Nlabels:]
    
    return expected_spectra, expected_labels


# -------------------------------------------------------------------------------
# loading training labels and spectra
# -------------------------------------------------------------------------------

print 'loading training labels...'
f = open('data/training_labels_apogee_hip.pickle', 'r')
training_labels = pickle.load(f)
f.close()

print 'loading normalized spectra...'
f = open('data/apogee_spectra_norm_hip.pickle', 'r')    
spectra = pickle.load(f)
f.close()

wl = spectra[:, 0, 0]
fluxes = spectra[:, :, 1].T
ivars = (1./(spectra[:, :, 2]**2)).T 
        
# -------------------------------------------------------------------------------
# remove duplicates
# -------------------------------------------------------------------------------
        
foo, idx = np.unique(training_labels['APOGEE_ID'], return_index = True)
training_labels = training_labels[idx]
fluxes = fluxes[idx, :]
ivars = ivars[idx, :]
        
# -------------------------------------------------------------------------------
# data masking
# -------------------------------------------------------------------------------
        
masking = training_labels['K'] < 0.
training_labels = training_labels[~masking]
fluxes = fluxes[~masking]
ivars = ivars[~masking]

'''
masking2 = (training_labels['LOGG'] <= 2.2) * \
           (training_labels['LOGG'] >= 0) * \
           (training_labels['TEFF'] > 4300.)
training_labels = training_labels[masking2]
fluxes = fluxes[masking2]
ivars = ivars[masking2]
'''

# -------------------------------------------------------------------------------
# scaling!
# -------------------------------------------------------------------------------
#ivots = np.median(foo, axis=blah)
#scales = np.std(foo, axis=blah) # be careful
#scales[scales <= 0.] = 1. # brittle
#sdata = (data - pivots) / scales
#sivars = ...

# -------------------------------------------------------------------------------
# calculate K_MAG_ABS and Q
# -------------------------------------------------------------------------------

# M = m - 5(log d - 1)
# sigma_M = np.sqrt(sigma_m^2 + sigma_d^2/d^2)
# parallax = 1/d
# parallaxes given in milli arcsec

Q = 10**(0.2*training_labels['K']) * training_labels['parallax']/100.                    # assumes parallaxes is in mas
Q_err = training_labels['parallax_error'] * 10**(0.2*training_labels['K'])/100. 
Q = Column(Q, name = 'Q_MAG')
Q_err = Column(Q_err, name = 'Q_MAG_ERR')
training_labels.add_column(Q, index = 12)
training_labels.add_column(Q_err, index = 13)

# -------------------------------------------------------------------------------
# latex
# -------------------------------------------------------------------------------

latex = {}
latex["TEFF"] = r"$T_{\rm eff}$"
latex["LOGG"] = r"$\log g$"
latex["FE_H"] = r"$\rm [Fe/H]$"
latex["ALPHA_M"] = r"$[\alpha/\rm M]$"
latex["C_FE"] = r"$\rm [C/Fe]$"
latex["N_FE"] = r"$\rm [N/Fe]$"
latex["Q_MAG"] = r"$Q$"

plot_limits = {}
plot_limits['TEFF'] = (3000, 7000)
plot_limits['FE_H'] = (-2.5, 1)
plot_limits['LOGG'] = (0, 2.5)
plot_limits['ALPHA_M'] = (-.2, .6)
plot_limits['Q_MAG'] = (0, .5)
plot_limits['N_FE'] = (-.2, .6)
plot_limits['C_FE'] = (-.2, .3)

# -------------------------------------------------------------------------------
# 
# -------------------------------------------------------------------------------

def make_label_input(labels, training_labels):
    tr_label_input = np.array([training_labels[x] for x in labels]).T
    tr_ivar_input = 1./((np.array([training_labels[x+'_ERR'] for x in labels]).T)**2)
    for x in range(tr_label_input.shape[1]):
        bad = np.logical_or(tr_label_input[:, x] < -100., tr_label_input[:, x] > 9000.) # magic
        tr_label_input[bad, x] = np.median(tr_label_input[:, x])
        tr_ivar_input[bad, x] = 0.
    # remove one outlier in T_eff and [N/Fe]!
    bad = tr_label_input[:, 0] > 5200.
    tr_label_input[bad, 0] = np.median(tr_label_input[:, 0])
    tr_ivar_input[bad, 0] = 0.  
    bad = tr_label_input[:, 5] < -0.6
    tr_label_input[bad, 5] = np.median(tr_label_input[:, 5])
    tr_ivar_input[bad, 5] = 0.     
    return tr_label_input, tr_ivar_input

labels = np.array(['TEFF', 'FE_H', 'LOGG', 'ALPHA_M', 'Q_MAG', 'N_FE', 'C_FE'])
Nlabels = len(labels)
latex_labels = [latex[l] for l in labels]
tr_label_input, tr_ivar_input = make_label_input(labels, training_labels)


# 100 best objects (714)
input_ids = np.arange(len(tr_label_input)) # (tr_ivar_input[:, labels == 'Q_MAG'] > 200.).flatten() # magic
tr_label_input = tr_label_input[input_ids, :]
tr_ivar_input = tr_ivar_input[input_ids, :]  

corner.corner(tr_label_input, labels = latex_labels)
plt.savefig('plots/corner_{}.pdf'.format(len(tr_label_input)))

fluxes = fluxes[input_ids, :]   
ivars = ivars[input_ids, :]     

# -------------------------------------------------------------------------------
# HMF
# -------------------------------------------------------------------------------

nodata = True
if nodata:
    data = 1. * tr_label_input
    ivar = 1. * tr_ivar_input
else:
    data = np.concatenate((tr_label_input, fluxes), axis=1)
    ivar = np.concatenate((tr_ivar_input, ivars), axis=1)

np.random.seed(42)

folds = 5
K = 1 # has to be less than the number of dimensions
name = 'ivartest0_testdatamedian'


N, D = data.shape
noisify = False
if noisify:
    name += '_noise'
    adderr = 0.05
    data[:, 4] += adderr * np.random.normal(size = N)
    ivar[:, 4] = 1. / (1. / ivar[:, 4] + adderr * adderr) # brittle

# cross validation
for i in range(5):
    expectations = cross_validate(data, ivar, K, 4, folds)
    Plot1to1(expectations, data, ivar, K, labels, Nlabels, name)
    f = open('plots/results_data/data_K{0}_{1}.pickle'.format(K, name), 'w')
    pickle.dump(expectations, f)
    f.close()
    K += 1    

# -------------------------------------------------------------------------------'''

