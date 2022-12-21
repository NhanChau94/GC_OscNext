"""
author : N. Chau
Related functions for kde implementaion to build pdf
"""
import sys
import math
import pickle as pkl
import numpy as np
import scipy
from scipy.interpolate import pchip_interpolate
# FFT KDE:
from KDEpy import FFTKDE
from KDEpy.bw_selection import improved_sheather_jones

# sklearn:

# sklearn
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


#############################################################################
# FFT kde
#############################################################################


def kde_FFT(x, x_grid, bandwidth=0.03, weights=None, logscale=None):
    """Kernel Density Estimation with KDEpy using FFTkde"""

    if bandwidth=='ISJnD':
        # Implementation of ISJ in > 1D following discussion here:
        # https://github.com/tommyod/KDEpy/issues/81
        # Assuming each dimension has separate bw values and no cross-term in bw matrix
        # dimension:
        n = x.shape[1]
        print('dimension: {}'.format(n))
        bw_array = np.zeros(n)
        for i in range(n):
            bw_array[i] = improved_sheather_jones(x[:, [i]], weights=weights)
        print('bandwidth: ')
        print(bw_array)
        x_scaled = x/bw_array
        grid_scaled = x_grid/bw_array
        y_scaled = FFTKDE(bw=1).fit(x_scaled, weights=weights).evaluate(grid_scaled)
        # print('dimension of kde output: {}'.format(y_scaled.shape))
        # if do the kde in logscale -> f(x)~f(logx)/x
        # divide the by the variables whose ids decclare in the logscale array
        y = y_scaled/np.prod(bw_array)

    elif bandwidth=='scott':
        # dimension:
        d = x.shape[1]
        # number of points:
        n = x.shape[0]
        print('dimension: {}'.format(d))
        print('length of the sample {}'.format(n))
        bw_scott = n**(-1./(d+4))
        print('bandwidth: {}'.format(bw_scott))
        y = FFTKDE(bw=bw_scott, kernel='gaussian').fit(x, weights=weights).evaluate(x_grid)

    elif bandwidth=='ISJ':
        # 1 bandwidth for all dimensions
        n = x.shape[1]
        xdata = x.T.flatten()
        if weights is not None:
            w = np.array([])
            for i in range(n):
                w = np.append(w, weights)
            bw = improved_sheather_jones(np.array([xdata]).T, weights=w)
        else:
            bw = improved_sheather_jones(np.array([xdata]).T)
        print('bandwidth: {}'.format(bw))
        y = FFTKDE(bw=bw, kernel='gaussian').fit(x, weights=weights).evaluate(x_grid)

    else:    
        y = FFTKDE(bw=bandwidth, kernel='gaussian').fit(x, weights=weights).evaluate(x_grid)

    return y


# Mirror data at boundary i.e reflection
# bound: dictionary {dimension that bounded: bound value}
def MirroringData(data, bound):
    mirrordata = np.zeros(data.shape)
    for i in range(data.shape[0]):
        if i in bound:
            mirrordata[i] = 2*bound[i] - data[i]       
        else:
            mirrordata[i] = data[i]       
    return np.concatenate((data, mirrordata), axis=1)


# -----------------------------------------------------------
## FFT kde only applied when data point within the evaluation grids
## Thus sometimes I need to extend the grid to more points for each dimension so that it contains all the data
## But these points might not be used later!
def Extend_EvalPoints(E_true, E_reco, maxEtrue, maxEreco, psi_true, psi_reco):
    # E true
    Etrue_width = E_true[1] - E_true[0]
    while E_true[-1]<maxEtrue:
        E_true = np.append(E_true, E_true[-1]+Etrue_width)
    while E_true[0]>0:    
        E_true = np.append(E_true[0]-Etrue_width, E_true)
        

    logEreco_width = np.log10(E_reco[1]) - np.log10(E_reco[0])
    while E_reco[-1]<maxEreco:
        E_reco = np.append(E_reco, pow(10, np.log10(E_reco[-1])+logEreco_width))
    while E_reco[0]>0.99:    
        E_reco = np.append(pow(10, np.log10(E_reco[0])-logEreco_width), E_reco)
    

    psitrue_width = psi_true[1] - psi_true[0]
    while psi_true[-1]<180.:
        psi_true = np.append(psi_true, psi_true[-1]+psitrue_width)
    while psi_true[0]>0:    
        psi_true = np.append(psi_true[0]-psitrue_width, psi_true)

    psireco_width = psi_reco[1] - psi_reco[0]
    while psi_reco[-1]<180.:
        psi_reco = np.append(psi_reco, psi_reco[-1]+psireco_width)
    while psi_reco[0]>0:    
        psi_reco = np.append(psi_reco[0]-psireco_width, psi_reco)

    return E_true, E_reco, psi_true, psi_reco




#############################################################################
# sklearn kde
#############################################################################
def kde_sklearn(x, x_grid, bandwidth=0.03, weight=None, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x, sample_weight=weight)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid)
    return np.exp(log_pdf)