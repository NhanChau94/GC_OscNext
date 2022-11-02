#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/RHEL_7_x86_64/bin/python

import numpy as np
from optparse import OptionParser

import sys
import pickle as pkl
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DetResponse/")
from Detector import *

# KDE:
from kde.pykde import gaussian_kde
from sklearn.neighbors import KernelDensity


def kde_icecube(x, x_grid, bandwidth=0.03, **kwargs):
    """Kernel Density Estimation with icecube package"""

    if bandwidth == "adaptive":
        adaptive = True
        weight_adaptive_bw = True
        alpha = 0.3
        kde = gaussian_kde(x, **kwargs,adaptive=adaptive,
                                    weight_adaptive_bw=weight_adaptive_bw,alpha=alpha)
    else:
        kde = gaussian_kde(x, **kwargs)
        # for scipy: scale to try to match sklearn in case of scalar
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # divide the bandwidth by the sample standard deviation here.
        # https://stackoverflow.com/questions/21000140/relation-between-2d-kde-bandwidth-in-sklearn-vs-bandwidth-in-scipy
        # if (isinstance(bandwidth, str)==False): 
        #     bandwidth = bandwidth / x.std(ddof = 1)

    kde.set_bandwidth(bandwidth)
    return kde.evaluate(x_grid)

def kde_sklearn(x, x_grid, bandwidth=0.03, weight=0, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x, sample_weight=weight)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid)
    return np.exp(log_pdf)
    # return kde_skl    


def KDE_RespMatrix(MCcut, Bin, bw_method, method="kde", nu_type="nu_e", pid=0):
    # Separate MC by each channel nutype->PID:
    # nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]

    pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}
    PID = [[0.,0.5],[0.5, 0.85],[0.85, 1]]

    Resp = dict()
    print("Computing {} PID bin".format(pid))
    Resp[pid] = dict()
    print("----{}".format(nu_type))

    loc = np.where(  (MCcut["nutype"]==pdg_encoding[nu_type]) & (MCcut["PID"]>=PID[pid][0])
                    & (MCcut["PID"]<PID[pid][1]) )

    #Extract MC events: 
    #NOTE: input psi in deg!
    psitrue = MCcut["psi_true"][loc]
    Etrue = MCcut["E_true"][loc]
    psireco = MCcut["psi_reco"][loc]
    Ereco = MCcut["E_reco"][loc]
    w = MCcut["w"][loc]
        
    psiE_train = np.vstack([np.log(psitrue), np.log(Etrue), np.log(psireco), np.log10(Ereco)])
    
    #Evaluate points:
    ##Equal spacing in the final variables: reco Psi & log10(E), true psi and true E
    trueEeval = Bin["true_energy_center"]
    truePsieval = Bin["true_psi_center"]
    recoEeval = Bin["reco_energy_center"]
    recoPsieval = Bin["reco_psi_center"]

    g_psi_true, g_energy_true, g_psi_reco, g_energy_reco = np.meshgrid(truePsieval, trueEeval,
                                                            recoPsieval, recoEeval)                      
    psi_eval_true = g_psi_true.T.flatten()
    E_eval_true = g_energy_true.T.flatten()
    psi_eval_reco = g_psi_reco.T.flatten()
    E_eval_reco = g_energy_reco.T.flatten()

    ##Evaluate the KDE in log(Psi)-log10E
    psiE_eval = np.vstack([np.log(psi_eval_true), np.log(E_eval_true), 
                        np.log(psi_eval_reco), np.log10(E_eval_reco)])
    print("Evaluating KDE.....")    
    if method=="sklearn":
        kde_w = kde_sklearn(psiE_train.T, psiE_eval.T, bandwidth=bw_method, weight=w)
        #Needs to be divided by evaluation angle
        kde_weight = kde_w.reshape(psi_eval_true.shape)/(psi_eval_true* psi_eval_reco* E_eval_true)
    else:
        kde_w = kde_icecube(psiE_train, psiE_eval, bandwidth=bw_method, weights=w)
        #Needs to be divided by evaluation angle
        kde_weight = kde_w/(psi_eval_true* psi_eval_reco* E_eval_true)

    # Fill into histogram:
    Psitrue_edges = Bin["true_psi_edges"]
    Etrue_edges = Bin["true_energy_edges"]
    Psireco_edges = Bin["reco_psi_edges"]
    Ereco_edges = Bin["reco_energy_edges"]

    H, edges = np.histogramdd((psi_eval_true, E_eval_true, psi_eval_reco, E_eval_reco),
                                bins = (Psitrue_edges, Etrue_edges, Psireco_edges, Ereco_edges),
                                weights=kde_weight)
    Resp[pid][nu_type] = H
    return Resp                 


#----------------------------------------------------------------------------------------------------------------------
#Define parameters needed
#----------------------------------------------------------------------------------------------------------------------

parser = OptionParser()
# i/o options
parser.add_option("-b", "--bw", type = "string", action = "store", default = "None", metavar  = "<Bandwidth>", help = "Bandwith method: scalar, scott, adaptive",)
parser.add_option("-m", "--method", type = "string", action = "store", default = "None", metavar  = "<package>", help = "kde Package to use: sklearn or kde",)
parser.add_option("-p", "--pid", type = int, action = "store", default = 0, metavar  = "<pid>", help = "which pid bin: 0(showers), 1(middles), 2(tracks)",)
parser.add_option("-n", "--nutype", type = "string", action = "store", default = "None", metavar  = "<nutype>", help = "which nutype: nu_e, nu_mu, nu_tau, nu_e_bar, nu_mu_bar, nu_tau_bar",)

(options, args) = parser.parse_args()

bw = options.bw
method = options.method
pid = options.pid
nutype = options.nutype

# Extract the MC:
pdg = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}
set = "{}0000".format(abs(pdg[nutype]))
MC = ExtractMC([set])
# Create binning:
# mass: true binning depends on the DM mass
mass = 1000

# Binning:
# E true
Etrue_center = np.linspace(1., mass, 100)
Ewidth = (mass-1.)/(100.-1.)
Etrue_edges = [E - Ewidth/2. for E in Etrue_center]
Etrue_edges.append(Etrue_center[-1] + Ewidth/2.)
# Psi true
Psitrue_edges = np.linspace(0, 180, 51)
Psiwidth = 180./50.
Psitrue_center = np.array([Psitrue_edges[i]+Psiwidth/2. for i in range(len(Psitrue_edges)-1)])
# E reco
Ereco_edges = pow(10., np.linspace(np.log10(1.), np.log10(1e3), 50+1))
Ereco_center = np.array([np.sqrt(Ereco_edges[i]*Ereco_edges[i+1]) for i in range(len(Ereco_edges) - 1)])
# Psi reco
Psireco_edges = np.linspace(0., 180., 18+1)
Psireco_center = np.array( [(Psireco_edges[i]+Psireco_edges[i+1])/2. for i in range(len(Psireco_edges)-1)] )

# PID
PID_edges = np.array([0.,0.5,0.85,1.])
PID_center = np.array( [(PID_edges[i]+PID_edges[i+1])/2. for i in range(len(PID_edges)-1)] )

Bin = GroupBinning(Etrue_edges, Psitrue_edges, Etrue_center, Psitrue_center,
                Ereco_edges, Psireco_edges, Ereco_center, Psireco_center, PID_edges, PID_center)


Resp = KDE_RespMatrix(MC, Bin, bw, method, nu_type=nutype, pid=pid)

output = dict()
output["Resp"] = Resp
output["Bin"] = Bin
outfile = "/data/user/tchau/Sandbox/GC_OscNext/DetResponse/PreComp/RespMatrix_{}_bw{}_{}_pid{}.pkl".format(method, bw, nutype, pid)
pkl.dump(output, open(outfile, "wb"))
