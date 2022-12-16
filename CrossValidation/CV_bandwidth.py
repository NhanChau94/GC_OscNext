#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/RHEL_7_x86_64/bin/python


import numpy as np
from optparse import OptionParser

import sys
import pickle as pkl
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DetResponse/")
from Detector import *

##---------------------------------------------##
##Cross-validation method##
##Optimise bandwidth choice
##Required: 
##   - sample (x) 
##   - array of bandwidths to evaluate (banwdiths)
##Optional (Defaul=None):
##   - weights related to sample x (weighs)
##---------------------------------------------##
def bw_crossvalid(x, bandwidths,weights=None):
    
    '''
    Cross-validation method - Implemented within sklearn
    '''
    
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': bandwidths}, cv=3, n_jobs=-1)
    print ("Grid created")
    
    grid.fit(x.T, sample_weight=weights)
    print ("Grid estimated")

    #Use best estimator to compute the KDE
    bw = grid.best_estimator_.bandwidth
    print ("Best bw estimate found")
    print ("Best bw estimate: {}".format(bw))

    return float(bw)

##---------------------------------------------##
##Cross-validation method##
##Use CV on Monte-Carlo sample
##Required: 
##   - MC dictionary 
##   - neutrino type
##   - pid: 0, 1, 2 (if <0: no pid)
##   - Variables used (string): + 2Dreco (reco psi, reco E) 
##                              + 2Dtrue (true psi, true E)
##                              + 4D (true psi, true E, reco psi, reco E)   
##   - use weight or not (bool)
##---------------------------------------------##    
def Bandwidth_MC(MCdict, nutype, pid, variables, use_weight):
    pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}
    PID = [[0.,0.5],[0.5, 0.85],[0.85, 1]]
    
    print("----{}".format(nutype))

    if pid>=0 and pid<3:
        loc = np.where(  (MCdict["nutype"]==pdg_encoding[nutype]) & (MCdict["PID"]>=PID[pid][0])
                    & (MCdict["PID"]<PID[pid][1]) & MCdict["w"]<5000)
        print("----pid: {}".format(pid))

            
    else:
        loc = np.where(  (MCdict["nutype"]==pdg_encoding[nutype]) )                        
        print('no pid accounted')
    psitrue = MCdict["psi_true"][loc]
    Etrue = MCdict["E_true"][loc]
    psireco = MCdict["psi_reco"][loc]
    Ereco = MCdict["E_reco"][loc]
    weight = MCdict["w"][loc]

    # Prepare sample for training:
    if variables=='2Dreco':
        psiE_train = np.vstack([np.log10(psireco),np.log10(Ereco)])
    if variables=='2Dtrue':
        psiE_train = np.vstack([np.log10(psitrue),np.log10(Etrue)])
    if variables=='4D':
        psiE_train = np.vstack([np.log10(psitrue),np.log10(Etrue), np.log10(psireco),np.log10(Ereco)])


    # Bw from scott and silverman rule of thumb:
    d = psiE_train.shape[0]
    n = psiE_train.shape[1]
    print('dimension: {}'.format(d))
    print('length of the sample {}'.format(n))
    bw_scott = n**(-1./(d+4))
    bw_silverman = (n * (d + 2) / 4.)**(-1. / (d + 4))
    # Bandwidth values for scanning:
    bw = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])

    print('Bandwidths to scan:')
    print(bw)

    if use_weight:
        weights = weight
    else:
        weights = None

    # print('Weight: {}'.format(weights))    
    bw_cv = bw_crossvalid(psiE_train, bw, weights = weights)

    return bw_cv, bw_scott, bw_silverman

# 
#----------------------------------------------------------------------------------------------------------------------
#Define parameters needed
#----------------------------------------------------------------------------------------------------------------------

parser = OptionParser()
# i/o options
parser.add_option("-v", "--var", type = "string", action = "store", default = "None", metavar  = "<var>", help = "variables used: 2D(true, reco), ",)
parser.add_option("-p", "--pid", type = int, action = "store", default = 0, metavar  = "<pid>", help = "which pid bin: 0(showers), 1(middles), 2(tracks), -1(no pid)",)
parser.add_option("-n", "--nutype", type = "string", action = "store", default = "nu_e", metavar  = "<nutype>", help = "which nutype: nu_e, nu_mu, nu_tau, nu_e_bar, nu_mu_bar, nu_tau_bar",)
parser.add_option("-w", "--weight",
                  action="store_true", dest="weight", default=False,
                  help="use weight")

(options, args) = parser.parse_args()

var = options.var
pid = options.pid
nutype = options.nutype
weight = options.weight

# Extract the MC:
pdg = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}
set = "{}0000".format(abs(pdg[nutype]))
MC = ExtractMC([set])
output = dict()
if pid>=0:
    outfile = "/data/user/tchau/Sandbox/GC_OscNext/CrossValidation/BW/{}_pid{}_{}_weight{}.pkl".format(nutype, pid, var, weight)
else:
    outfile = "/data/user/tchau/Sandbox/GC_OscNext/CrossValidation/BW/{}_nopid_{}_weight{}.pkl".format(nutype, var, weight)

bw_output = Bandwidth_MC(MC, nutype, pid, var, weight)
output['bw_cv'] = bw_output[0]
output['bw_scott'] = bw_output[1]
output['bw_silverman'] = bw_output[2]
pkl.dump(output, open(outfile, "wb"))