#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python
import numpy as np
import sys, os
import pickle as pkl
from optparse import OptionParser

sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DMfit/DMfit")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/PDFs")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DetResponse")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Utils")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Spectra")

from Detector import *
from Signal import *
from Plot_Histogram import *
from Background import *
from Jfactor import *


from modeling import PdfBase, Model, Parameter
from data import DataSet
from llh import LikelihoodRatioTest


def SignalRecovery(SignalPDF, ScrSignalPDF, SignalPDF_inj, ScrSignalPDF_inj, BkgPDF, Ndata, llh_type='SignalSub', ntrials=200, npoints=50):
    sig_fit = Parameter(value=0.5, limits=(0,1), fixed=False, name="sig_fit")
    sig_inj = Parameter(value=0.0, limits=(0,1), fixed=True, name="sig_inj")

    model = (sig_fit* SignalPDF) + (1-sig_fit)*(BkgPDF) + sig_fit* BkgPDF - sig_fit*ScrSignalPDF
    pseudo_data = (sig_inj* SignalPDF_inj) + (1-sig_inj)*(BkgPDF)#+ sig_inj* BkgPDF - sig_inj*ScrSignalPDF
    lr = LikelihoodRatioTest(model = model, null_model = pseudo_data)

    f_inj = np.linspace(0, 0.005, npoints)
    ds = DataSet()
    signal = dict()
    for inj in f_inj:
        # change the signal injection:
        pseudo_data.parameters["sig_inj"].value = inj
        signal[inj] = np.array([])
        for n in range(ntrials):
            ds.sample(Ndata, pseudo_data)
            
            # If Scramble pseudo data is used to estimate the background:
            if llh_type=='SignalSub':
                lr.models['H1'] = (sig_fit* SignalPDF) + (1-pseudo_data.parameters["sig_inj"])*(BkgPDF) + pseudo_data.parameters["sig_inj"]* ScrSignalPDF_inj - sig_fit*ScrSignalPDF
            elif llh_type=='Normal':
                lr.models['H1'] = (sig_fit* SignalPDF) + (1-sig_fit)*((1-pseudo_data.parameters["sig_inj"])*(BkgPDF) + pseudo_data.parameters["sig_inj"]* ScrSignalPDF)
                    
            lr.data = ds
            lr.fit("H1")
            fitval = lr.models['H1'].parameters["sig_fit"].value
            signal[inj] = np.append(signal[inj], fitval)

    #Extracting percentile: 
    mean = np.array([])
    low1 = np.array([])
    low2 = np.array([])

    up1 = np.array([])
    up2 = np.array([])

    for inj in f_inj:
        arr1 = np.percentile(signal[inj], [32, 50, 68])
        arr2 = np.percentile(signal[inj], [10, 50, 90])

        mean = np.append(mean, arr1[1])
        low1 = np.append(low1, arr1[0])
        up1 = np.append(up1, arr1[2])
        low2 = np.append(low2, arr2[0])
        up2 = np.append(up2, arr2[2])
    output = dict()
    output['mean'] = mean
    output['10'] = low2
    output['90'] = up2
    output['32'] = low1
    output['68'] = up1
    output['f_inj'] = f_inj
    return output



parser = OptionParser()
# i/o options
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = 'string', action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("-m", "--mass", type = float, action = "store", default = None, metavar  = "<mass>", help = "mass value",)
parser.add_option("-s", "--spectra", type = 'string', action = "store", default = "Charon", metavar  = "<spectra>", help = "Spectra: Charon or PPPC4",)
parser.add_option("--mcfit", type = 'string', action = "store", default = "1122", metavar  = "<mcfit>", help = "MC set use for fitting",)
parser.add_option("--mcinj", type = 'string', action = "store", default = "1122", metavar  = "<mcinj>", help = "MC set use for the injection",)
parser.add_option("--Jfit", type = 'string', action = "store", default = "nominal", metavar  = "<Jfit>", help = "Jfactor use for fitting",)
parser.add_option("--Jinj", type = 'string', action = "store", default = "nominal", metavar  = "<Jinj>", help = "Jfactor use for the injection",)
parser.add_option("--llh", type = 'string', action = "store", default = "SignalSub", metavar  = "<llh>", help = "LLH type",)

(options, args) = parser.parse_args()


channel = options.channel
profile = options.profile
spectra = options.spectra
mcfit = options.mcfit
mcinj = options.mcinj
Jfit = options.Jfit
Jinj = options.Jinj
mass = options.mass
llh = options.llh

if mass < 3000:
    Bin = Std_Binning(mass, N_Etrue=100)
else:
    Bin = Std_Binning(3000, N_Etrue=300)

# Compute rate
Bkg = ScrambleBkg(Bin, bw="ISJ", oversample=10)
Reco = RecoRate(channel, mass, profile, Bin, type="Resp", spectra='Charon', set=mcfit)
if Jfit!='nominal':
    MyJ = Jf(profile=profile)
    J_Clumpy = MyJ.Jfactor_Clumpy(errors=Jfit)
    J_int = Interpolate_Jfactor(J_Clumpy, Bin['true_psi_center'])
    Reco.hist['Jfactor'] = J_int
Rate = Reco.ComputeRecoRate()
Reco.ResetAllHists()


if Jfit!='nominal':
    MyJ = Jf(profile=profile)
    J_Clumpy = MyJ.Jfactor_Clumpy(errors=Jfit)
    J_int = Interpolate_Jfactor(J_Clumpy, Bin['true_psi_center'])
    Reco.hist['Jfactor'] = J_int
Reco.Scramble = True
Rate_Scr = Reco.ComputeRecoRate()
Reco.ResetAllHists()


if Jinj!='nominal':
    MyJ = Jf(profile=profile)
    J_Clumpy = MyJ.Jfactor_Clumpy(errors=Jinj)
    J_int = Interpolate_Jfactor(J_Clumpy, Bin['true_psi_center'])
    Reco.hist['Jfactor'] = J_int
Reco.Scramble = False
Reco.set=mcinj
Rate_inj=Reco.ComputeRecoRate()
Reco.ResetAllHists()

if Jinj!='nominal':
    MyJ = Jf(profile=profile)
    J_Clumpy = MyJ.Jfactor_Clumpy(errors=Jinj)
    J_int = Interpolate_Jfactor(J_Clumpy, Bin['true_psi_center'])
    Reco.hist['Jfactor'] = J_int
Reco.Scramble = True
Reco.set=mcinj
Rate_inj_scr=Reco.ComputeRecoRate()

BurnSample = DataHist(Bin)

# Create PDF 
SignalPDF = PdfBase(Rate.flatten()/np.sum(Rate.flatten()), name="SignalPDF")
SignalPDF_inj = PdfBase(Rate_inj.flatten()/np.sum(Rate_inj.flatten()), name="SignalPDF_inj")
SignalPDF_Scr = PdfBase(Rate_Scr.flatten()/np.sum(Rate_Scr.flatten()), name="ScrSignalPDF")
SignalPDF_inj_scr = PdfBase(Rate_inj_scr.flatten()/np.sum(Rate_inj_scr.flatten()), name="ScrSignalPDF_inj")

BkgPDF = PdfBase(Bkg.flatten()/np.sum(Bkg.flatten()), name="Bkg")

output = SignalRecovery(SignalPDF, SignalPDF_Scr, SignalPDF_inj, SignalPDF_inj_scr, BkgPDF, np.sum(10* BurnSample), llh_type=llh)

if Jinj=='nominal' and Jfit=='nominal': #normal Jfactor case
    path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/SignalRecovery/{}_{}_{}GeV_MCfit{}_MCinj{}_llh{}.pkl'.format(channel, profile, mass, mcfit, mcinj, llh)
else:
    path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/SignalRecovery/{}_{}_{}GeV_MCfit{}_MCinj{}_Jfit{}_Jinj{}_llh{}.pkl'.format(channel, profile, mass, mcfit, mcinj, Jfit, Jinj, llh)

pkl.dump(output, open(path, "wb"))