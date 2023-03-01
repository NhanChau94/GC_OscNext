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

parser.add_option("--gcinj", type = int, action = "store", default = 0, metavar  = "<gcinj>", help = "if gc is injected (0:no, 1:yes)",)
parser.add_option("--gcmodel", type = int, action = "store", default = 0, metavar  = "<gcmodel>", help = "if gc is accounted in the fit model (0:no, 1:yes)",)


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
gcinj = options.gcinj
gcmodel = options.gcmodel

if mass < 3000:
    Bin = Std_Binning(mass, N_Etrue=300)
else:
    Bin = Std_Binning(3000, N_Etrue=500)

# Compute DM rate
Reco = RecoRate(channel, mass, profile, Bin, type="Resp", spectra='Charon', set=mcfit)
# Different options for Jfactor
if Jfit!='nominal':
    MyJ = Jf(profile=profile)
    J_Clumpy = MyJ.Jfactor_Clumpy(errors=Jfit)
    J_int = Interpolate_Jfactor(J_Clumpy, Bin['true_psi_center'])
    Reco.hist['Jfactor'] = J_int
Reco.Scramble = False
DMRate = Reco.ComputeRecoRate()
Reco.ResetAllHists()
Reco.Scramble = True
DMRateScr=Reco.ComputeRecoRate()
Reco.ResetAllHists()

if Jinj!='nominal':
    MyJ = Jf(profile=profile)
    J_Clumpy = MyJ.Jfactor_Clumpy(errors=Jinj)
    J_int = Interpolate_Jfactor(J_Clumpy, Bin['true_psi_center'])
    Reco.hist['Jfactor'] = J_int
Reco.Scramble = False
Reco.set=mcinj
DMRate_inj=Reco.ComputeRecoRate()
Reco.ResetAllHists()

Reco.Scramble = True
Reco.set=mcinj
DMRateScr_inj=Reco.ComputeRecoRate()

# Bkg and GC astro rate
exposure = 8* 365.*24.* 60.* 60.
Bkg = ScrambleBkg(Bin, bw="ISJ", oversample=10)
BurnSample = DataHist(Bin)
Ndata = 10*np.sum(BurnSample) # expected total number of data after 8 years
BkgRate = 10*np.sum(BurnSample)*Bkg/(np.sum(Bkg))/(exposure)

GCRate = GC_RecoRate(Bin, method='evtbyevt', set='1122', scrambled=False)[0]
GCRateScr = GC_RecoRate(Bin, method='evtbyevt', set='1122', scrambled=True)[0]


# Create the PDF object
SignalPDF = PdfBase(DMRate.flatten()/np.sum(DMRate.flatten()), name="SignalPDF")
ScrSignalPDF = PdfBase(DMRateScr.flatten()/np.sum(DMRateScr.flatten()), name="ScrSignalPDF")

SignalPDF_inj = PdfBase(DMRate_inj.flatten()/np.sum(DMRate_inj.flatten()), name="SignalPDF_inj")
ScrSignalPDF_inj = PdfBase(DMRateScr_inj.flatten()/np.sum(DMRateScr_inj.flatten()), name="ScrSignalPDF_inj")

GCPDF = PdfBase(GCRate.flatten()/np.sum(GCRate.flatten()), name="GC")
ScrGCPDF = PdfBase(GCRateScr.flatten()/np.sum(GCRateScr.flatten()), name="GCScr")

# Assuming the Scr Bkg from burn sample is the atm Bkg
BkgPDF = PdfBase(BkgRate.flatten()/np.sum(BkgRate.flatten()), name="BkgAtm")

# The data with the assumption of signal fraction xi_true and galactic fraction gc_true:
xi_true = np.sum(DMRate*1e-23)/(np.sum(BkgRate))
gc_true = np.sum(GCRate)/(np.sum(BkgRate))
dm_inj = Parameter(value=xi_true, limits=(0,1), fixed=True, name="dm_inj")
gc_inj = Parameter(value=gc_true, limits=(0,1), fixed=True, name="gc_inj")


# use for model fitting:
dm_H1 = Parameter(value=0., limits=(0,1), fixed=False, name="dm_H1")
gc_H1 = Parameter(value=gc_true, limits=(0,1), fixed=True, name="gc_H1")
dm_H0 = Parameter(value=xi_true, limits=(0,1), fixed=True, name="dm_H0")
gc_H0 = Parameter(value=gc_true, limits=(0,1), fixed=True, name="gc_H0")

# Set gc fraction to zero in case it is not injected or included in the fit
if gcinj==0:
    print('\n no GC in the pseudo data \n')
    gc_inj.value=0
if gcmodel==0:
    print('\n no GC in the fit model \n')
    gc_H1.value=0
    gc_H0.value=0


pseudo_data = dm_inj* SignalPDF_inj + gc_inj* GCPDF + (1-dm_inj-gc_inj)* BkgPDF

# Scramble bkg now yields:
ScrBkgPDF = dm_inj* ScrSignalPDF_inj + gc_inj* ScrGCPDF + (1-dm_inj-gc_inj)* BkgPDF

# llh model
if llh=='SignalSub':
    modelH0 = dm_H0* SignalPDF + gc_H0* GCPDF + ScrBkgPDF - dm_H0* ScrSignalPDF - gc_H0* ScrGCPDF
    modelH1 = dm_H1* SignalPDF + gc_H1* GCPDF + ScrBkgPDF - dm_H1* ScrSignalPDF - gc_H1* ScrGCPDF
elif llh=='Poisson':
    modelH0 = dm_H0* SignalPDF + gc_H0* GCPDF + (1-dm_H0-gc_H0)*ScrBkgPDF
    modelH1 = dm_H1* SignalPDF + gc_H1* GCPDF + (1-dm_H1-gc_H1)*ScrBkgPDF


f_inj = np.linspace(0, 0.005, 50)
ds = DataSet()
signal = dict()
lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)

for inj in f_inj:
    # change the signal injection:
    pseudo_data.parameters["dm_inj"].value = inj
    if llh=='SignalSub':
        lr.models['H1'].parameters["dm_inj"].value = inj
    signal[inj] = np.array([])
    for n in range(500):
        ds.sample(Ndata, pseudo_data)
 
        lr.data = ds
        lr.fit("H1")
        fitval = lr.models['H1'].parameters["dm_H1"].value
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
output['gcinj'] = f_inj
output['gcmodel'] = gc_H1.value
output['gcinj'] = gc_inj.value



if Jinj=='nominal' and Jfit=='nominal': #normal Jfactor case
    path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/SignalRecovery/{}_{}_{}GeV_MCfit{}_MCinj{}_llh{}_gcinj{}_gcmodel{}.pkl'.format(channel, profile, mass, mcfit, mcinj, llh, gcinj, gcmodel)
else:
    path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/SignalRecovery/{}_{}_{}GeV_MCfit{}_MCinj{}_Jfit{}_Jinj{}_llh{}_gcinj{}_gcmodel{}.pkl'.format(channel, profile, mass, mcfit, mcinj, Jfit, Jinj, llh, gcinj, gcmodel)

pkl.dump(output, open(path, "wb"))