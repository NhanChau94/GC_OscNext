#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python
import numpy as np
import sys, os
import pickle as pkl
from optparse import OptionParser

base_path=os.getenv('GC_DM_BASE')
data_path=os.getenv('GC_DM_DATA')
output_path=os.getenv('GC_DM_OUTPUT')
sys.path.append(f"{base_path}/Utils/")
sys.path.append(f"{base_path}/Spectra/")
sys.path.append(f"{base_path}/DetResponse/")
sys.path.append(f"{base_path}/PDFs/")
sys.path.append(f"{base_path}/DMfit/DMfit")

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
parser.add_option("--process", type = 'string', action = "store", default = "ann", metavar  = "<process>", help = "process: ann or decay",)
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = 'string', action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("-m", "--mass", type = float, action = "store", default = 100, metavar  = "<mass>", help = "mass value",)
parser.add_option("-s", "--spectra", type = 'string', action = "store", default = "Charon", metavar  = "<spectra>", help = "Spectra: Charon or PPPC4",)
parser.add_option("--mcfit", type = 'string', action = "store", default = "1122", metavar  = "<mcfit>", help = "MC set use for fitting",)
parser.add_option("--mcinj", type = 'string', action = "store", default = "1122", metavar  = "<mcinj>", help = "MC set use for the injection",)
parser.add_option("--Jfit", type = 'string', action = "store", default = "nominal", metavar  = "<Jfit>", help = "Jfactor use for fitting: nominal, error1, error2",)
parser.add_option("--Jinj", type = 'string', action = "store", default = "nominal", metavar  = "<Jinj>", help = "Jfactor use for the injection: nominal, error1, error2",)
parser.add_option("--llh", type = 'string', action = "store", default = "SignalSub", metavar  = "<llh>", help = "LLH type",)
parser.add_option("--gpmodel", type = "string", action = "store", default = None, metavar  = "<gpmodel>", help = "GP model in the fit, use among: pi0, pi0_IC, KRA50, KRA50_IC, KRA5, KRA5_IC, None",)
parser.add_option("--gpinj", type = "string", action = "store", default = None, metavar  = "<gpinj>", help = "GP model used for injection, use among:  pi0, pi0_IC, KRA50, KRA50_IC, KRA5, KRA5_IC, None",)
parser.add_option("--fixGP", type = int, action = "store", default = 1, metavar  = "<fixGP>", help = "in case of including GP, fix its fraction (1) or fit/marginalize it (0)",)



(options, args) = parser.parse_args()

process = options.process
channel = options.channel
profile = options.profile
spectra = options.spectra
mcfit = options.mcfit
mcinj = options.mcinj
Jfit = options.Jfit
Jinj = options.Jinj
mass = options.mass
llh = options.llh
gpinj = options.gpinj
gpmodel = options.gpmodel
fixGP = bool(options.fixGP)

#############################################################################################################
#   0 - Define process (annihilation/decay) and binning scheme 
#

if process=='ann': Etrue_max = mass
if process=='decay': Etrue_max = mass/2.

if Etrue_max < 3000:
    Bin = Std_Binning(Etrue_max, N_Etrue=300)
else:
    # OscNext only select events up to ~ 3TeV
    Bin = Std_Binning(3000, N_Etrue=500)
    
#############################################################################################################
#   1 - Define Signal expectation object and compute the signal expectation
#

Reco = RecoRate(channel, mass, profile, Bin,process=process, type="Resp", spectra='Charon', set=mcfit)

# Different options for Jfactor
if Jfit!='nominal':
    MyJ = Jf(profile=profile)
    J_Clumpy = MyJ.Jfactor_Clumpy(errors=Jfit)
    J_int = Interpolate_Jfactor(J_Clumpy, Bin['true_psi_center'])
    Reco.hist['Jfactor'] = J_int

# Compute original and scrambled signal reconstruction rate
Reco.Scramble = False
DMRate = Reco.ComputeRecoRate()
Reco.ResetAllHists()

Reco.Scramble = True
DMRateScr=Reco.ComputeRecoRate()
Reco.ResetAllHists()

# Compute injection rate
if Jinj!='nominal': # manually load different Jfactor input: error1, error2
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

#############################################################################################################
#   2 - Background rate assuming data ~ 10xBurnSample
#

exposure = 8* 365.*24.* 60.* 60.
Bkg = ScrambleBkg(Bin, bandwidth="ISJ", oversample=10)
BurnSample = DataHist(Bin)
Ndata = 10*np.sum(BurnSample) # expected total number of data after 8 years
BkgRate = 10*np.sum(BurnSample)*Bkg/(np.sum(Bkg))/(exposure)


#############################################################################################################
#   3 - Create PDF object, fraction parameters, and LLR models
#

# Signal PDF in the LLR model
SignalPDF = PdfBase(DMRate.flatten()/np.sum(DMRate.flatten()), name="SignalPDF")
ScrSignalPDF = PdfBase(DMRateScr.flatten()/np.sum(DMRateScr.flatten()), name="ScrSignalPDF")

# Signal PDF injected (into (pseudo)data)
SignalPDF_inj = PdfBase(DMRate_inj.flatten()/np.sum(DMRate_inj.flatten()), name="SignalPDF_inj")
ScrSignalPDF_inj = PdfBase(DMRateScr_inj.flatten()/np.sum(DMRateScr_inj.flatten()), name="ScrSignalPDF_inj")

# Assuming the Scr Bkg from burn sample is the atm Bkg
BkgPDF = PdfBase(BkgRate.flatten()/np.sum(BkgRate.flatten()), name="BkgAtm")

# Signal fraction parameters in: injection, H1 hypothesis, H0 hypothesis
dm_inj = Parameter(value=0., limits=(0,1), fixed=True, name="dm_inj")
dm_H1 = Parameter(value=0., limits=(0,1), fixed=False, name="dm_H1")
dm_H0 = Parameter(value=0., limits=(0,1), fixed=True, name="dm_H0")

# The (pseudo) data then:
pseudo_data = dm_inj* SignalPDF_inj + (1-dm_inj)* BkgPDF

# Bkg as RA scrambled data then yields:
ScrBkgPDF = dm_inj* ScrSignalPDF_inj + (1-dm_inj)* BkgPDF

# LLR model: signal subtraction or normal poissonian
if llh=='SignalSub':
    modelH0 = dm_H0* SignalPDF + ScrBkgPDF - dm_H0* ScrSignalPDF
    modelH1 = dm_H1* SignalPDF + ScrBkgPDF - dm_H1* ScrSignalPDF
else:
    modelH0 = dm_H0* SignalPDF + (1-dm_H0)*ScrBkgPDF
    modelH1 = dm_H1* SignalPDF + (1-dm_H1)*ScrBkgPDF

# In case including GP in the LLR model or in the injection:
# Different models for GP
scales = {'pi0':1, 'pi0_IC':4.95, 'KRA50':1., 'KRA50_IC':0.37, 'KRA5':1., 'KRA5_IC':0.55}
template = {'pi0':'Fermi-LAT_pi0_map.npy', 'pi0_IC':'Fermi-LAT_pi0_map.npy', 
                'KRA50':'KRA-gamma_maps_energies.tuple.npy', 'KRA50_IC':'KRA-gamma_maps_energies.tuple.npy', 
                'KRA5':'KRA-gamma_5PeV_maps_energies.tuple.npy', 'KRA5_IC':'KRA-gamma_5PeV_maps_energies.tuple.npy'}

if gpinj!=None and gpinj!='None':
    # Inject GP to the data

    GPRate_inj = GP_RecoRate(Bin, template=f'{data_path}/GP_template/'+template[gpinj], scale=scales[gpinj])
    GPRate_inj_scr = GP_RecoRate(Bin, template=f'{data_path}/GP_template/'+template[gpinj], scale=scales[gpinj], scrambled=True)

    GPPDF_inj = PdfBase(GPRate_inj.flatten()/np.sum(GPRate_inj.flatten()), name="GC_inj")
    ScrGPPDF_inj = PdfBase(GPRate_inj_scr.flatten()/np.sum(GPRate_inj_scr.flatten()), name="GCScr_inj")

    gc_true = np.sum(GPRate_inj)*exposure/Ndata
    gc_inj = Parameter(value=gc_true, limits=(0,1), fixed=True, name="gc_inj")
    pseudo_data = dm_inj* SignalPDF + gc_inj* GPPDF_inj + (1-dm_inj-gc_inj)* BkgPDF
    ScrBkgPDF = dm_inj* ScrSignalPDF + gc_inj* ScrGPPDF_inj + (1-dm_inj-gc_inj)* BkgPDF

if gpmodel!=None and gpmodel!='None':
    # Add the GP to the LLR model

    GPRate_model = GP_RecoRate(Bin, template='/data/user/tchau/DarkMatter_OscNext/GP_template/'+template[gpmodel], scale=scales[gpmodel])
    GPRate_model_scr = GP_RecoRate(Bin, template='/data/user/tchau/DarkMatter_OscNext/GP_template/'+template[gpmodel], scale=scales[gpmodel], scrambled=True)

    GPPDF_model = PdfBase(GPRate_model.flatten()/np.sum(GPRate_model.flatten()), name="GC_model")
    ScrGPPDF_model = PdfBase(GPRate_model_scr.flatten()/np.sum(GPRate_model_scr.flatten()), name="GCScr_model")

    gc_model = np.sum(GPRate_model)*exposure/Ndata
    gc_H1 = Parameter(value=gc_model, limits=(gc_model* (1.-0.95), gc_model* (1.+0.95)), fixed=fixGP, name="gc_H1")
    gc_H0 = Parameter(value=gc_model, limits=(0,1), fixed=fixGP, name="gc_H0")

    if llh=='SignalSub':
        modelH0 = dm_H0* SignalPDF + gc_H0* GPPDF_model + ScrBkgPDF - dm_H0* ScrSignalPDF - gc_H0* ScrGPPDF_model
        modelH1 = dm_H1* SignalPDF + gc_H1* GPPDF_model + ScrBkgPDF - dm_H1* ScrSignalPDF - gc_H1* ScrGPPDF_model
    else:
        modelH0 = dm_H0* SignalPDF + gc_H0* GPPDF_model + (1-dm_H0-gc_H0)*ScrBkgPDF
        modelH1 = dm_H1* SignalPDF + gc_H1* GPPDF_model + (1-dm_H1-gc_H1)*ScrBkgPDF

# Create asimov/median dataset and the LLR model object
ds = DataSet()
ds.asimov(Ndata, pseudo_data)
lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)
lr.data = ds


#############################################################################################################
#   4 - Scan DM injection fraction values and make signal recovery fit
#

# First compute the 90 CL value of DM fraction and then scan from 0->2*  90 CL value
xi_CL = lr.upperlimit_llhinterval('dm_H1', 'dm_H0', 90)  
f_inj = np.linspace(0, 2*xi_CL, 50)
signal = dict()
ntrial = 500
for inj in f_inj:
    # change the signal injection:
    pseudo_data.parameters["dm_inj"].value = inj
    lr.models['H1'].parameters["dm_inj"].value = inj
    signal[inj] = np.array([])
    for n in range(ntrial):
        # resampling the data to obtain median, 60, 90% containment of the fit value
        ds.sample(Ndata, pseudo_data) 
        lr.data = ds
        lr.fit("H1")
        fitval = lr.models['H1'].parameters["dm_H1"].value
        signal[inj] = np.append(signal[inj], fitval)
        # print("#"*30)
        # print("injection: {}".format(inj))
        # print("fit value: {}".format(fitval))

#Extracting percentile: 
mean = np.array([])
low1 = np.array([])
low2 = np.array([])

up1 = np.array([])
up2 = np.array([])

for inj in f_inj:
    arr1 = np.percentile(signal[inj], [16, 50, 84])
    arr2 = np.percentile(signal[inj], [5., 50., 95.])

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
output['xi_CL'] = xi_CL
print(f_inj)
print(mean)


if Jinj=='nominal' and Jfit=='nominal': #normal Jfactor case
    path = '{}/SignalRecovery/{}_{}_{}_{}GeV_MCfit{}_MCinj{}_llh{}_gcinj{}_gcmodel{}_fixgp{}.pkl'.format(output_path, process, channel, profile, mass, mcfit, mcinj, llh, gpinj, gpmodel, fixGP)
else:
    path = '{}/SignalRecovery/{}_{}_{}_{}GeV_MCfit{}_MCinj{}_Jfit{}_Jinj{}_llh{}_gcinj{}_gcmodel{}_fixgp{}.pkl'.format(output_path, process, channel, profile, mass, mcfit, mcinj, Jfit, Jinj, llh, gpinj, gpmodel, fixGP)

if not (os.path.exists(f'{output_path}/SignalRecovery/')): os.makedirs(f'{output_path}/SignalRecovery/')
pkl.dump(output, open(path, "wb"))