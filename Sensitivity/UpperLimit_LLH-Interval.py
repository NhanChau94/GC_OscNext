#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python

import sys
import numpy as np
import pickle as pkl
from optparse import OptionParser

sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DMfit/DMfit")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/PDFs")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DetResponse")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Utils")

from Detector import *
from Signal import *
from Plot_Histogram import *

from modeling import PdfBase, Model, Parameter
from data import DataSet
from llh import LikelihoodRatioTest
from scipy.interpolate import interp1d

def UpperLimit(Rate, Rate_Scr, Bkg, Data, LLH="SignalSub", CL=1.64, deltaxi=5e-5, exposure= 2933.8*24*60*60):
    SignalPDF = PdfBase(Rate.flatten()/np.sum(Rate.flatten()), name="SignalPDF")
    ScrSignalPDF = PdfBase(Rate_Scr.flatten()/np.sum(Rate_Scr.flatten()), name="ScrSignalPDF")
    BkgPDF = PdfBase(Bkg.flatten()/np.sum(Bkg.flatten()), name="Bkg")
    DataPDF = PdfBase(Data.flatten()/np.sum(Data.flatten()), name="Data")


    sig_fit = Parameter(value=0.001, limits=(0,1), fixed=False, name="fit")
    sig_fix = Parameter(value=0.0, limits=(0,1), fixed=True, name="fix")
    sig_inj = Parameter(value=0.0, limits=(0,1), fixed=True, name="inj")

    # signal subtraction
    if LLH=="SignalSub":
        modelH1 = (sig_fit* SignalPDF) + (1-sig_fit)*(BkgPDF) + sig_fit* BkgPDF - sig_fit*ScrSignalPDF
        modelH0 = (sig_fix* SignalPDF) + (1-sig_fix)*(BkgPDF) + sig_fix* BkgPDF - sig_fix*ScrSignalPDF
    elif LLH=="Poisson":
    # # in case of normal likelihood:
        modelH1 = (sig_fit* SignalPDF) + (1-sig_fit)*(BkgPDF)
        modelH0 = (sig_fix* SignalPDF) + (1-sig_fix)*(BkgPDF)

    lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)
    # pseudo_data = (sig_inj* SignalPDF) + (1-sig_inj)*(BkgPDF) + sig_inj* BkgPDF - sig_inj*ScrSignalPDF
    data = DataSet()
    data.asimov(np.sum(Data), DataPDF)
    lr.data = data

    T = np.array([0])
    x = deltaxi
    xi = np.array([0])
    while T[-1]<CL*1.5:
        xi = np.append(xi,x)
        lr.models['H0'].parameters["fix"].value = x
        lr.fit("H0")
        lr.fit("H1")
        if lr.models['H1'].parameters["fit"].value > x:
            T = np.append(T, 0)
        else:    
            T = np.append(T, lr.TS)
        x+=deltaxi
        
    
    f = interp1d(T, xi)
    xi_CL = f(CL)

    # Convert to thermal cross-section:
    Nsignal = xi_CL* np.sum(Data)
    sigma = Nsignal/(np.sum(Rate*exposure))
    return sigma




parser = OptionParser()
# i/o options
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = 'string', action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("-s", "--spectra", type = 'string', action = "store", default = "Charon", metavar  = "<spectra>", help = "Spectra: Charon or PPPC4",)
parser.add_option("-m", "--mass", type = float, action = "store", default = None, metavar  = "<mass>", help = "mass values: in case of specify only one value of mass will be input",)
parser.add_option("-u", "--up", type = float, action = "store", default = 100, metavar  = "<up>", help = "Dark Matter mass up",)
parser.add_option("-l", "--low", type = float, action = "store", default = 1, metavar  = "<low>", help = "Dark Matter mass low",)
parser.add_option("-n", "--n", type = int, action = "store", default = 100, metavar  = "<n>", help = "Dark Matter mass - N point scan",)


(options, args) = parser.parse_args()


channel = options.channel
profile = options.profile
up = options.up
low = options.low
n = options.n

Bin = Std_Binning(300, N_Etrue=100)
Reco = RecoRate(channel, 300, profile, Bin, type="Resp", spectra='Charon')

Bkg_bwISJ = ScrambleBkg(Bin, bw="ISJ", oversample=10)
UL = np.array([])
masses = np.exp(np.linspace(np.log(low), np.log(up), n))
for mass in masses:
    # Bin
    if mass <1400:
        Bin = Std_Binning(mass, N_Etrue=100)
    else:
        Bin = Std_Binning(1400, N_Etrue=100)
    Reco.mass = mass
    Reco.bin = Bin
    Reco.Scramble = False    
    Rate = Reco.ComputeRecoRate()
    Reco.Scramble = True
    Rate_Scr = Reco.ComputeRecoRate()
    BurnSample = DataHist(Bin)
    UL = np.append(UL, UpperLimit(Rate, Rate_Scr, Bkg_bwISJ, 10*np.sum(BurnSample)*Bkg_bwISJ/(np.sum(Bkg_bwISJ))))

path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/UpperLimit/{}_{}_{}.pkl'.format(channel, profile, n)
outdict = dict()
outdict['mass'] = masses
outdict['UL'] = UL
print('masses: {}'.format(masses))
print('UL: {}'.format(UL))

pkl.dump(outdict, open(path, "wb"))