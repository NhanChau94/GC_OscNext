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
from Background import *

from modeling import PdfBase, Model, Parameter
from data import DataSet
from llh import LikelihoodRatioTest
from scipy.interpolate import interp1d

def UpperLimit(Rate, Rate_Scr, Bkg, Data, LLH="SignalSub", CL=1.64, deltaxi=5e-6, exposure= 2933.8*24*60*60, method='interpolate', sampling=False):
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
    if sampling==False:
        data.asimov(np.sum(Data), DataPDF)
    else:
        data.sample(np.sum(Data), DataPDF)
    lr.data = data

    if method=='interpolate':
        T = np.array([0])
        x = deltaxi
        xi = np.array([0])
        while T[-1]<CL*2.:
            xi = np.append(xi,x)
            lr.models['H0'].parameters["fix"].value = x
            lr.fit("H0")
            lr.fit("H1")
            if lr.models['H1'].parameters["fit"].value > x:
                T = np.append(T, 0)
            else:    
                T = np.append(T, lr.TS)
            x+=deltaxi                    
        f = interp1d(T, xi)# kind='quadratic')
        xi_CL = f(CL)
    elif method=='bisection':
        xi_CL = lr.upperlimit_llhinterval('fit', 'fix', 90)  

    print('='*20)    
    print('signal fraction: {}'.format(xi_CL))
    lr.models['H0'].parameters["fix"].value = xi_CL
    lr.fit("H0")
    lr.fit("H1")
    print('TS value at the output upper limit: {}'.format(lr.TS))

    # Convert to thermal cross-section:
    Nsignal = xi_CL* np.sum(Data)
    sigma = Nsignal/(np.sum(Rate*exposure))
    return sigma




parser = OptionParser()
# i/o options
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = 'string', action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("-s", "--spectra", type = 'string', action = "store", default = "Charon", metavar  = "<spectra>", help = "Spectra: Charon or PPPC4",)
parser.add_option("--mc", type = 'string', action = "store", default = "0000", metavar  = "<spectra>", help = "MC set",)
parser.add_option("-b", "--bkg", type = 'string', action = "store", default = "FFT", metavar  = "<bkg>", help = "Background type: FFT with ISJ or sklearn with CV bandwidth",)
parser.add_option("-m", "--mass", type = float, action = "store", default = None, metavar  = "<mass>", help = "mass values: in case of specify only one value of mass will be input",)
parser.add_option("-u", "--up", type = float, action = "store", default = 100, metavar  = "<up>", help = "Dark Matter mass up",)
parser.add_option("-l", "--low", type = float, action = "store", default = 1, metavar  = "<low>", help = "Dark Matter mass low",)
parser.add_option("-n", "--n", type = int, action = "store", default = 100, metavar  = "<n>", help = "Dark Matter mass - N point scan",)
parser.add_option("--method", type = 'string', action = "store", default = "interpolate", metavar  = "<method>", help = "method for getting the UL: interpolate or bisection",)
parser.add_option("--nsample", type = int, action = "store", default = 0, metavar  = "<n>", help = "Sampling to make brazillian plot: 0 = no sampling",)


(options, args) = parser.parse_args()


channel = options.channel
profile = options.profile
up = options.up
low = options.low
n = options.n
mc = options.mc
bkg = options.bkg
method = options.method
nsample = options.nsample

Bin = Std_Binning(300, N_Etrue=100)
Reco = RecoRate(channel, 300, profile, Bin, type="Resp", spectra='Charon', set=mc)

if bkg=='FFT':
    BkgPDF = ScrambleBkg(Bin, bw="ISJ", oversample=10)
elif bkg=='sklearn':
    BkgPDF = ScrambleBkg(Bin, bw=0.03, method='sklearn' ,oversample=10)    

if nsample!=0:
        mean = np.array([])    
        low1 = np.array([])  
        low2 = np.array([])  
        up1 = np.array([])
        up2 = np.array([])
else:
    UL = np.array([])

masses = np.exp(np.linspace(np.log(low), np.log(up), n))
for mass in masses:
    # Bin
    if mass < 3000:
        Bin = Std_Binning(mass, N_Etrue=300)
    else:
        Bin = Std_Binning(3000, N_Etrue=300)
    Reco.mass = mass
    Reco.bin = Bin
    
    Reco.Scramble = False    
    Rate = Reco.ComputeRecoRate()
    Reco.ResetAllHists()

    Reco.Scramble = True
    Rate_Scr = Reco.ComputeRecoRate()
    Reco.ResetAllHists()
    BurnSample = DataHist(Bin)
    if nsample==0:
        UL = np.append(UL, UpperLimit(Rate, Rate_Scr, BkgPDF, 10*np.sum(BurnSample)*BkgPDF/(np.sum(BkgPDF)), method=method))
    else:
        UL_dist = np.array([])
        for i in range(nsample):
            UL_dist = np.append(UL_dist, UpperLimit(Rate, Rate_Scr, BkgPDF, 10*np.sum(BurnSample)*BkgPDF/(np.sum(BkgPDF)), method=method, sampling=True))
        arr1 = np.percentile(UL_dist, [2.5, 50, 97.5])
        arr2 = np.percentile(UL_dist, [16, 50, 84])

        #Compute 1 and 2 sigma bands

        mean = np.append(mean, arr1[1])
        low1 = np.append(low1, arr1[0])
        up1 = np.append(up1, arr1[2])
        low2 = np.append(low2, arr2[0])
        up2 = np.append(up2, arr2[2])    


outdict = dict()
outdict['mass'] = masses
if nsample==0:
    path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/UpperLimit/{}_{}_{}points_MC{}_BKG{}_ULby{}.pkl'.format(channel, profile, n, mc, bkg, method)

    outdict['UL'] = UL
    print('='*20)
    print('masses: {}'.format(masses))
    print('UL: {}'.format(UL))
    print('='*20)
else:
    path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/UpperLimit/{}_{}_{}points_MC{}_BKG{}_ULby{}_nsample{}.pkl'.format(channel, profile, n, mc, bkg, method, nsample)

    outdict['mean'] = mean
    outdict['16'] = low1
    outdict['84'] = up1
    outdict['2.5'] = low2
    outdict['97.5'] = up2
   

pkl.dump(outdict, open(path, "wb"))