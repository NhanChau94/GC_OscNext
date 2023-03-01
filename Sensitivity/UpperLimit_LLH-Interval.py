#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python

import sys
import numpy as np
import pickle as pkl
from optparse import OptionParser

sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DMfit/DMfit")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/PDFs")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DetResponse")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Utils")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Spectra")


from Detector import *
from Signal import *
from Background import *
from Jfactor import *

from modeling import PdfBase, Model, Parameter
from data import DataSet
from llh import LikelihoodRatioTest
from scipy.interpolate import interp1d

def UpperLimit(DMRate, DMRateScr, GCRate, GCRateScr, BkgPDF, DataPDF, Ndata, gcmodel=True,
               llh="SignalSub", CL=1.64, deltaxi=5e-6, exposure= 2933.8*24*60*60, method='interpolate', sampling=False):

    # Create the PDF object
    SignalPDF = PdfBase(DMRate.flatten()/np.sum(DMRate.flatten()), name="SignalPDF")
    ScrSignalPDF = PdfBase(DMRateScr.flatten()/np.sum(DMRateScr.flatten()), name="ScrSignalPDF")

    GCPDF = PdfBase(GCRate.flatten()/np.sum(GCRate.flatten()), name="GC")
    ScrGCPDF = PdfBase(GCRateScr.flatten()/np.sum(GCRateScr.flatten()), name="GCScr")

    # use for model fitting:
    if gcmodel==True:
        gc_assumed = np.sum(GCRate)* exposure / Ndata 
    else:
        gc_assumed = 0.    
    dm_H1 = Parameter(value=0., limits=(0,1), fixed=False, name="dm_H1")
    gc_H1 = Parameter(value=gc_assumed, limits=(0,1), fixed=True, name="gc_H1")
    dm_H0 = Parameter(value=0., limits=(0,1), fixed=True, name="dm_H0")
    gc_H0 = Parameter(value=gc_assumed, limits=(0,1), fixed=True, name="gc_H0")

    if llh=='SignalSub':
        modelH0 = dm_H0* SignalPDF + gc_H0* GCPDF + BkgPDF - dm_H0* ScrSignalPDF - gc_H0* ScrGCPDF
        modelH1 = dm_H1* SignalPDF + gc_H1* GCPDF + BkgPDF - dm_H1* ScrSignalPDF - gc_H1* ScrGCPDF
    elif llh=='Poisson':
        modelH0 = dm_H0* SignalPDF + gc_H0* GCPDF + (1-dm_H0-gc_H0)*BkgPDF
        modelH1 = dm_H1* SignalPDF + gc_H1* GCPDF + (1-dm_H1-gc_H1)*BkgPDF

    lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)
    # pseudo_data = (sig_inj* SignalPDF) + (1-sig_inj)*(BkgPDF) + sig_inj* BkgPDF - sig_inj*ScrSignalPDF
    data = DataSet()
    if sampling==False:
        data.asimov(Ndata, DataPDF)
    else:
        data.sample(Ndata, DataPDF)
    lr.data = data

    if method=='interpolate':
        T = np.array([0])
        x = deltaxi
        xi = np.array([0])
        while T[-1]<CL*2.:
            xi = np.append(xi,x)
            lr.models['H0'].parameters["dm_H0"].value = x
            lr.fit("H0")
            lr.fit("H1")
            if lr.models['H1'].parameters["dm_H1"].value > x:
                T = np.append(T, 0)
            else:    
                T = np.append(T, lr.TS)
            x+=deltaxi                    
        f = interp1d(T, xi)# kind='quadratic')
        xi_CL = f(CL)
    elif method=='bisection':
        xi_CL = lr.upperlimit_llhinterval('dm_H1', 'dm_H0', 90)  

    print('='*20)    
    print('signal fraction: {}'.format(xi_CL))
    lr.models['H0'].parameters["dm_H0"].value = xi_CL
    lr.fit("H0")
    lr.fit("H1")
    print('TS value at the output upper limit: {}'.format(lr.TS))

    # Convert to thermal cross-section:
    Nsignal = xi_CL* Ndata
    sigma = Nsignal/(np.sum(DMRate*exposure))
    return sigma




parser = OptionParser()
# i/o options
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = 'string', action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("--process", type = 'string', action = "store", default = "ann", metavar  = "<process>", help = "process: ann or decay",)
parser.add_option("-s", "--spectra", type = 'string', action = "store", default = "Charon", metavar  = "<spectra>", help = "Spectra: Charon or PPPC4",)
parser.add_option("--mc", type = 'string', action = "store", default = "0000", metavar  = "<spectra>", help = "MC set",)
parser.add_option("-b", "--bkg", type = 'string', action = "store", default = "FFT", metavar  = "<bkg>", help = "Background type: FFT with ISJ or sklearn with CV bandwidth",)
parser.add_option("-m", "--mass", type = float, action = "store", default = None, metavar  = "<mass>", help = "mass values: in case of specify only one value of mass will be input",)
parser.add_option("-u", "--up", type = float, action = "store", default = 100, metavar  = "<up>", help = "Dark Matter mass up",)
parser.add_option("-l", "--low", type = float, action = "store", default = 1, metavar  = "<low>", help = "Dark Matter mass low",)
parser.add_option("-n", "--n", type = int, action = "store", default = 100, metavar  = "<n>", help = "Dark Matter mass - N point scan",)
parser.add_option("--method", type = 'string', action = "store", default = "interpolate", metavar  = "<method>", help = "method for getting the UL: interpolate or bisection",)
parser.add_option("--nsample", type = int, action = "store", default = 0, metavar  = "<nsample>", help = "Sampling to make brazillian plot: 0 = no sampling",)
parser.add_option("--errorJ", type = 'string', action = "store", default = "nominal", metavar  = "<errorJ>", help = "Variance on Jfactor from Nesti&Salucci: nominal, errors1, errors2",)
parser.add_option("--exposure", type = float, action = "store", default = 2933.8*24*60*60, metavar  = "<exposure>", help = "exposure time (default: ~8 years)",)

parser.add_option("--gcinj", type = int, action = "store", default = 0, metavar  = "<gcinj>", help = "if gc is injected (0:no, 1:yes)",)
parser.add_option("--gcmodel", type = int, action = "store", default = 0, metavar  = "<gcmodel>", help = "if gc is accounted in the fit model (0:no, 1:yes)",)

(options, args) = parser.parse_args()


channel = options.channel
profile = options.profile
process = options.process
up = options.up
low = options.low
n = options.n
mc = options.mc
bkg = options.bkg
method = options.method
nsample = options.nsample
errorJ = options.errorJ
exposure = options.exposure

print(process)
print(profile)
print(channel)
print(nsample)

if options.gcinj==0:
    gcinj=False
else:
    gcinj=True

if options.gcmodel==0:
    gcmodel=False
else:
    gcmodel=True

Bin = Std_Binning(300, N_Etrue=100)
Reco = RecoRate(channel, 300, profile, Bin, process=process,type="Resp", spectra='Charon', set=mc)

if bkg=='FFT':
    Bkg = ScrambleBkg(Bin, bw="ISJ", oversample=10)
elif bkg=='sklearn':
    Bkg = ScrambleBkg(Bin, bw=0.03, method='sklearn' ,oversample=10)

# GC astro:
GCRate = GC_RecoRate(Bin, method='evtbyevt', set='1122', scrambled=False)[0]
GCRateScr = GC_RecoRate(Bin, method='evtbyevt', set='1122', scrambled=True)[0]

BurnSample = DataHist(Bin)
Ndata = 10*np.sum(BurnSample) # expected total number of data after 8 years

# Assuming the Scr Bkg from burn sample is the atm Bkg
BkgPDF = PdfBase(Bkg.flatten()/np.sum(Bkg.flatten()), name="BkgAtm")
GCPDF = PdfBase(GCRate.flatten()/np.sum(GCRate.flatten()), name="GCinData")
ScrGCPDF = PdfBase(GCRateScr.flatten()/np.sum(GCRateScr.flatten()), name="GCScr_inData")


# pseudodata and Scramble bkg in case of no signal yields:
if gcinj==True:
    gc_true = np.sum(GCRate)* exposure/(np.sum(Ndata))
else: gc_true = 0    
gc_inj = Parameter(value=gc_true, limits=(0,1), fixed=True, name="gc_inj")
ScrBkgPDF = gc_inj* ScrGCPDF + (1-gc_inj)* BkgPDF
DataPDF = gc_inj* GCPDF + (1-gc_inj)* BkgPDF


if nsample!=0:
        mean = np.array([])    
        low1 = np.array([])  
        low2 = np.array([])  
        up1 = np.array([])
        up2 = np.array([])
else:
    UL = np.array([])

# Manually load Jfactor in case for the error is considered:
if errorJ!='nominal':
    MyJ = Jf(profile=profile)
    J_Clumpy = MyJ.Jfactor_Clumpy(errors=errorJ)
    J_int = Interpolate_Jfactor(J_Clumpy, Bin['true_psi_center'])
    


masses = np.exp(np.linspace(np.log(low), np.log(up), n))
for mass in masses:
    # Bin
    if process=='ann': Etrue_max = mass
    if process=='decay': Etrue_max = mass/2.

    if Etrue_max < 3000:
        Bin = Std_Binning(Etrue_max, N_Etrue=300)
    else:
        Bin = Std_Binning(3000, N_Etrue=500)
    Reco.mass = mass
    Reco.bin = Bin
    
    Reco.Scramble = False
    if errorJ!='nominal':
        Reco.hist['Jfactor'] = J_int
    Rate = Reco.ComputeRecoRate()
    Reco.ResetAllHists()

    Reco.Scramble = True
    if errorJ!='nominal':
        Reco.hist['Jfactor'] = J_int
    Rate_Scr = Reco.ComputeRecoRate()
    Reco.ResetAllHists()
    BurnSample = DataHist(Bin)
    if nsample==0:
        UL = np.append(UL, UpperLimit(Rate, Rate_Scr, GCRate, GCRateScr, ScrBkgPDF, DataPDF, Ndata, gcmodel=gcmodel,
            exposure=exposure, method=method, sampling=False))
    else:
        UL_dist = np.array([])
        for i in range(nsample):
            UL_dist = np.append(UL_dist, UpperLimit(Rate, Rate_Scr,GCRate, GCRateScr, ScrBkgPDF, DataPDF, Ndata, gcmodel=gcmodel,
            exposure=exposure, method=method, sampling=True))
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
    if errorJ=='nominal':
        path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/UpperLimit/{}_{}_{}_{}points_MC{}_BKG{}_ULby{}_gcinj{}_gcmodel{}.pkl'.format(process, channel, profile, n, mc, bkg, method, gcinj, gcmodel)
    else:
        path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/UpperLimit/{}_{}_{}_{}points_MC{}_BKG{}_ULby{}_Jfactor{}.pkl'.format(process, channel, profile, n, mc, bkg, method, errorJ)
        
    outdict['UL'] = UL
    print('='*20)
    print('masses: {}'.format(masses))
    print('UL: {}'.format(UL))
    print('='*20)
else:
    path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/UpperLimit/{}_{}_{}_{}points_MC{}_BKG{}_ULby{}_nsample{}.pkl'.format(process, channel, profile, n, mc, bkg, method, nsample)

    outdict['mean'] = mean
    outdict['16'] = low1
    outdict['84'] = up1
    outdict['2.5'] = low2
    outdict['97.5'] = up2


pkl.dump(outdict, open(path, "wb"))