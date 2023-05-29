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

def UpperLimit(DMRate, DMRateScr, ScrBkgPDF, DataPDF, Ndata, GPmodel='None',
               llh="SignalSub", exposure= 2933.8*24*60*60, sampling=False, process='ann', fixGP=True):

    # Create the PDF object
    SignalPDF = PdfBase(DMRate.flatten()/np.sum(DMRate.flatten()), name="SignalPDF")
    ScrSignalPDF = PdfBase(DMRateScr.flatten()/np.sum(DMRateScr.flatten()), name="ScrSignalPDF")
      
    dm_H1 = Parameter(value=0., limits=(0,1), fixed=False, name="dm_H1")
    dm_H0 = Parameter(value=0., limits=(0,1), fixed=True, name="dm_H0")

    if GPmodel!=None and GPmodel!='None':
        scales = {'pi0':1, 'pi0_IC':4.95, 'KRA50':1., 'KRA50_IC':0.37, 'KRA5':1., 'KRA5_IC':0.55}
        template = {'pi0':'Fermi-LAT_pi0_map.npy', 'pi0_IC':'Fermi-LAT_pi0_map.npy', 
                    'KRA50':'KRA-gamma_maps_energies.tuple.npy', 'KRA50_IC':'KRA-gamma_maps_energies.tuple.npy', 
                    'KRA5':'KRA-gamma_5PeV_maps_energies.tuple.npy', 'KRA5_IC':'KRA-gamma_5PeV_maps_energies.tuple.npy'}
    
        GPRate_model = GP_RecoRate(Bin, template='/data/user/tchau/DarkMatter_OscNext/GP_template/'+template[GPmodel], scale=scales[GPmodel])
        GPRate_model_scr = GP_RecoRate(Bin, template='/data/user/tchau/DarkMatter_OscNext/GP_template/'+template[GPmodel], scale=scales[GPmodel], scrambled=True)

        GPPDF_model = PdfBase(GPRate_model.flatten()/np.sum(GPRate_model.flatten()), name="GC_model")
        ScrGPPDF_model = PdfBase(GPRate_model_scr.flatten()/np.sum(GPRate_model_scr.flatten()), name="GCScr_model")

        gc_model = np.sum(GPRate_model)*exposure/Ndata
        gc_H1 = Parameter(value=gc_model, limits=(gc_model* (1.-0.95), gc_model* (1.+0.95)), fixed=fixGP, name="gc_H1")
        gc_H0 = Parameter(value=gc_model, limits=(0,1), fixed=fixGP, name="gc_H0")

        # Add the GP to the LLR model
        if llh=="SignalSub":
            modelH0 = dm_H0* SignalPDF + gc_H0* GPPDF_model + ScrBkgPDF - dm_H0* ScrSignalPDF - gc_H0* ScrGPPDF_model
            modelH1 = dm_H1* SignalPDF + gc_H1* GPPDF_model + ScrBkgPDF - dm_H1* ScrSignalPDF - gc_H1* ScrGPPDF_model
        else:
            modelH0 = dm_H0* SignalPDF + gc_H0* GPPDF_model + (1-dm_H0-gc_H0)*ScrBkgPDF
            modelH1 = dm_H1* SignalPDF + gc_H1* GPPDF_model + (1-dm_H1-gc_H1)*ScrBkgPDF

    else:
        if llh=="SignalSub":
            modelH0 = dm_H0* SignalPDF + ScrBkgPDF - dm_H0* ScrSignalPDF
            modelH1 = dm_H1* SignalPDF + ScrBkgPDF - dm_H1* ScrSignalPDF
        else:
            modelH0 = dm_H0* SignalPDF + (1-dm_H0)*ScrBkgPDF
            modelH1 = dm_H1* SignalPDF + (1-dm_H1)*ScrBkgPDF


    lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)
    data = DataSet()
    if sampling==False:
        data.asimov(Ndata, DataPDF)
    else:
        data.sample(Ndata, DataPDF)
    lr.data = data

    xi_CL = lr.upperlimit_llhinterval('dm_H1', 'dm_H0', 90)  

    print('='*20)    
    print('signal fraction: {}'.format(xi_CL))
    lr.models['H0'].parameters["dm_H0"].value = xi_CL
    lr.fit("H0")
    lr.fit("H1")
    print('TS value at the output upper limit: {}'.format(lr.TS))

    # Convert to thermal cross-section/lifetime:
    Nsignal = xi_CL* Ndata
    sigma = Nsignal/(np.sum(DMRate*exposure))
    if process=='decay':
        sigma = 1./sigma
    return sigma, xi_CL




parser = OptionParser()
# i/o options
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = 'string', action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("--process", type = 'string', action = "store", default = "decay", metavar  = "<process>", help = "process: ann or decay",)
parser.add_option("-s", "--spectra", type = 'string', action = "store", default = "Charon", metavar  = "<spectra>", help = "Spectra: Charon or PPPC4",)
parser.add_option("--mc", type = 'string', action = "store", default = "0000", metavar  = "<spectra>", help = "MC set",)
parser.add_option("-b", "--bkg", type = 'string', action = "store", default = "FFT", metavar  = "<bkg>", help = "Background type: FFT with ISJ or sklearn with CV bandwidth",)
parser.add_option("-m", "--mass", type = float, action = "store", default = None, metavar  = "<mass>", help = "mass values: in case of specify only one value of mass will be input",)
parser.add_option("-u", "--up", type = float, action = "store", default = 100, metavar  = "<up>", help = "Dark Matter mass up",)
parser.add_option("-l", "--low", type = float, action = "store", default = 1, metavar  = "<low>", help = "Dark Matter mass low",)
parser.add_option("-n", "--n", type = int, action = "store", default = 100, metavar  = "<n>", help = "Dark Matter mass - N point scan",)
parser.add_option("--nsample", type = int, action = "store", default = 0, metavar  = "<nsample>", help = "Sampling to make brazillian plot: 0 = no sampling",)
parser.add_option("--errorJ", type = 'string', action = "store", default = "nominal", metavar  = "<errorJ>", help = "Variance on Jfactor from Nesti&Salucci: nominal, errors1, errors2",)
parser.add_option("--exposure", type = float, action = "store", default = 9.3* 365.25 *24*60*60, metavar  = "<exposure>", help = "exposure time (default: 9.3 years)",)

parser.add_option("--gcinj", type = 'string', action = "store", default = 'None', metavar  = "<gcinj>", help = "if gc is injected (0:no, 1:yes)",)
parser.add_option("--gcmodel", type = 'string', action = "store", default = 'None', metavar  = "<gcmodel>", help = "if gc is accounted in the fit model (0:no, 1:yes)",)
parser.add_option("--fixGP", type = int, action = "store", default = 1, metavar  = "<fixGP>", help = "in case of including GP, fix it (1) or fit/marginalize it (0)",)

(options, args) = parser.parse_args()


channel = options.channel
profile = options.profile
process = options.process
up = options.up
low = options.low
n = options.n
mc = options.mc
bkg = options.bkg
nsample = options.nsample
errorJ = options.errorJ
exposure = options.exposure
fixGP = options.fixGP
GPinj = options.gcinj
GPmodel = options.gcmodel

print(process)
print(profile)
print(channel)
print(nsample)


Bin = Std_Binning(300, N_Etrue=100)
# Reco = RecoRate(channel, 300, profile, Bin, process=process,type="Resp", spectra='Charon', set=mc)
Reco = RecoRate(channel, 300, profile, Bin, process=process,type="Resp", spectra='Charon', set=mc)

if bkg=='FFT':
    Bkg = ScrambleBkg(Bin, bandwidth="ISJ", oversample=10)
elif bkg=='sklearn':
    Bkg = ScrambleBkg(Bin, bandwidth=0.03, method='sklearn' ,oversample=10)
elif bkg=='precomp':
    loadpdf = pkl.load(open('/data/user/tchau/DarkMatter_OscNext/PDFs/Background/RAScramble_burnsample_FFTkde.pkl', 'rb'))
    Bkg = loadpdf['pdf']

BurnSample = DataHist(Bin)
Ndata = 10*np.sum(BurnSample) # expected total number of data after 8 years

# Assuming the Scr Bkg from burn sample is the atm Bkg
BkgPDF = PdfBase(Bkg.flatten()/np.sum(Bkg.flatten()), name="BkgAtm")


# in case including GP in the model:
# Different models for GP
scales = {'pi0':1, 'pi0_IC':4.95, 'KRA50':1., 'KRA50_IC':0.37, 'KRA5':1., 'KRA5_IC':0.55}
template = {'pi0':'Fermi-LAT_pi0_map.npy', 'pi0_IC':'Fermi-LAT_pi0_map.npy', 
                'KRA50':'KRA-gamma_maps_energies.tuple.npy', 'KRA50_IC':'KRA-gamma_maps_energies.tuple.npy', 
                'KRA5':'KRA-gamma_5PeV_maps_energies.tuple.npy', 'KRA5_IC':'KRA-gamma_5PeV_maps_energies.tuple.npy'}

if GPinj!=None and GPinj!='None':

    GPRate_inj = GP_RecoRate(Bin, template='/data/user/tchau/DarkMatter_OscNext/GP_template/'+template[GPinj], scale=scales[GPinj])
    GPRate_inj_scr = GP_RecoRate(Bin, template='/data/user/tchau/DarkMatter_OscNext/GP_template/'+template[GPinj], scale=scales[GPinj], scrambled=True)

    GPPDF_inj = PdfBase(GPRate_inj.flatten()/np.sum(GPRate_inj.flatten()), name="GC_inj")
    ScrGPPDF_inj = PdfBase(GPRate_inj_scr.flatten()/np.sum(GPRate_inj_scr.flatten()), name="GCScr_inj")

    gc_true = np.sum(GPRate_inj)*exposure/Ndata
    gc_inj = Parameter(value=gc_true, limits=(0,1), fixed=True, name="gc_inj")
    DataPDF = gc_inj* GPPDF_inj + (1-gc_inj)* BkgPDF
    ScrBkgPDF =  gc_inj* ScrGPPDF_inj + (1-gc_inj)* BkgPDF

else:
    f = Parameter(value=1., limits=(0,1), fixed=True, name="factor")

    DataPDF = f*BkgPDF
    ScrBkgPDF = f*BkgPDF

if nsample!=0:
        median = np.array([])    
        low1 = np.array([])  
        low2 = np.array([])  
        up1 = np.array([])
        up2 = np.array([])
else:
    UL = np.array([])
    fraction = np.array([])

# Manually load Jfactor in case for the error is considered:
if errorJ!='nominal':
    MyJ = Jf(profile=profile, process=process)
    J_Clumpy = MyJ.Jfactor_Clumpy(errors=errorJ)
    J_int = Interpolate_Jfactor(J_Clumpy, Bin['true_psi_center'])
    

masses = np.exp(np.linspace(np.log(low), np.log(up), n))
for mass in masses:
    # Bin
    if process=='ann': Etrue_max = mass
    if process=='decay': Etrue_max = mass/2.

    # if Etrue_max < 3000:
    #     Bin = Std_Binning(Etrue_max, N_Etrue=300)
    # else:
    #     Bin = Std_Binning(3000, N_Etrue=500)
    
    if Etrue_max < 2000:
        Bin = Std_Binning(Etrue_max, N_Etrue=300)
    else:
        Bin = Std_Binning(2000, N_Etrue=500)

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


    if nsample==0:
        limit, frac = UpperLimit(Rate, Rate_Scr, ScrBkgPDF, DataPDF, Ndata, GPmodel=GPmodel,
            exposure=exposure, sampling=False, process=process, fixGP=bool(fixGP)) 
        UL = np.append(UL, limit)
        fraction = np.append(fraction, frac)
    else:
        UL_dist = np.array([])
        for i in range(nsample):
            UL_dist = np.append(UL_dist, UpperLimit(Rate, Rate_Scr, ScrBkgPDF, DataPDF, Ndata, GPmodel=GPmodel,
            exposure=exposure, sampling=True, process=process, fixGP=bool(fixGP))[0] )
        arr1 = np.percentile(UL_dist, [2.5, 50, 97.5])
        arr2 = np.percentile(UL_dist, [16, 50, 84])

        #Compute 1 and 2 sigma bands

        median = np.append(median, arr1[1])
        low1 = np.append(low1, arr1[0])
        up1 = np.append(up1, arr1[2])
        low2 = np.append(low2, arr2[0])
        up2 = np.append(up2, arr2[2])    


outdict = dict()
outdict['mass'] = masses
if nsample==0:
    if errorJ=='nominal':
        path = '/data/user/tchau/DarkMatter_OscNext/Sensitivity/UpperLimit/{}_{}_{}_{}points_MC{}_BKG{}_gcinj{}_gcmodel{}_fixgc{}.pkl'.format(process, channel, profile, n, mc, bkg, GPinj, GPmodel, fixGP)
    else:
        path = '/data/user/tchau/DarkMatter_OscNext/Sensitivity/UpperLimit/{}_{}_{}_{}points_MC{}_BKG{}_Jfactor{}_fixgc{}.pkl'.format(process, channel, profile, n, mc, bkg, errorJ, fixGP)
        
    outdict['UL'] = UL
    outdict['fraction'] = fraction

    print('='*20)
    print('masses: {}'.format(masses))
    print('UL: {}'.format(UL))
    print('signal fraction:{}'.format(fraction))
    print('='*20)
else:
    path = '/data/user/tchau/DarkMatter_OscNext/Sensitivity/UpperLimit/{}_{}_{}_{}points_MC{}_BKG{}_nsample{}_gcinj{}_gcmodel{}_fixgc{}.pkl'.format(process, channel, profile, n, mc, bkg, nsample, GPinj, GPmodel, fixGP)

    outdict['median'] = median
    outdict['16'] = low1
    outdict['84'] = up1
    outdict['2.5'] = low2
    outdict['97.5'] = up2


pkl.dump(outdict, open(path, "wb"))