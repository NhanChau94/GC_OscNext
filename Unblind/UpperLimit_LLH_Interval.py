#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python

import sys, os
import numpy as np
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
from Background import *
from Jfactor import *

from modeling import PdfBase, Model, Parameter
from data import DataSet
from llh import LikelihoodRatioTest


#############################################################################################################
#   Compute upper limit from DM expectation, Scramble Bkg and Data
#
def UpperLimit(DMRate, DMRateScr, ScrBkgPDF, DataPDF, Ndata, GPmodel='None',
               llh="SignalSub", exposure= 9.3* 365.25 *24*60*60, sampling=False, process='ann', fixGP=True):

    #############################################################################################################
    #   Create the signal PDF and itinitiate signal fraction parameter
    #

    SignalPDF = PdfBase(DMRate.flatten()/np.sum(DMRate.flatten()), name="SignalPDF")
    ScrSignalPDF = PdfBase(DMRateScr.flatten()/np.sum(DMRateScr.flatten()), name="ScrSignalPDF")     
    dm_H1 = Parameter(value=0., limits=(0,1), fixed=False, name="dm_H1")
    dm_H0 = Parameter(value=0., limits=(0,1), fixed=True, name="dm_H0")
    f = Parameter(value=1., limits=(0,1), fixed=True, name="factor 1.")
    #############################################################################################################
    #   In case of including Galactic Plane in the LLR
    #
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
            modelH0 = dm_H0* SignalPDF + gc_H0* GPPDF_model + f*ScrBkgPDF - dm_H0* ScrSignalPDF - gc_H0* ScrGPPDF_model
            modelH1 = dm_H1* SignalPDF + gc_H1* GPPDF_model + f*ScrBkgPDF - dm_H1* ScrSignalPDF - gc_H1* ScrGPPDF_model
        else:
            modelH0 = dm_H0* SignalPDF + gc_H0* GPPDF_model + (1-dm_H0-gc_H0)*ScrBkgPDF
            modelH1 = dm_H1* SignalPDF + gc_H1* GPPDF_model + (1-dm_H1-gc_H1)*ScrBkgPDF

    else:
        if llh=="SignalSub":
            modelH0 = dm_H0* SignalPDF + f*ScrBkgPDF - dm_H0* ScrSignalPDF
            modelH1 = dm_H1* SignalPDF + f*ScrBkgPDF - dm_H1* ScrSignalPDF
        else:
            modelH0 = dm_H0* SignalPDF + (1-dm_H0)*ScrBkgPDF
            modelH1 = dm_H1* SignalPDF + (1-dm_H1)*ScrBkgPDF

    #############################################################################################################
    #   Create LLR models, parsing the data set (asimov or resampling) then compute 90%CL limit of signal fraction
    #
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
parser.add_option("--process", type = 'string', action = "store", default = "ann", metavar  = "<process>", help = "process: ann or decay",)
parser.add_option("-s", "--spectra", type = 'string', action = "store", default = "Charon", metavar  = "<spectra>", help = "Spectra: Charon or PPPC4",)
parser.add_option("--mc", type = 'string', action = "store", default = "1122", metavar  = "<spectra>", help = "MC set",)

parser.add_option("-u", "--up", type = float, action = "store", default = 100, metavar  = "<up>", help = "Dark Matter mass upper value for scanning",)
parser.add_option("-l", "--low", type = float, action = "store", default = 100, metavar  = "<low>", help = "Dark Matter mass lower value for scanning",)
parser.add_option("-n", "--n", type = int, action = "store", default = 1, metavar  = "<n>", help = "# of point scan on DM mass from defined lower to upper value",)
parser.add_option("--errorJ", type = 'string', action = "store", default = "nominal", metavar  = "<errorJ>", help = "Variance on Jfactor from Nesti&Salucci: nominal, errors1, errors2",)
parser.add_option("--exposure", type = float, action = "store", default = 9.3* 365.25 *24*60*60, metavar  = "<exposure>", help = "exposure time (default: 9.3 years)",)

parser.add_option("--gcmodel", type = 'string', action = "store", default = 'None', metavar  = "<gcmodel>", help = "GP astro in the LLR, use among: pi0, pi0_IC, KRA50, KRA50_IC, KRA5, KRA5_IC, None",)
parser.add_option("--fixGP", type = int, action = "store", default = 1, metavar  = "<fixGP>", help = "in case of including GP, fix its fraction (1) or fit/marginalize it (0)",)

parser.add_option("--mass_default",
                  action="store_true", dest="mass_default", default=False,
                  help="Set mass scan values to the default used for the analysis as indicated in the wikipage")  

(options, args) = parser.parse_args()


channel = options.channel
profile = options.profile
process = options.process
up = options.up
low = options.low
n = options.n
mc = options.mc
errorJ = options.errorJ
exposure = options.exposure
fixGP = options.fixGP
GPmodel = options.gcmodel
mass_default = options.mass_default

#############################################################################################################
#   0 - Defined default mass range for each channels
#
if mass_default:
    if process=='ann':
        masses = {"WW":[90, 8000], "bb":[15, 8000], 'tautau':[5, 4000], 'mumu':[5, 1000], "nuenue":[5, 200],"numunumu":[5, 200],"nutaunutau":[5,200]}
    elif process=='decay':    
        masses = {"WW":[180, 8000], "bb":[30, 8000], 'tautau':[5, 8000], 'mumu':[5, 2000], "nuenue":[5, 400],"numunumu":[5, 400],"nutaunutau":[5,400]}
    up = masses[channel][1]
    low = masses[channel][0]
    n = 30

#############################################################################################################
#   1 -  Binning scheme, Signal expectation object, Bkg PDF, Data
#
Bin = Std_Binning(300) # just a dummy mass value 300GeV which later will be scanned
Reco = RecoRate(channel, 300, profile, Bin, process=process,type="Resp", spectra='Charon', set=mc)



# For bkg from data: create 100 kde estimation with 100 different seed -> average over them
# or kde on 100 RA scrambling sample?
# Bkg = ScrambleBkg(Bin, bandwidth="ISJ", oversample=10, sample=bkg)
Nsample = 1
np.random.seed(2023)
array_seed = np.random.randint(0,2023, size=Nsample)
Bkg = ScrambleBkg(Bin, bandwidth="ISJ", oversample=100, sample='data', seed=int(array_seed[0]), kde=True)
for s in array_seed[1:Nsample]:
    Bkg += ScrambleBkg(Bin, bandwidth="ISJ", oversample=100, sample='data', seed=int(s), kde=True)
Bkg = Bkg/Nsample

# load data
data_hist = DataHist(Bin, sample='data')
Ndata = np.sum(data_hist)
DataPDF = PdfBase(Bkg.flatten()/np.sum(Bkg.flatten()), name="data")
# DataPDF = PdfBase(data_hist.flatten()/np.sum(data_hist.flatten()), name="actual_data")

# Mask out zero data bin
# loc = np.where(data_hist==0)
# Bkg[loc] = 0



# Scr bkg pdf
ScrBkgPDF = PdfBase(Bkg.flatten()/np.sum(Bkg.flatten()), name="Scramble_bkg_pdf")




UL = np.array([])
fraction = np.array([])

# Manually load the error Jfactor in case it is considered:
if errorJ!='nominal':
    MyJ = Jf(profile=profile, process=process)
    J_Clumpy = MyJ.Jfactor_Clumpy(errors=errorJ)
    J_int = Interpolate_Jfactor(J_Clumpy, Bin['true_psi_center'])
    

#############################################################################################################
#   2 -  scan mass range, 
#        for each mass compute corresponing signal PDF and pass to the LLR model for limit estimation
#
masses = np.exp(np.linspace(np.log(low), np.log(up), n))
for mass in masses:
    if process=='ann': Etrue_max = mass
    if process=='decay': Etrue_max = mass/2.

    if Etrue_max < 3000:
        Bin = Std_Binning(Etrue_max)
    else:
        Bin = Std_Binning(3000)
    
    # if Etrue_max < 2000:
    #     Bin = Std_Binning(Etrue_max, N_Etrue=300)
    # else:
    #     Bin = Std_Binning(2000, N_Etrue=500)

    # Signal PDF
    Reco.mass = mass
    Reco.bin = Bin    
    Reco.Scramble = False
    if errorJ!='nominal':
        Reco.hist['Jfactor'] = J_int
    Rate = Reco.ComputeRecoRate()
    Reco.ResetAllHists()

    # Scrambled Signal PDF
    Reco.Scramble = True
    if errorJ!='nominal':
        Reco.hist['Jfactor'] = J_int
    Rate_Scr = Reco.ComputeRecoRate()
    Reco.ResetAllHists()


    limit, frac = UpperLimit(Rate, Rate_Scr, ScrBkgPDF, DataPDF, Ndata, GPmodel=GPmodel,
        exposure=exposure, sampling=False, process=process, fixGP=bool(fixGP)) 
    UL = np.append(UL, limit)
    fraction = np.append(fraction, frac)


outdict = dict()
outdict['mass'] = masses

if errorJ=='nominal':
    path = '{}/UpperLimit/{}_{}_{}_{}points_MC{}_unblind_gcmodel{}_fixgc{}.pkl'.format(output_path, process, channel, profile, n, mc, GPmodel, fixGP)
else:
    path = '{}/UpperLimit/{}_{}_{}_{}points_MC{}_unblind_Jfactor{}.pkl'.format(output_path, process, channel, profile, n, mc, errorJ)
    
outdict['UL'] = UL
outdict['fraction'] = fraction

# pkl.dump(outdict, open(path, "wb"))
print('='*20)
print('masses: {}'.format(masses))
print('UL: {}'.format(UL))
print('signal fraction:{}'.format(fraction))
print('='*20)
