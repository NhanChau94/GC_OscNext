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

from modeling import PdfBase, Model, Parameter
from data import DataSet
from llh import LikelihoodRatioTest

# Function return the TS distribution
def TS_distribution(mass, channel, profile, process, mcfit, mcinj, Ntrial, SignalSub=True, Bkgtrial=True, 
                    GPmodel=None, GPinj=None, fixGP=True, null=False):

    print("Channel and mass: {}, {} GeV".format(channel, mass))
    print("MC set for model: {}".format(mcfit))
    print("MC set for injection: {}".format(mcinj))
    print("Signal sub: {}".format(SignalSub))
    print("Number of trials: {}".format(Ntrial))
    print("parsing value for process: {}".format(process))

#############################################################################################################
#   0 - Define process (annihilation/decay) and binning scheme 
#

    Etrue_max = mass
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
    Reco.Scramble = False
    DMRate = Reco.ComputeRecoRate()
    Reco.ResetAllHists()
    Reco.Scramble = True
    DMRateScr=Reco.ComputeRecoRate()
    Reco.ResetAllHists()

#############################################################################################################
#   2 - Background rate assuming data ~ 10xBurnSample
#
    # Bkg 
    exposure = 8* 365.*24.* 60.* 60.
    Bkg = ScrambleBkg(Bin, bandwidth="ISJ", oversample=10)
    BurnSample = DataHist(Bin)
    Ndata = 10*np.sum(BurnSample) # expected total number of data after 8 years
    BkgRate = 10*np.sum(BurnSample)*Bkg/(np.sum(Bkg))/(exposure)


#############################################################################################################
#   3 - Create PDF object, fraction parameters, and LLR models
#

    # Signal
    SignalPDF = PdfBase(DMRate.flatten()/np.sum(DMRate.flatten()), name="SignalPDF")
    ScrSignalPDF = PdfBase(DMRateScr.flatten()/np.sum(DMRateScr.flatten()), name="ScrSignalPDF")

    if mcinj!=mcfit:
        Reco.set=mcinj
        Reco.Scramble = False
        DMRate_inj=Reco.ComputeRecoRate()
        Reco.ResetAllHists()

        Reco.Scramble = True
        DMRateScr_inj=Reco.ComputeRecoRate()
        SignalPDF_inj = PdfBase(DMRate_inj.flatten()/np.sum(DMRate_inj.flatten()), name="SignalPDF_inj")
        ScrSignalPDF_inj = PdfBase(DMRateScr_inj.flatten()/np.sum(DMRateScr_inj.flatten()), name="ScrSignalPDF_inj")
    else:
        SignalPDF_inj = SignalPDF
        ScrSignalPDF_inj = ScrSignalPDF

    # Assuming the Scr Bkg from burn sample is the atm Bkg
    BkgPDF = PdfBase(BkgRate.flatten()/np.sum(BkgRate.flatten()), name="BkgAtm")

    # Signal fraction parameters in: injection, H1 hypothesis, H0 hypothesis
    dm_inj = Parameter(value=0., limits=(0,1), fixed=True, name="dm_inj")
    # use for model fitting:
    dm_H1 = Parameter(value=0., limits=(0,1), fixed=False, name="dm_H1")
    dm_H0 = Parameter(value=0., limits=(0,1), fixed=True, name="dm_H0")

    pseudo_data = dm_inj* SignalPDF_inj + (1-dm_inj)* BkgPDF
    # Scramble bkg now yields:
    ScrBkgPDF = dm_inj* ScrSignalPDF_inj + (1-dm_inj)* BkgPDF
    if SignalSub==True:
        modelH0 = dm_H0* SignalPDF + ScrBkgPDF - dm_H0* ScrSignalPDF
        modelH1 = dm_H1* SignalPDF + ScrBkgPDF - dm_H1* ScrSignalPDF
    else:
        modelH0 = dm_H0* SignalPDF + (1-dm_H0)*ScrBkgPDF
        modelH1 = dm_H1* SignalPDF + (1-dm_H1)*ScrBkgPDF

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
        pseudo_data = dm_inj* SignalPDF + gc_inj* GPPDF_inj + (1-dm_inj-gc_inj)* BkgPDF
        ScrBkgPDF = dm_inj* ScrSignalPDF + gc_inj* ScrGPPDF_inj + (1-dm_inj-gc_inj)* BkgPDF

    if GPmodel!=None and GPmodel!='None':
        # models = ['pi0', 'pi0_IC', 'KRA50', 'KRA50_IC', 'KRA5', 'KRA5_IC']
        
        GPRate_model = GP_RecoRate(Bin, template='/data/user/tchau/DarkMatter_OscNext/GP_template/'+template[GPmodel], scale=scales[GPmodel])
        GPRate_model_scr = GP_RecoRate(Bin, template='/data/user/tchau/DarkMatter_OscNext/GP_template/'+template[GPmodel], scale=scales[GPmodel], scrambled=True)

        GPPDF_model = PdfBase(GPRate_model.flatten()/np.sum(GPRate_model.flatten()), name="GC_model")
        ScrGPPDF_model = PdfBase(GPRate_model_scr.flatten()/np.sum(GPRate_model_scr.flatten()), name="GCScr_model")

        gc_model = np.sum(GPRate_model)*exposure/Ndata
        gc_H1 = Parameter(value=gc_model, limits=(gc_model* (1.-0.95), gc_model* (1.+0.95)), fixed=fixGP, name="gc_H1")
        gc_H0 = Parameter(value=gc_model, limits=(0,1), fixed=fixGP, name="gc_H0")

        # Add the GP to the LLR model
        if SignalSub==True:
            modelH0 = dm_H0* SignalPDF + gc_H0* GPPDF_model + ScrBkgPDF - dm_H0* ScrSignalPDF - gc_H0* ScrGPPDF_model
            modelH1 = dm_H1* SignalPDF + gc_H1* GPPDF_model + ScrBkgPDF - dm_H1* ScrSignalPDF - gc_H1* ScrGPPDF_model
        else:
            modelH0 = dm_H0* SignalPDF + gc_H0* GPPDF_model + (1-dm_H0-gc_H0)*ScrBkgPDF
            modelH1 = dm_H1* SignalPDF + gc_H1* GPPDF_model + (1-dm_H1-gc_H1)*ScrBkgPDF


    # Create asimov/median dataset and the LLR model object
    lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)
    ds = DataSet()
    ds.asimov(Ndata, pseudo_data)
    lr.data = ds
    if null: # in case H0 and data is signal-free
        pseudo_data.parameters["dm_inj"].value = 0.
        lr.models['H1'].parameters["dm_inj"].value = 0.
        lr.models['H0'].parameters["dm_inj"].value = 0.
        lr.models['H0'].parameters["dm_H0"].value = 0.
        TSdist = np.array([])
        for i in range(Ntrial):
            ds.sample(Ndata, pseudo_data)
            lr.data = ds
            lr.fit('H1')
            lr.fit('H0')
            TSdist = np.append(TSdist, lr.TS)
    else: # assuming 90%CL injection to H0 and data
        ul = lr.upperlimit_llhinterval('dm_H1', 'dm_H0', 90)
        if Bkgtrial==True:
            dm_trial = 0.
        else:
            dm_trial = ul
        pseudo_data.parameters["dm_inj"].value = dm_trial
        lr.models['H1'].parameters["dm_inj"].value = dm_trial
        lr.models['H0'].parameters["dm_inj"].value = dm_trial

        lr.models['H0'].parameters["dm_H0"].value = ul

        TSdist = np.array([])
        for i in range(Ntrial):
            ds.sample(Ndata, pseudo_data)
            lr.data = ds
            TSdist = np.append(TSdist, lr.TS_llhinterval(ul, 'dm_H1', 'dm_H0'))
    return TSdist


#----------------------------------------------------------------------------------------------------------------------
#Parser
#----------------------------------------------------------------------------------------------------------------------

parser = OptionParser()
# i/o options
parser.add_option("-m", "--mass", type = float, action = "store", default = 100, metavar  = "<mass>", help = "Dark Matter mass",)
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = "string", action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("--process", type = "string", action = "store", default = "ann", metavar  = "<process>", help = "ann or decay process",)
parser.add_option("--mcfit", type = "string", action = "store", default = '1122', metavar  = "<mcfit>", help = "MC set use in model",)
parser.add_option("--mcinj", type = "string", action = "store", default = '1122', metavar  = "<mcinj>", help = "MC set use in injection",)
parser.add_option("-n", "--Ntrials", type = int, action = "store", default = 1000, metavar  = "<Ntrials>", help = "number of trials",)
parser.add_option("-f", "--file", type = "string", action = "store", default = None, metavar  = "<file>", help = "output file",)
parser.add_option("-b", "--bkgtrial", type = int, action = "store", default = 0, metavar  = "<bkgtrial>", help = "if use bkg trials",)

parser.add_option("--signalsub",
                  action="store_true", dest="signalsub", default=False,
                  help="use signal subtraction likelihood")

parser.add_option("--null",
                  action="store_true", dest="null", default=False,
                  help="compute -2LLR for null hypothesis i.e: TS(0|0), otherwise compute TS for the checking case of loglikelihood interval")                  


parser.add_option("--GPmodel", type = "string", action = "store", default = None, metavar  = "<GPmodel>", help = "GP model in the fit, use among: pi0, pi0_IC, KRA50, KRA50_IC, KRA5, KRA5_IC, None",)
parser.add_option("--GPinject", type = "string", action = "store", default = None, metavar  = "<GPinject>", help = "GP model used for injection, use amog:  pi0, pi0_IC, KRA50, KRA50_IC, KRA5, KRA5_IC, None",)
parser.add_option("--fixGP", type = int, action = "store", default = 1, metavar  = "<fixGP>", help = "in case of including GP, fix its fraction (1) or fit/marginalize it (0)",)



(options, args) = parser.parse_args()


mass = options.mass
channel = options.channel
profile = options.profile
mcfit = options.mcfit
mcinj = options.mcinj
Ntrials = options.Ntrials
signalsub = options.signalsub
file = options.file
bkgtrial = options.bkgtrial
process=options.process
null = options.null
GPmodel = options.GPmodel
GPinject = options.GPinject
fixGP = options.fixGP

TS = TS_distribution(mass, channel, profile, process, mcfit, mcinj, Ntrials, SignalSub=signalsub, Bkgtrial=bool(bkgtrial), GPmodel=GPmodel, GPinj=GPinject, fixGP=bool(fixGP), null=null)


if null==False:
    path = '{}/TSdist/TSdist_{}_{}_{}_{}_MCfit{}_Mcinj{}_SignalSubtraction{}_BkgTrial{}'.format(output_path, channel, mass, process, profile, mcfit, mcinj, signalsub, bkgtrial)
else:
    path = '{}/TSdist/TSdist_{}_{}_{}_{}_MCfit{}_Mcinj{}_SignalSubtraction{}_null'.format(output_path, channel, mass, process, profile, mcfit, mcinj, signalsub)

if GPinject!=None and GPinject!='None':
    path += f'_GPinject{GPinject}'
if GPmodel!=None and GPinject!='None':
    path += f'_GPmodel{GPmodel}'

if not (os.path.exists(path)): os.makedirs(path)
outfile = "{}/{}.pkl".format(path, file)
pkl.dump(TS, open(outfile, "wb"))