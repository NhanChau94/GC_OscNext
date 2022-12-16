#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python
import numpy as np
import sys, os
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

def TS_distribution(mass, channel, profile, MCset, Ntrial, signal_injection, SignalSub=True, Bkgtrial=True):

    print("Channel and mass: {}, {} GeV".format(channel, mass))
    print("MC set: {}".format(MCset))
    print("Signal sub: {}".format(SignalSub))
    print("Number of trials: {}".format(Ntrial))
    # First the signal pdf
    # channel and mass
    # Bin
    if mass<=1400:
        Bin = Std_Binning(mass)
    else:
        Bin = Std_Binning(1400)   
    
    Reco = RecoRate(channel, 
                mass, 
                profile, 
                Bin,
                set=MCset)
    Reco_Scr = RecoRate(channel, 
                mass, 
                profile, 
                Bin,
                Scramble=True,
                set=MCset)

    Rate = Reco.ComputeRecoRate()
    Rate_Scr = Reco_Scr.ComputeRecoRate()


    # Bkg from scramble data
    Bkg_bwISJ = ScrambleBkg(Bin, bw="ISJ", oversample=10)

    # Create PDF object
    SignalPDF = PdfBase(Rate.flatten()/np.sum(Rate.flatten()), name="SignalPDF")
    ScrSignalPDF = PdfBase(Rate_Scr.flatten()/np.sum(Rate_Scr.flatten()), name="ScrSignalPDF")
    BkgPDF = PdfBase(Bkg_bwISJ.flatten()/np.sum(Bkg_bwISJ.flatten()), name="Bkg")

    # Burn Sample:
    Burnsample = 10* DataHist(Bin)


    fsig = Parameter(value=0.5, limits=(0,1), fixed=False, name="f_sig")
    siginjec = Parameter(value=signal_injection, limits=(0,1), fixed=True, name="signal_injection")
    

    if SignalSub: 
        print("use signal subtraction likelihood")
        modelH1 = (fsig* SignalPDF) + (1-siginjec)*(BkgPDF) + siginjec* ScrSignalPDF -fsig*ScrSignalPDF
        modelH0 = (siginjec* SignalPDF) + (1-siginjec)*(BkgPDF)
    else:
        print("normal likelihood")
        modelH1 = fsig* SignalPDF + (1-fsig)* ((1-siginjec)*(BkgPDF) + siginjec* ScrSignalPDF)
        modelH0 = siginjec* SignalPDF + (1-siginjec)* ((1-siginjec)*(BkgPDF) + siginjec* ScrSignalPDF)

    print("----------------------- H1:")
    print(modelH1)
    print("----------------------- H0:")
    print(modelH0)    

    lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)
    ds = DataSet()
    if Bkgtrial:
        pseudodata = (1-siginjec)*(BkgPDF) + siginjec*BkgPDF
    else:
        pseudodata = (1-siginjec)*(BkgPDF) + siginjec*SignalPDF
    TSdist = np.array([])
    for i in range(Ntrial):
        # pseudo data as null model
        ds.sample(np.sum(Burnsample), pseudodata)
        lr.data = ds
        lr.fit("H1")
        lr.fit("H0")
        if lr.models['H1'].parameters["f_sig"].value > signal_injection:
            TSdist = np.append(TSdist, 0)
        else:    
            TSdist = np.append(TSdist, lr.TS)
    return TSdist


#----------------------------------------------------------------------------------------------------------------------
#Parser
#----------------------------------------------------------------------------------------------------------------------

parser = OptionParser()
# i/o options
parser.add_option("-m", "--mass", type = float, action = "store", default = 100, metavar  = "<mass>", help = "Dark Matter mass",)
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = 'string', action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("-s", "--set", type = "string", action = "store", default = '0000', metavar  = "<set>", help = "MC set",)
parser.add_option("-i", "--injection", type = float, action = "store", default = 0, metavar  = "<injection>", help = "signal injection",)
parser.add_option("-n", "--Ntrials", type = int, action = "store", default = 1000, metavar  = "<Ntrials>", help = "number of trials",)
parser.add_option("-f", "--file", type = "string", action = "store", default = 'None', metavar  = "<file>", help = "output file",)
parser.add_option("-b", "--bkgtrial", type = int, action = "store", default = 1, metavar  = "<bkgtrial>", help = "if use bkg trials",)

parser.add_option("-l", "--likelihood",
                  action="store_true", dest="likelihood", default=False,
                  help="use signal subtraction likelihood")

(options, args) = parser.parse_args()


mass = options.mass
channel = options.channel
profile = options.profile
set = options.set
injection = options.injection
Ntrials = options.Ntrials
likelihood = options.likelihood
file = options.file
bkgtrial = options.bkgtrial
if bkgtrial==1:
    TS = TS_distribution(mass, channel, profile, set, Ntrials, injection, SignalSub=likelihood, Bkgtrial=True)
else:
    TS = TS_distribution(mass, channel, profile, set, Ntrials, injection, SignalSub=likelihood, Bkgtrial=False)


path = '/data/user/tchau/Sandbox/GC_OscNext/Sensitivity/TSdist/TSdist_{}_{}_MC{}_Sig{}_SignalSubtraction{}_BkgTrial{}'.format(channel, mass, set, injection, likelihood, bkgtrial)
if not (os.path.exists(path)): os.makedirs(path)
outfile = "{}/{}.pkl".format(path, file)
pkl.dump(TS, open(outfile, "wb"))