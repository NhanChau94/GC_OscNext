#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python

import sys, os
import numpy as np
import pickle as pkl
from optparse import OptionParser

base_path=os.getenv('GC_DM_BASE')
data_path=os.getenv('GC_DM_DATA')
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

parser = OptionParser()
# i/o options
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = 'string', action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("--process", type = 'string', action = "store", default = "ann", metavar  = "<process>", help = "process: ann or decay",)
parser.add_option("--bkg", type = 'string', action = "store", default = "FFT", metavar  = "<bkg>", help = "bkg type",)
parser.add_option("--sample", type = 'string', action = "store", default = "null_hypothesis", metavar  = "<sample>", help = "sample type: RA null_hypothesis i.e RA_scramble or poissonian fluctuation of background",)
parser.add_option("--mass", type = float, action = "store", default = 100, metavar  = "<mass>", help = "Dark Matter mass",)
# parser.add_option("--oversample", type = int, action = "store", default = 100, metavar  = "<oversample>", help = "number of oversample for bkg estimation",)

parser.add_option("--ntrials", type = int, action = "store", default = 10000, metavar  = "<ntrials>", help = "number of trials used for TS distribution",)
parser.add_option("--output_path", type = 'string', action = "store", default = '', metavar  = "<output_path>", help = "output path",)



(options, args) = parser.parse_args()


channel = options.channel
profile = options.profile
process = options.process
mass = options.mass
bkg = options.bkg
sample = options.sample
ntrials = options.ntrials
output_path = options.output_path

# Bin, Signal PDF
if mass<=3000:
    Bin = Std_Binning(mass)
else:
    Bin = Std_Binning(3000)

Reco = RecoRate(channel, mass, profile, Bin, type="Resp", set='1122', spectra='Charon', process=process)
DMRate = Reco.ComputeRecoRate()

Reco.Scramble = True
Reco.hist['Resp'] = None
DMRateScr = Reco.ComputeRecoRate()

# Bkg estimation
if bkg=='FFT':
    Bkg = ScrambleBkg(Bin, bandwidth="ISJ", oversample=100, sample='data', seed=2023, kde=True)
elif bkg=='sklearn':
    Bkg = ScrambleBkg(Bin, bandwidth=0.03, method='sklearn', oversample=100, sample='data', seed=2023, kde=True)
elif bkg=='nokde':
    Nsample = 100
    np.random.seed(2023)
    array_seed = np.random.randint(0,2023, size=Nsample)
    Bkg = ScrambleBkg(Bin, oversample=1, sample='data', seed=int(array_seed[0]), kde=False)
    for s in array_seed[1:Nsample]:
        Bkg += ScrambleBkg(Bin, oversample=1, sample='data', seed=int(s), kde=False)
    Bkg = Bkg/Nsample

# Create the PDF object
SignalPDF = PdfBase(DMRate.flatten()/np.sum(DMRate.flatten()), name="SignalPDF")
ScrSignalPDF = PdfBase(DMRateScr.flatten()/np.sum(DMRateScr.flatten()), name="ScrSignalPDF")

# ScrBkg PDF
BkgPDF = PdfBase(Bkg.flatten()/np.sum(Bkg.flatten()), name="BkgScr")

# model
dm_H1 = Parameter(value=0., limits=(-1,1), fixed=False, name="dm_H1")
dm_H0 = Parameter(value=0., limits=(0,1), fixed=True, name="dm_H0")
f = Parameter(value=1., limits=(0.5,1.5), fixed=True, name="1.")
modelH0 = dm_H0* SignalPDF + f*BkgPDF - dm_H0* ScrSignalPDF
modelH1 = dm_H1* SignalPDF + f*BkgPDF - dm_H1* ScrSignalPDF


Ndata = np.sum(DataHist(Bin, sample='data'))
TSdist = np.array([])
for i in range(ntrials):
    lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)
    ds = DataSet()

    if sample=='null_hypothesis':
        ds.sample(Ndata, Bkg.flatten()/np.sum(Bkg)) 
    elif sample=='RA_scramble':
        data = ScrambleBkg(Bin, oversample=1, seed=i, sample='data', kde=False)
        ds.asimov(Ndata, data.flatten()/np.sum(data))  
    lr.data = ds

    lr.fit('H1')
    lr.fit('H0')
    if lr.models['H1'].parameters["dm_H1"].value>0.:
        TSdist = np.append(TSdist, lr.TS)
    else:
        TSdist = np.append(TSdist, 0)

if output_path!='':
    path = f'{output_path}'
    pkl.dump(TSdist, open(path, "wb"))