#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python
import sys, os
import numpy as np
import re
import pickle as pkl
from scipy.stats import norm
from optparse import OptionParser

base_path=os.getenv('GC_DM_BASE')
output_path=os.getenv('GC_DM_OUTPUT')
sys.path.append(f"{base_path}/Utils/")
sys.path.append(f"{base_path}/PDFs/")
sys.path.append(f"{base_path}/DMfit/DMfit")

from Signal import *
from Background import *
from Plot_Histogram import *
from Utils import *

from modeling import PdfBase, Model, Parameter
from data import DataSet
from llh import LikelihoodRatioTest

import scipy.stats as stats


def compute_pvalue(ts_value):
    p=1-(1/2+1/2*stats.chi2.cdf(ts_value, 1))
    return p

def compute_zscore(ts_value):
    p=compute_pvalue(ts_value)
    z = stats.norm.ppf(1 - p)
    return z

def TS_histogram(xi, signal_pdf, scr_signal_pdf, bkg_pdf, data):
    Ndata = np.sum(data)
    n_H1 = (xi*signal_pdf + bkg_pdf - xi* scr_signal_pdf)* Ndata
    n_H0 = Ndata* bkg_pdf
    chiH1 = stats.poisson.logpmf(data, n_H1)
    chiH0 = stats.poisson.logpmf(data, n_H0)

    # chiH1 = data - n_H1 - data* np.log(data/n_H1)
    # chiH0 = data - n_H0 - data* np.log(data/n_H0)


    TS = 2*(chiH1 - chiH0)
    # remove nan value cause by 0 entry bin in data
    loc = np.where( TS!=TS )
    TS[loc] = 0
    loc = np.where( chiH0!=chiH0 )
    chiH0[loc] = 0
    loc = np.where( chiH1!=chiH1 )
    chiH1[loc] = 0

    return TS, -2*chiH0, -2*chiH1

def discovery_ts(mass, channel, profile, process, Ntrials, sample='data', bkg='data'):
    #############################################################################################################
    #   0 - Create signal and background PDFs
    #
    # Bin
    if mass<=3000:
        Bin = Std_Binning(mass)
    else:
        Bin = Std_Binning(3000)

    # Signal PDF and scrambled signal PDF:
    Reco = RecoRate(channel, mass, profile, Bin, type="Resp", set='1122', spectra='Charon', process=process)
    DMRate = Reco.ComputeRecoRate()

    Reco.Scramble = True
    Reco.hist['Resp'] = None
    DMRateScr = Reco.ComputeRecoRate()

    # RA scrambled Background
    if bkg=='burnsample':
        Bkg = ScrambleBkg(Bin, bandwidth="ISJ", oversample=10, sample='burnsample')
    else:
        # For bkg from data: create 10 estimation with 10 different seed -> average over them
        # Bkg = ScrambleBkg(Bin, bandwidth="ISJ", oversample=10, sample=bkg)
        Nsample = 100
        np.random.seed(2023)
        array_seed = np.random.uniform(0,2023, size=Nsample)
        Bkg = ScrambleBkg(Bin, bandwidth="ISJ", oversample=1, sample='data', seed=int(array_seed[0]), kde=True)
        for s in array_seed[1:Nsample]:
            Bkg += ScrambleBkg(Bin, bandwidth="ISJ", oversample=1, sample='data', seed=int(s), kde=True)
        Bkg = Bkg/Nsample

    #############################################################################################################
    #   1 - Load data
    #
    if 'RA_scr_data' in sample:
        if 'with_seed' in sample:
            seed = int(re.findall(r'\d+', sample)[0])
            data_hist = ScrambleBkg(Bin, bandwidth="ISJ", sample='data', oversample=1, kde=False, seed=seed)
        else:
            data_hist = ScrambleBkg(Bin, bandwidth="ISJ", sample='data', oversample=1, kde=False)
            
    else:
        data_hist = DataHist(Bin, sample=sample)
    Ndata = np.sum(data_hist)


    #############################################################################################################
    #   2 - Create PDFs, models
    #

    SignalPDF = PdfBase(DMRate.flatten()/np.sum(DMRate.flatten()), name="SignalPDF")
    ScrSignalPDF = PdfBase(DMRateScr.flatten()/np.sum(DMRateScr.flatten()), name="ScrSignalPDF")
    BkgPDF = PdfBase(Bkg.flatten()/np.sum(Bkg.flatten()), name="BkgScr")

    dm_H1 = Parameter(value=0., limits=(-1.,1.), fixed=False, name="signal_fraction")
    f = Parameter(value=1., limits=(0.5,1.5), fixed=True, name="1.") # a 1. factor to make model work!

    modelH0 = f*BkgPDF 
    modelH1 = dm_H1* SignalPDF + f*BkgPDF - dm_H1* ScrSignalPDF

    #############################################################################################################
    #   3 - TS distribution under the assumption of null hypothesis
    #
    lr_hypothesis = LikelihoodRatioTest(model = modelH1, null_model = modelH0)

    ds = DataSet()
    TSdist = np.array([])
    for i in range(Ntrials):
        ds.sample(Ndata, Bkg.flatten()/np.sum(Bkg.flatten()))    
        lr_hypothesis.data = ds
        lr_hypothesis.fit('H0')
        lr_hypothesis.fit('H1')
        if lr_hypothesis.models['H1'].parameters["signal_fraction"].value>=0:
            TSdist = np.append(TSdist, lr_hypothesis.TS)
        else:
            TSdist = np.append(TSdist, 0)

    #############################################################################################################
    #   4 - Actual TS value of the data and the best fit signal fraction
    #
    lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)
    data = DataSet()
    data.asimov(Ndata, data_hist.flatten()/np.sum(data_hist.flatten()))    
    lr.data = data
    lr.fit('H0')
    lr.fit('H1')
    xi_bf = lr.models['H1'].parameters["signal_fraction"].value
    if lr.models['H1'].parameters["signal_fraction"].value>=0:
        TSdata = lr.TS
    else:
        TSdata = 0

    #############################################################################################################
    #   5 - Scan the LLR to make sure the fit work
    #
    LLR = dict()
    LLR['LLR_scan'] = np.array([])
    LLR['xi_scan'] = np.linspace(-0.05, 0.05, 100)
    lr.models['H1'].parameters["signal_fraction"].fixed=True
    for xi in LLR['xi_scan']:
        lr.models['H1'].parameters["signal_fraction"].value=xi
        lr.fit("H0")
        lr.fit("H1")
        LLR['LLR_scan'] = np.append(LLR['LLR_scan'], lr.TS)


    #############################################################################################################
    #   6 - histogram of TS distribution
    #
    TShistogram = TS_histogram(xi_bf, DMRate/np.sum(DMRate), DMRateScr/np.sum(DMRateScr), Bkg/np.sum(Bkg), data_hist)

    return TSdist, TSdata, xi_bf, LLR, TShistogram

parser = OptionParser()
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = 'string', action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("--process", type = 'string', action = "store", default = "ann", metavar  = "<process>", help = "process: ann or decay",)
parser.add_option("-s", "--spectra", type = 'string', action = "store", default = "Charon", metavar  = "<spectra>", help = "Spectra: Charon or PPPC4",)
parser.add_option("--mass_default", action="store_true", dest="mass_default", default=False, help="Set mass scan values to the default used for the analysis as indicated in the wikipage")  
parser.add_option("-u", "--up", type = float, action = "store", default = 100, metavar  = "<up>", help = "Dark Matter mass upper value for scanning",)
parser.add_option("-l", "--low", type = float, action = "store", default = 100, metavar  = "<low>", help = "Dark Matter mass lower value for scanning",)
parser.add_option("-n", "--n", type = int, action = "store", default = 1, metavar  = "<n>", help = "# of point scan on DM mass from defined lower to upper value",)
parser.add_option("--ntrials", type = int, action = "store", default = 10000, metavar  = "<ntrials>", help = "# of trials",)
parser.add_option("--bkg", type = 'string', action = "store", default = 'data', metavar  = "<bkg>", help = "building background as RA scrambling of burnsample of full data",)
parser.add_option("--data", type = 'string', action = "store", default = 'data', metavar  = "<data>", help = "data sample to use: burnsample, data or RA_scr_data",)
parser.add_option("--outfile", type = 'string', action = "store", default = '', metavar  = "<outfile>", help = "path to the output file",)


(options, args) = parser.parse_args()
channel = options.channel
profile = options.profile
process = options.process
up = options.up
low = options.low
n = options.n
ntrials = options.ntrials
mass_default = options.mass_default
bkg=options.bkg
data=options.data
outfile = options.outfile

print("****** input values:")
print(f"bkg:{bkg}")
print(f"data:{data}")


#############################################################################################################
#   0 - Defined default mass range for each channels
#
if mass_default:
    if process=='ann':
        masses = {"WW":[90, 8000], "bb":[15, 8000], 'tautau':[5, 4000], 'mumu':[5, 1000], "nuenue":[5, 200],"numunumu":[5, 200],"nutaunutau":[5,200]}
    elif process=='decay':    
        masses = {"WW":[180, 8000], "bb":[30, 8000], 'tautau':[5, 8000], 'mumu':[5, 2000], "nuenue":[5, 400],"numunumu":[5, 400],"nutaunutau":[5,400]}
    
    low = masses[channel][0]
    up = masses[channel][1]
    n = 30

#############################################################################################################
#   1 - Loop on mass values and compute significance for each mass
#       then save to a dictionary
#
masses = np.exp(np.linspace(np.log(low), np.log(up), n))
results = dict()

for m in masses:
    ts_dist, ts_data, xi_bf, llr, ts_histogram = discovery_ts(m, channel, profile, process, ntrials, sample=data, bkg=bkg)
    results[m] = dict()
    results[m]['TSdist'] = ts_dist
    results[m]['TSdata'] = ts_data
    results[m]['xi_bestfit'] = xi_bf
    results[m]['LLR'] = llr
    results[m]['pvalue'] = compute_pvalue(ts_data)
    results[m]['zscore'] = compute_zscore(ts_data)
    results[m]['TS_histogram'] = dict()
    results[m]['TS_histogram']['TS'] = ts_histogram[0]
    results[m]['TS_histogram']['-2LLH_H0'] = ts_histogram[1]
    results[m]['TS_histogram']['-2LLH_H1'] = ts_histogram[2]

    print(f'mass : {m}')
    print(f'TS : {ts_data}')
    print(f'xi_bestfit : {xi_bf}')
    print(f'p-value : {compute_pvalue(ts_data)}')
    print(f'z-score : {compute_zscore(ts_data)}')

print(f'Saving output in {outfile}')
if outfile!='':
    pkl.dump(results, open(outfile, "wb"))