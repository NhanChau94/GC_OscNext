#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python
import sys, os
import numpy as np
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

def compute_pvalue(ts_values, ts_data):
    """
    Compute the p-value from a test statistics distribution and the test statistic value of the data.
    
    Args:
        ts_values (array-like): Array of test statistics values.
        ts_data (float): Test statistic value of the data.
    
    Returns:
        p_value (float): The computed p-value.
    """
    num_greater = np.sum(ts_values >= ts_data)  # Count the number of test statistics values greater than or equal to ts_data
    p_value = (num_greater + 1) / (len(ts_values) + 1)  # Add 1 to numerator and denominator for Laplace smoothing
    # p_value = (num_greater) / (len(ts_values)) 


    return p_value

def compute_zscore(ts_values, ts_data):
    """
    Compute the significance in sigma from a test statistics distribution and the test statistic value of the data.
    
    Args:
        ts_values (array-like): Array of test statistics values.
        ts_data (float): Test statistic value of the data.
    
    Returns:
        significance (float): The computed significance in sigma.
    """
    p_value = compute_pvalue(ts_values, ts_data)
    z_score = norm.ppf(1 - p_value/2.) # 2 tail test    
    return z_score


def discovery_ts(mass, channel, profile, process, Ntrials, sample='burnsample'):
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
    loadpdf = pkl.load(open('/data/user/tchau/DarkMatter_OscNext/PDFs/Background/RAScramble_burnsample_FFTkde.pkl', 'rb'))
    Bkg = loadpdf['pdf']
    # Bkg = ScrambleBkg(Bin, bandwidth="ISJ", oversample=10)

    #############################################################################################################
    #   1 - Load data
    #
    data_hist = DataHist(Bin, sample=sample)


    #############################################################################################################
    #   2 - Create PDFs, models and LLR object
    #

    SignalPDF = PdfBase(DMRate.flatten()/np.sum(DMRate.flatten()), name="SignalPDF")
    ScrSignalPDF = PdfBase(DMRateScr.flatten()/np.sum(DMRateScr.flatten()), name="ScrSignalPDF")
    BkgPDF = PdfBase(Bkg.flatten()/np.sum(Bkg.flatten()), name="BkgScr")

    dm_H1 = Parameter(value=0., limits=(0,1), fixed=False, name="signal_fraction")
    f = Parameter(value=1., limits=(0,1), fixed=True, name="1.") # a 1. factor to make model work!

    modelH0 = f*BkgPDF 
    modelH1 = dm_H1* SignalPDF + f*BkgPDF - dm_H1* ScrSignalPDF

    lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)
    ds = DataSet()

    #############################################################################################################
    #   3 - TS distribution under the assumption of null hypothesis
    #

    TSdist = np.array([])
    Ndata = np.sum(data_hist)
    for i in range(Ntrials):
        ds.sample(Ndata, Bkg.flatten()/np.sum(Bkg.flatten()))    
        lr.data = ds
        lr.fit('H1')
        lr.fit('H0')
        TSdist = np.append(TSdist, lr.TS)

    #############################################################################################################
    #   4 - Actual TS value of the data
    #
    ds.asimov(Ndata, data_hist.flatten()/np.sum(data_hist.flatten()))    
    lr.data = ds
    lr.fit('H1')
    lr.fit('H0')
    TSdata = lr.TS

    return TSdist, TSdata

parser = OptionParser()
parser.add_option("-c", "--channel", type = "string", action = "store", default = "WW", metavar  = "<channel>", help = "Dark matter channel",)
parser.add_option("-p", "--profile", type = 'string', action = "store", default = "NFW", metavar  = "<profile>", help = "GC profile",)
parser.add_option("--process", type = 'string', action = "store", default = "decay", metavar  = "<process>", help = "process: ann or decay",)
parser.add_option("-s", "--spectra", type = 'string', action = "store", default = "Charon", metavar  = "<spectra>", help = "Spectra: Charon or PPPC4",)
parser.add_option("--mass_default", action="store_true", dest="mass_default", default=False, help="Set mass scan values to the default used for the analysis as indicated in the wikipage")  
parser.add_option("-u", "--up", type = float, action = "store", default = 100, metavar  = "<up>", help = "Dark Matter mass upper value for scanning",)
parser.add_option("-l", "--low", type = float, action = "store", default = 100, metavar  = "<low>", help = "Dark Matter mass lower value for scanning",)
parser.add_option("-n", "--n", type = int, action = "store", default = 1, metavar  = "<n>", help = "# of point scan on DM mass from defined lower to upper value",)
parser.add_option("--ntrials", type = int, action = "store", default = 10000, metavar  = "<ntrials>", help = "# of trials",)
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
outfile = options.outfile

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
results['mass'] = masses
results['TS'] = np.array([])
results['TSdist'] = np.array([])
results['pvalue'] = np.array([])
results['zscore'] = np.array([])

for m in masses:
    ts_values, ts_data = discovery_ts(m, channel, profile, process, ntrials, sample='burnsample')
    pvalue = compute_pvalue(ts_values, ts_data)
    zscore = compute_zscore(ts_values, ts_data)
    results['TS'] = np.append(results['TS'], ts_data)
    results['TSdist'] = np.append(results['TSdist'], ts_values)
    results['pvalue'] = np.append(results['pvalue'], pvalue)
    results['zscore'] = np.append(results['zscore'], zscore)    

if outfile!='':
    pkl.dump(results, open(outfile, "wb"))