"""
author : N. Chau
Creating signal event distribution + pdfs with a detector response
"""
import sys, os
import math
import pickle as pkl
import numpy as np
import scipy
from scipy.interpolate import pchip_interpolate

base_path=os.getenv('GC_DM_BASE')
data_path=os.getenv('GC_DM_DATA')
sys.path.append(f"{base_path}/Utils/")
sys.path.append(f"{base_path}/Spectra/")
sys.path.append(f"{base_path}/DetResponse/")
sys.path.append(f"{base_path}/PDFs/")

from Detector import *
from Utils import *
from Interpolate import *
from NuSpectra import *
from Jfactor import *
from KDE_implementation import *



#############################################################################
# Set of functions for spectra and Jfactor
#############################################################################

# set low values to zero in case needed
def cutspectra(spec, cut):
    for flv in ["nu_e", "nu_mu", "nu_tau"]:
        for i, val in enumerate(spec[flv]["dNdE"]):
            if val<cut:
                spec[flv]["dNdE"][i] = 0
    return spec



##---------------------------------------------##
##Interpolate the Jfactor at the desired psi values
##Required:
##  -  Jfactor: Precomputed Jfactor file
##  -  psival: psi values for interpolation
##Output:
##  -  Interpolated Jfactor
##---------------------------------------------##

def Interpolate_Jfactor(Jfactor, psival):
    y_interp = scipy.interpolate.splrep(Jfactor["psi"], Jfactor["J"])
    interp_Jpsi = scipy.interpolate.splev(psival, y_interp, der=0)

    return interp_Jpsi

##---------------------------------------------##
##Interpolate the Spectra at the desired energy values
##Required:
##  -  spectra: Precomputed spectra (dictionary of each neutrino flavour)
##  -  Eval: energy values for interpolation
##  -  Emax: maximum energy considered (usually mass value for annihilation and mass/2. for decay)
##Output:
##  -  Interpolated spectra
##---------------------------------------------##

def Interpolate_Spectra(spectra, Eval, Emax, cutlow=True):

    #Define array holding interpolated values of spectra
    interp_dNdE = dict()
    nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]

    for nu_type in nu_types:
        #Energy
        spectra_E = np.array(spectra[nu_type]["E"])
        #Spectra
        spectra_dNdE = np.array(spectra[nu_type]["dNdE"])
        #Define fct from which interpolate from
        # y_interp = scipy.interpolate.splrep(spectra_E, spectra_dNdE)
        # interp_dNdE[nu_type] = scipy.interpolate.splev(Eval, y_interp, der=0)
        interp_dNdE[nu_type] = pchip_interpolate(spectra_E, spectra_dNdE*spectra_E, Eval)/Eval

        # bb and nunu channel of mass below 100GeV gives weird features due to lack of stat in pythia table of charon
        # set values of spectra below 1e-5 to zero and only interpolate the part >1e-5
        if cutlow==True:
            low_v = np.where(interp_dNdE[nu_type]<1e-5)[0]
            interp_dNdE[nu_type][low_v] = 0.
        # put spectra to zero for energy > Emax (Emax=mass for annihilation, mass/2 for decay)
        loc = np.where(Eval>Emax)
        interp_dNdE[nu_type][loc] = 0.

    return interp_dNdE



##---------------------------------------------##
##Compute the expected rate
##Required:
##  -  Spectra from Charon (PPPC4)
##  -  Jfactor
##  -  Energy and angular values used for Spectra and Jfactor
##Output:
##  -  2D expected rate distribution in psi and energy:
##     Rate[nutype][psi][E]
##---------------------------------------------##
def TrueRate(Spectra, Jfactor):
    Rate = dict()
    for nu_type in Spectra.keys():
        Rate[nu_type] = np.array(Jfactor[:,None]* Spectra[nu_type])
    return Rate



#############################################################################
# Set of functions for response matrix cimputation from MC
#############################################################################

##---------------------------------------------##
##Compute the Response Matrix
##Required:
##  -  MCcut: MC events
##  -  Bin: binning scheme
##  -  bw_method: bandwidth method for kde
##  -  maxEtrue: maximum true energy considered
##  -  maxEreco: maximum reco energy considered
##  -  Scramble: if scrambling the Right Ascension
##  -  mirror: if using reflection at psi=0
##Output:
##  -  return the pdf of response matrix (differential of response matrix):
##     Resp[nutype][psi][E] = d(OneWeight/N_evt)/(dEtrue dpsitrue dpsireco d(log10 Ereco)) normalized to 1
##---------------------------------------------##


def KDE_RespMatrix(MCcut, Bin, bw_method, maxEtrue=3000, maxEreco=1000, Scramble=False, mirror=True):

    #Evaluate points:
    print("Preparing evaluation grid") 
    ##Equal spacing in the final variables: reco Psi & log10(E_reco), true psi and true E
    trueEeval, recoEeval, truePsieval, recoPsieval = Extend_EvalPoints(Bin["true_energy_center"], Bin["reco_energy_center"], maxEtrue, maxEreco, Bin["true_psi_center"], Bin["reco_psi_center"])  
    # print('Etrue: {}'.format(trueEeval))
    # print('Ereco: {}'.format(recoEeval))
    # print('Psitrue: {}'.format(truePsieval))
    # print('Psireco: {}'.format(recoPsieval))
    
    g_psi_true, g_energy_true, g_psi_reco, g_energy_reco = np.meshgrid(truePsieval, trueEeval,
                                                            recoPsieval, recoEeval, indexing='ij')                      
    psi_eval_true = g_psi_true.flatten()
    E_eval_true = g_energy_true.flatten()
    psi_eval_reco = g_psi_reco.flatten()
    E_eval_reco = g_energy_reco.flatten()

    ##Evaluate the KDE in log(Psi)-E
    psiE_eval = np.vstack([psi_eval_true, E_eval_true, 
            psi_eval_reco, np.log10(E_eval_reco)])



    # Separate MC by each channel nutype (for now 1PID)
    nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}
    # PID = [[0.,0.5],[0.5, 0.85],[0.85, 1]]
    PID = [[0.,1.]]
    Resp = dict()
    for pid in PID:
        # print("Computing {} PID bin".format(pid))
        # Resp[pidbin] = dict()
        # pidbin += 1
        for nu_type in nu_types:
            print("----{}".format(nu_type))

            psireco=MCcut["psi_reco"]
            if Scramble:
                print("Scrambled Response matrix")
                RAreco = MCcut["RA_reco"]
                Decreco = MCcut["Dec_reco"]
                # Create scramble RA:
                RAreco_Scr = np.random.uniform(0,2.*np.pi, size=len(RAreco))
                # Get correct psi from scramble RA and original DEC
                psireco = np.rad2deg(psi_f(RAreco_Scr, Decreco))

            loc = np.where(  (MCcut["nutype"]==pdg_encoding[nu_type]) & (MCcut["PID"]>=pid[0])
                            & (MCcut["PID"]<pid[1]) 
                            & (MCcut["E_reco"] < maxEreco)
                            & (MCcut["E_reco"] > np.min(Bin["reco_energy_center"]))
                            & (MCcut["E_true"] < maxEtrue)
                            & (MCcut["E_true"] > np.min(Bin["true_energy_center"]))
                            # & (MCcut["psi_true"] < np.max(Bin["true_psi_center"]))
                            # & (MCcut["psi_true"] > np.min(Bin["true_psi_center"]))
                            # & (psireco < np.max(Bin["reco_psi_center"]))
                            # & (psireco > np.min(Bin["reco_psi_center"]))
                            )
        
            #Extract MC events: 
            #NOTE: input psi in deg!
            psitrue = MCcut["psi_true"][loc]
            Etrue = MCcut["E_true"][loc]
            psireco = psireco[loc]
            Ereco = MCcut["E_reco"][loc]
            w = MCcut["w"][loc]        

            print("Preparing train grid")    
            # psiE_train = np.vstack([np.log(psitrue), Etrue, np.log(psireco), np.log10(Ereco)])
            
            psiE_train = np.vstack([psitrue, Etrue, psireco, np.log10(Ereco)])

            if mirror:
                psiE_train=MirroringData(psiE_train, {0:0, 2:0})
                w=np.concatenate((w,w))
                print("Correct bias at boundary psi=0 using mirror data (reflection)")
                # extend grid point to contain the mirror data
                recoPsieval_width = recoPsieval[1] - recoPsieval[0]
                while recoPsieval[0]>-180.:
                    recoPsieval=np.append(recoPsieval[0]-recoPsieval_width, recoPsieval)
                
                truePsieval_width = truePsieval[1] - truePsieval[0]
                while truePsieval[0]>-180.:
                    truePsieval=np.append(truePsieval[0]-truePsieval_width, truePsieval)
                
                g_psi_true, g_energy_true, g_psi_reco, g_energy_reco = np.meshgrid(truePsieval, trueEeval,
                                                                        recoPsieval, recoEeval, indexing='ij')                      
                psi_eval_true = g_psi_true.flatten()
                E_eval_true = g_energy_true.flatten()
                psi_eval_reco = g_psi_reco.flatten()
                E_eval_reco = g_energy_reco.flatten()

                psiE_eval = np.vstack([psi_eval_true, E_eval_true, 
                        psi_eval_reco, np.log10(E_eval_reco)]) 


            print("Evaluating KDE.....")    

            kde_w = kde_FFT(psiE_train.T, psiE_eval.T, bandwidth=bw_method, weights=w)
            kde_weight = kde_w
                                    
            # Fill into histogram:
            Psitrue_edges = Bin["true_psi_edges"]
            Etrue_edges = Bin["true_energy_edges"]
            Psireco_edges = Bin["reco_psi_edges"]
            Ereco_edges = Bin["reco_energy_edges"]

            H, edges = np.histogramdd((psi_eval_true, E_eval_true, psi_eval_reco, E_eval_reco),
                                bins = (Psitrue_edges, Etrue_edges, Psireco_edges, Ereco_edges),
                                weights=kde_weight)

            # N = np.histogramdd((psi_eval_true, E_eval_true, psi_eval_reco, E_eval_reco),
            #                     bins = (Psitrue_edges, Etrue_edges, Psireco_edges, Ereco_edges))
            # if np.min(N)==0:
            #     return 'There is an empty bin!!'                    
            # H = H/N[0]
             
            if mirror:
                norm = np.sum(w)/(2*np.sum(H))
            else:
                norm = np.sum(w)/np.sum(H) 
            Resp[nu_type] = H*norm
    return Resp      

##---------------------------------------------##
##Interpolation on a dense precomputed response matrix
##Required:
##  -  MCset: id of MC sample (1000, 1122, ...)
##  -  Bin: binning scheme
##  -  Scramble: if scrambling the Right Ascension
##  -  logEtrue: if the grid is in log10 Etrue instead of linear Etrue
##Output:
##  -  return the interpolated response matrix
##---------------------------------------------##

def RespMatrix_Interpolated(MCset, Bin, Scramble=False, logEtrue=True):
    Evaltrue = Bin['true_energy_center']
    Evalreco = Bin['reco_energy_center']
    Psievaltrue = Bin['true_psi_center']
    Psievalreco = Bin['reco_psi_center']

    # Access precomputed response matrix and its grid
    if logEtrue:
        indict = pkl.load(open("{}/DetResponse/Resp_MC{}_logE.pkl".format(data_path, MCset), "rb"))
    else:
        indict = pkl.load(open("{}/DetResponse/Resp_MC{}.pkl".format(data_path, MCset), "rb"))
    if Scramble:
        Resp = indict["Resp_Scr"]
    else:
        Resp = indict["Resp"]

    psitrue = indict['Bin']['true_psi_center']
    Etrue = indict['Bin']['true_energy_center']
    psireco = indict['Bin']['reco_psi_center']
    Ereco = indict['Bin']['reco_energy_center']

    nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    Resp_interpolated = dict()
    for nu in nu_types:
        if logEtrue:
            Resp_interpolated[nu] = EqualGridInterpolator((psitrue, np.log10(Etrue), psireco, np.log10(Ereco)), Resp[nu], order=1, fill_value=0)(np.meshgrid(Psievaltrue, np.log10(Evaltrue), Psievalreco, np.log10(Evalreco),  indexing='ij'))
        else:
            Resp_interpolated[nu] = EqualGridInterpolator((psitrue, Etrue, psireco, np.log10(Ereco)), Resp[nu], order=1)(np.meshgrid(Psievaltrue, Evaltrue, Psievalreco, np.log10(Evalreco),  indexing='ij'))
    
    return Resp_interpolated

##---------------------------------------------##
##Histogram of a data sample
##Required:
##  -  Bin: binning scheme
##  -  sample: burn sample or full data
##Output:
##  -  return data histogram
##---------------------------------------------##

def DataHist(Bin, sample='burnsample'):
    if sample=='burnsample':
        dat_dir = f"{data_path}/Sample/Burnsample/"
        input_files = []
        # Take all burnsample:
        for year in range(2012, 2021):
            infile = dat_dir + "OscNext_Level7_v02.00_burnsample_{}_pass2_variables_NoCut.pkl".format(year)
            dat = pkl.load(open(infile, 'rb'))
            input_files = np.append(input_files, dat['burnsample'])
    
    array_PID = np.array([])
    array_recopsi = np.array([])
    array_recoE = np.array([])

    # input_file = data['burnsample']
    for input_file in input_files:
        # define cut:
        loc = np.where((input_file["L7muon_classifier_up"]>0.4) &
                        (input_file["L4noise_classifier"]>0.95) &
                        (input_file["L7reco_vertex_z"]>-500.) &
                        (input_file["L7reco_vertex_z"]<-200.) &
                        (input_file["L7reco_vertex_rho36"]<300.) &
                        (input_file["L5nHit_DOMs"]>2.5) &
                        (input_file["L7_ntop15"]<2.5) &
                        (input_file["L7_nouter"]<7.5) &
                        (input_file["L7reco_time"]<14500.))
    
        array_PID = np.append(array_PID, input_file["PID"][loc])
        array_recopsi = np.append(array_recopsi, np.rad2deg(input_file["reco_psi"][loc]))
        array_recoE = np.append(array_recoE, input_file["reco_TotalEnergy"][loc])
    
    Psireco_edges = Bin["reco_psi_edges"]
    Ereco_edges = Bin["reco_energy_edges"]
    H = np.histogram2d(array_recopsi, array_recoE,
                     bins = (Psireco_edges, Ereco_edges))

    return H[0]





#############################################################################
# Set of functions for evt by evt reweight
#############################################################################




##---------------------------------------------------------------------
##Define cut on weight
##---------------------------------------------------------------------
def define_weightcut(weight, cut):
    
    H, edges = np.histogram(weight, bins=1000)
    zeroes = np.where(H==0.)
    
    #print (zeroes)
    
    i = 0
    n = 0

    while (i<zeroes[0].shape[0]-1) and (n<cut+1):
        
        #Check if consecutive zeroes
        if zeroes[0][i]+1 == zeroes[0][i+1]:
            n += 1
        #Reset n to zero if encounter non-nul value
        elif zeroes[0][i]+1 != zeroes[0][i+1]:
            #print ("Reset to zero")
            n = 0

        if n == cut:
            loc = zeroes[0][i]
            #print ("Location:", loc)
    
        i+=1
        
    if n >= cut:
        w_lim = edges[loc]
    else:
        w_lim = max(weight)
        
    return w_lim


##---------------------------------------------##
##Compute weights and extract other informations used for evt-by-evt reweight
##Required:
##  -  MCdict: MC events
##  -  Spectra: neutrino spectra from DM (dN/dE)
##  -  Jfactor
##  -  mass: DM mass
##  -  maxE: maximum true energy considered for the MC
##  -  weight_cut: if applying cut on very high weight events
##Output:
##  -  array of necessary event weights and other informations
##---------------------------------------------##
def ComputeWeight(MCdict, Spectra, Jfactor, mass, maxE=3000, weight_cut=True):
    nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}

    #PDF_variables
    array_recopsi = np.array([])
    array_recoE = np.array([])
    array_recoRA = np.array([])
    array_recoDec = np.array([])
    
    array_PID = np.array([])
    signal_w = np.array([])

    for nu_type in nu_types:
        loc = np.where( (MCdict["E_true"]<=mass) & (MCdict["nutype"]==pdg_encoding[nu_type]) & (MCdict["E_reco"]<=maxE) )
        if len(loc[0])==0:
            continue
        ##Sort all variables by increasing true_E values##
        ##NOTE: this is required for spectra interpolation
        # sort = MCdict["E_true"][loc].argsort()

        ##Simulation weight##
        genie_w = MCdict["w"][loc]

        ##Spectra interpolation##
        true_E = MCdict["E_true"][loc]
        dNdE = Interpolate_Spectra(Spectra, true_E, mass)

        ##Jfactor interpolation##
        #NOTE: input psi in deg!
        true_psi = MCdict["psi_true"][loc]
        Jpsi = Interpolate_Jfactor(Jfactor, true_psi)

        ##Signal weight##
        weight = (1./(2 * 4*math.pi * mass**2)) * genie_w * dNdE[nu_type] * Jpsi

        ##Reco variables:
        reco_psi = MCdict["psi_reco"][loc]
        reco_E = MCdict["E_reco"][loc]
        reco_RA = MCdict["RA_reco"][loc]
        reco_Dec = MCdict["Dec_reco"][loc]
        PID = MCdict["PID"][loc]

        ## perform cuts on weight
        if weight_cut:
            w_lim = define_weightcut(weight, 200) #Previously 200
            print ("##Applying cut on weight##")
            print ("Weight lim:", w_lim)
            w_loc = np.where(weight<=w_lim)
            #Renormalise weight for total weight to be unchanged
            weight = weight[w_loc] * (np.sum(weight)/np.sum(weight[w_loc]))
            true_psi = true_psi[w_loc]
            true_E = true_E[w_loc]
            reco_psi = reco_psi[w_loc]
            reco_E = reco_E[w_loc]
            reco_RA = reco_RA[w_loc]
            reco_Dec = reco_Dec[w_loc]
            PID = PID[w_loc] 


        ##group all nutype:
        array_recopsi = np.append(array_recopsi, reco_psi)
        array_recoE = np.append(array_recoE, reco_E)
        array_PID = np.append(array_PID, PID)
        array_recoRA = np.append(array_recoRA, reco_RA)
        array_recoDec = np.append(array_recoDec, reco_Dec)
        signal_w = np.append(signal_w, weight)
    return array_PID, array_recopsi, array_recoE, signal_w, array_recoRA, array_recoDec

##---------------------------------------------##
## evt-by-evt reweighting for computation of signal expectation
##---------------------------------------------##
def KDE_evtbyevt(MCdict, Spectra, Jfactor, mass, bw_method, Bin, Scramble=False, weight_cut=True, mirror=True):
    array_PID, array_recopsi, array_recoE, signal_w, array_recoRA, array_recoDec = ComputeWeight(MCdict, Spectra, Jfactor, mass, weight_cut=weight_cut)
    # Define PID cut:
    # PID = [[0.,0.5],[0.5, 0.85],[0.85, 1]]
    PID = [[0, 1]]


    #Evaluate points:
    print("Preparing evaluation grid") 
    ##Equal spacing in the final variables: reco Psi & log10(E), true psi and true E
    maxE = 3000.
    trueEeval, recoEeval, truePsieval, recoPsieval = Extend_EvalPoints(Bin["true_energy_center"], Bin["reco_energy_center"], maxE, 1500, Bin["true_psi_center"], Bin["reco_psi_center"])  

    g_psi_reco, g_energy_reco = np.meshgrid(recoPsieval, recoEeval, indexing='ij')                      
    psi_eval_reco = g_psi_reco.flatten()
    E_eval_reco = g_energy_reco.flatten()

    ##Evaluate the KDE in log(Psi)-log10E
    # psiE_eval = np.vstack([np.log(psi_eval_true), E_eval_true, 
    #                     np.log(psi_eval_reco), np.log10(E_eval_reco)])
    psiE_eval = np.vstack([psi_eval_reco, np.log10(E_eval_reco)])

    # pdf = np.zeros((len(PID),len(Psireco_edges)-1, len(Ereco_edges)-1))
    # i = 0
    # sum = 0
    for pidbin in PID:
        print("Computing {} PID bin".format(pidbin))
            
        loc = np.where( ( array_PID >= pidbin[0]) & ( array_PID <= pidbin[1]) 
                            & (array_recoE < maxE)
                            & (array_recoE > np.min(Bin["reco_energy_center"]))
                        )
                   
        #PDF_variables
        recopsi = array_recopsi[loc]
        if Scramble:
            print("Buid Scramble PDF--------")
            RAreco = array_recoRA[loc]
            Decreco = array_recoDec[loc]
            # Create scramble RA:
            RAreco_Scr = np.random.uniform(0,2.*np.pi, size=len(RAreco))
            # Get correct psi from scramble RA and original DEC
            recopsi = np.rad2deg(psi_f(RAreco_Scr, Decreco))
            
        recoE = array_recoE[loc]
        weight = signal_w[loc]

        psiE_train = np.vstack([recopsi,np.log10(recoE)])
        print(psiE_train.shape)
        if mirror:
            psiE_train=MirroringData(psiE_train, {0:0})
            weight=np.concatenate((weight,weight))
            print("Correct bias at boundary psi=0 using mirror data i.e reflection")
            # extend grid point to contain the mirror data
            recoPsieval_width = recoPsieval[1] - recoPsieval[0]
            while recoPsieval[0]>-180.:
                recoPsieval=np.append(recoPsieval[0]-recoPsieval_width, recoPsieval)
            
            g_psi_reco, g_energy_reco = np.meshgrid(recoPsieval, recoEeval, indexing='ij')                      
            psi_eval_reco = g_psi_reco.flatten()
            E_eval_reco = g_energy_reco.flatten()
            psiE_eval = np.vstack([psi_eval_reco, np.log10(E_eval_reco)])    

        ##Evaluate KDE##
        #In terms of log(psi)-log10(E)
        kde_w = kde_FFT(psiE_train.T, psiE_eval.T
                        ,bandwidth=bw_method
                            ,weights=weight)
        kde_weight = kde_w                

        # Fill into histogram:
        Psireco_edges = Bin["reco_psi_edges"]
        Ereco_edges = Bin["reco_energy_edges"]
        H, v0_edges, v1_edges = np.histogram2d(psi_eval_reco, E_eval_reco,
                                            bins = (Psireco_edges, Ereco_edges),
                                            weights=kde_weight)

    return H/np.sum(kde_weight)*np.sum(weight)



#############################################################################
# Central class for computing signal expectation 
# from true signal flux and detector response from MC
#############################################################################

class RecoRate:
    """docstring for RecoRate."""
    def __init__(
            self,
            channel, 
            mass, 
            profile, 
            bin, 
            process='ann',
            xsec=1,
            tdecay=1,
            type='Resp', 
            PreCompResp=True,
            interpolate_resp=True,
            spectra='Charon', 
            set='1122', 
            Scramble=False
            ):
        self.channel = channel
        self.mass = mass
        self.profile = profile
        self.bin = bin
        self.process = process
        self.xsec = xsec
        self.tdecay = tdecay
        self.type = type
        self.PreCompResp = PreCompResp
        self.interpolate_resp = interpolate_resp
        self.spectra = spectra
        self.set = set
        self.Scramble = Scramble
        self.MCdict = None
        self.hist = dict()
        self.hist['Spectra'] = None
        self.hist['Jfactor'] = None
        self.hist['TrueRate'] = None
        self.hist['Resp'] = None
        self.hist['RecoRate'] = None

    def ComputeSpectra(self):
        if self.hist['Spectra'] is not None:
            print('Spectra already computed, will not compute it again')
        else:
            print('*'*20)
            print('Computing Spectra')
            Nu = NuSpectra(self.mass, self.channel, self.process)
            if self.mass >= max(self.bin['true_energy_edges']):
                Nu.Emax = max(self.bin['true_energy_edges'])
            Nu.nodes=500
            Nu.bins=500

            if "PPPC4" in self.spectra:
                spectra_dict = Nu.SpectraPPPC4_AvgOsc()
            elif "Charon" in self.spectra:
                spectra_dict = Nu.SpectraCharon_nuSQUIDS()
            
            # bb and nunu channel of mass below 100GeV gives weird features due to lack of stat in pythia table of charon
            # set values of spectra below 1e-5 to zero and only interpolate the part >1e-5
            if self.mass < 100 and ("Charon" in self.spectra) and (self.channel=="bb" or self.channel=="nunu" or self.channel=="nuenue"
                        or self.channel=="numunumu" or self.channel=="nutaunutau"):
                cutlow=True
            else:
                cutlow=False
            if self.process=='ann':           
                self.hist['Spectra'] = Interpolate_Spectra(spectra_dict, self.bin['true_energy_center'], self.mass, cutlow=cutlow)
            elif self.process=='decay':
                self.hist['Spectra'] = Interpolate_Spectra(spectra_dict, self.bin['true_energy_center'], self.mass/2., cutlow=cutlow)

        return self.hist['Spectra']
    
    def ComputeJfactor(self):
        if self.hist['Jfactor'] is not None:
            print('Jfactor already computed, will not compute it again')
        else:
            print('*'*20)
            print('Computing Jfactor with default option: precomputed Clumpy file')
            MyJ = Jf(process=self.process, profile=self.profile)
            Jfactor = MyJ.Jfactor_Clumpy()
            self.hist['Jfactor'] = Interpolate_Jfactor(Jfactor, self.bin['true_psi_center'])
    
        return self.hist['Jfactor']    

    def ComputeTrueRate(self):
        print("*"*20)
        if self.hist['TrueRate'] is not None:
            print('True rate already computed, will not compute it again')
        else:    
            print("Computing true rate with {} spectra".format(self.spectra))
            print("channel: {} || mass: {} || profile: {} || process: {}\n".format(self.channel, self.mass, self.profile, self.process))
            
            spectra_dict = self.ComputeSpectra()
            jfactor = self.ComputeJfactor()

            # Compute the rate as Spectra x Jfactor for each neutrino flavours
            self.hist['TrueRate'] = TrueRate(spectra_dict, jfactor)
            if self.process=='ann':
                factor = (1./(2 * 4*math.pi * self.mass**2))* self.xsec
            elif self.process=='decay':
                factor = (1./(4*math.pi * self.mass * self.tdecay))
            for nu_type in self.hist['TrueRate'].keys():
                self.hist['TrueRate'][nu_type] *= factor
        return self.hist['TrueRate']

    def GetMC(self):
        print("*"*20)
        print("Accessing MC set {}".format(self.set))

        self.MCdict = ExtractMC(['14'+self.set, '12'+self.set, '16'+self.set])
        return self.MCdict

    def ComputeResp(self):
        print("*"*20)
        print("Computing Response Matrix")

        if self.hist['Resp'] is None:
            if self.PreCompResp:
                if self.interpolate_resp:
                    self.hist['Resp'] = RespMatrix_Interpolated(self.set, self.bin, Scramble=self.Scramble)
                else:
                    print("*"*20)
                    print('use directly the precomp grid of response matrix without any interpolation')
                    print('Not recommend for neutrino line since you might miss the monochomatic peak')
                  
                    respdict = pkl.load(open("{}/DetResponse/Resp_MC{}_logE.pkl".format(data_path, self.set), "rb"))
                    self.bin = respdict['Bin']
                    gridEtrue = np.meshgrid(self.bin['true_psi_center'], self.bin['true_energy_center'], 
                                    self.bin['reco_psi_center'], self.bin['reco_energy_center'], indexing='ij')[1]               
            
                    # precomputed response as dR/d(E_true), but the true E bin is in log -> change to dR/dlogE = E*dR/dE
                    if self.Scramble:                        
                        self.hist['Resp'] = respdict['Resp_Scr']
                    else:
                        self.hist['Resp'] = respdict['Resp']
                    for nu_type in self.hist['Resp'].keys():
                        self.hist['Resp'][nu_type] *= gridEtrue

            else:
                print('Compute Response Matrix from scratch (will take ~6-7 minutes!)')
                MC = self.GetMC()
                self.hist['Resp'] = KDE_RespMatrix(MC, self.bin, 'ISJ', maxEtrue=self.mass*1.25, maxEreco=1000, Scramble=self.Scramble)

            # Renormalize to the total weight (equal to the numerical integration of dR/dx in case equal binning in x)
            MCcut = self.GetMC()
            for nu_type in ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]:
                pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}
                loc_norm = np.where(  (MCcut["nutype"]==pdg_encoding[nu_type])
                    & (MCcut["E_reco"] <= np.max(self.bin["reco_energy_edges"]))
                    & (MCcut["E_reco"] >= np.min(self.bin["reco_energy_edges"]))
                    & (MCcut["E_true"] <= np.max(self.bin["true_energy_edges"]))
                    & (MCcut["E_true"] >= np.min(self.bin["true_energy_edges"]))
                        )
                w_norm = MCcut["w"][loc_norm]                
                self.hist['Resp'][nu_type] = self.hist['Resp'][nu_type]/(np.sum(self.hist['Resp'][nu_type]))* (np.sum(w_norm))
        else:
            print('Response Matrix already computed, will not compute it again')
        return self.hist['Resp']

    def ComputeRecoRate(self):
        print("*"*20)
        print("Buiding final reco rate using {} method".format(self.type))

        if self.type=='evtbyevt':
            MC = self.GetMC()
            Nu = NuSpectra(self.mass, self.channel, self.process)
            Nu.nodes=200
            Nu.bins=200
            #Different treatment of nu and anti-nu spectra
            if "PPPC4" in self.spectra:
                spectra_dict = Nu.SpectraPPPC4_AvgOsc()
            elif "Charon" in self.spectra:
                spectra_dict = Nu.SpectraCharon_nuSQUIDS()
            print('Computing Jfactor with default option: precomputed Clumpy file')
            MyJ = Jf(process=self.process, profile=self.profile)
            Jfactor = MyJ.Jfactor_Clumpy()
            self.hist['RecoRate']= KDE_evtbyevt(MC, spectra_dict, Jfactor, self.mass, 'ISJ', self.bin, Scramble=self.Scramble)
        
        elif self.type=='Resp':
            Resp = self.ComputeResp()                        
            Rate = self.ComputeTrueRate()
            self.hist['RecoRate'] = np.zeros((len(self.bin['reco_psi_center']), len(self.bin['reco_energy_center'])))
            for nutype in ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]:
                self.hist['RecoRate'] += np.tensordot(Resp[nutype], Rate[nutype], axes=([0,1], [0,1]))
            # self.hist['RecoRate'] *= (1./(2 * 4*math.pi * self.mass**2)) move this factor to truerate

        else:
            print("ERROR: Choose self.type among evtbyevt, Resp, and Resp_without_interpolation")
            sys.exit(1)
    
        return self.hist['RecoRate']

    def ResetAllHists(self):
        self.hist['Spectra'] = None
        self.hist['Jfactor'] = None
        self.hist['TrueRate'] = None
        self.hist['Resp'] = None
        self.hist['RecoRate'] = None
