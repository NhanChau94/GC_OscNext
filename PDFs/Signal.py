"""
author : N. Chau
Creating signal event distribution + pdfs with a detector response
"""
import sys
import math
import pickle as pkl
import numpy as np
import scipy
from scipy.interpolate import pchip_interpolate

sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Utils/")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Spectra/")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DetResponse/")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/PDFs/")
from Detector import *
from Utils import *
from Interpolate import *
from NuSpectra import *
from KDE_implementation import *

# cut low values to zero in case needed
def cutspectra(spec, cut):
    for flv in ["nu_e", "nu_mu", "nu_tau"]:
        for i, val in enumerate(spec[flv]["dNdE"]):
            if val<cut:
                spec[flv]["dNdE"][i] = 0
    return spec



##---------------------------------------------##
##Interpolate the Jfactor at the psi values used in response functions
##Required:
##  -  Precomputed Jfactor file
##  -  psi values for interpolation
##Output:
##  -  Interpolated Jfactor
##---------------------------------------------##

def Interpolate_Jfactor(inpath, psival):
    #Open file
    Jfactor = pkl.load(open(inpath,"rb"))

    y_interp = scipy.interpolate.splrep(Jfactor["psi"], Jfactor["J"])
    interp_Jpsi = scipy.interpolate.splev(psival, y_interp, der=0)

    return interp_Jpsi


def Interpolate_Spectra(type, Eval, channel, mass, process="ann"):
    Rate = NuSpectra(mass, channel, process)
    Rate.nodes=200
    Rate.bins=200
    #Different treatment of nu and anti-nu spectra
    if "PPPC4" in type:
        #No distinction is made between neutrinos and anti-neutrinos in PPPC4 tables
        nu_types = ["nu_e", "nu_mu", "nu_tau"]
        spectra = Rate.SpectraPPPC4_AvgOsc()
        # pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16}
    elif "Charon" in type:
        #Charon can compute separately nu and anitnu but they seems to be the same in case of Galactic Center
        spectra = Rate.SpectraCharon_nuSQUIDS()
        nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
        # pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}


    #Define array holding interpolated values of spectra
    interp_dNdE = dict()

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
        if mass < 100 and ("Charon" in type) and (channel=="bb" or channel=="nunu" or channel=="nuenue"
                        or channel=="numunumu" or channel=="nutaunutau"):
            # zeros = np.zeros(len(Eval))
            #Actually interpolate the spectra for our energy array
            #Only interpolate for the proper nu_type & above the energy threshold of the spectra
            neg_v = np.where(interp_dNdE[nu_type]<1e-5)[0]
            interp_dNdE[nu_type][neg_v] = 0.
        # put to zero for energy > mass
        loc = np.where(Eval>mass)
        interp_dNdE[nu_type][loc] = 0.

    if "PPPC4" in type: # in PPPC4 nu and nubar is the same so we just duplicate here (in fact Charon shows the same)
        interp_dNdE["nu_e_bar"] = interp_dNdE["nu_e"]
        interp_dNdE["nu_mu_bar"] = interp_dNdE["nu_mu"]
        interp_dNdE["nu_tau_bar"] = interp_dNdE["nu_tau"]


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





def KDE_RespMatrix(MCcut, Bin, bw_method, maxEtrue=3000, maxEreco=1000, Scramble=False, mirror=True):

    #Evaluate points:
    print("Preparing evaluation grid") 
    ##Equal spacing in the final variables: reco Psi & log10(E_reco), true psi and true E
    trueEeval, recoEeval, truePsieval, recoPsieval = Extend_EvalPoints(Bin["true_energy_center"], Bin["reco_energy_center"], maxEtrue, maxEreco, Bin["true_psi_center"], Bin["reco_psi_center"])  
    print('Etrue: {}'.format(trueEeval))
    print('Ereco: {}'.format(recoEeval))
    print('Psitrue: {}'.format(truePsieval))
    print('Psireco: {}'.format(recoPsieval))
    

    g_psi_true, g_energy_true, g_psi_reco, g_energy_reco = np.meshgrid(truePsieval, trueEeval,
                                                            recoPsieval, recoEeval, indexing='ij')                      
    psi_eval_true = g_psi_true.flatten()
    E_eval_true = g_energy_true.flatten()
    psi_eval_reco = g_psi_reco.flatten()
    E_eval_reco = g_energy_reco.flatten()

    ##Evaluate the KDE in log(Psi)-log10E
    # psiE_eval = np.vstack([np.log(psi_eval_true), E_eval_true, 
    #                     np.log(psi_eval_reco), np.log10(E_eval_reco)])
    psiE_eval = np.vstack([psi_eval_true, E_eval_true, 
            psi_eval_reco, np.log10(E_eval_reco)])
    # psiE_eval = np.vstack([psi_eval_true, E_eval_true, 
    #         psi_eval_reco, E_eval_reco])  


    # Separate MC by each channel nutype->PID:
    nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    # nu_types = ["nu_mu"]

    pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}
    # PID = [[0.,0.5],[0.5, 0.85],[0.85, 1]]
    PID = [[0.,1.]]
    Resp = dict()
    pidbin = 0
    for pid in PID:
        print("Computing {} PID bin".format(pid))
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
            #Needs to be divided by evaluation angle
            # kde_weight = kde_w.reshape(psi_eval_true.shape)
            # kde_weight = kde_w/(psi_eval_true * psi_eval_reco)
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


def RespMatrix_Interpolated(MCset, Bin, Scramble=False, logEtrue=True):
    Evaltrue = Bin['true_energy_center']
    Evalreco = Bin['reco_energy_center']
    Psievaltrue = Bin['true_psi_center']
    Psievalreco = Bin['reco_psi_center']

    # Access precomputed response matrix and its grid
    if logEtrue:
        indict = pkl.load(open("/data/user/tchau/Sandbox/GC_OscNext/DetResponse/PreComp/Resp_MC{}_logE.pkl".format(MCset), "rb"))
    else:
        indict = pkl.load(open("/data/user/tchau/Sandbox/GC_OscNext/DetResponse/PreComp/Resp_MC{}.pkl".format(MCset), "rb"))
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
            Resp_interpolated[nu] = EqualGridInterpolator((psitrue, np.log10(Etrue), psireco, np.log10(Ereco)), Resp[nu], order=1)(np.meshgrid(Psievaltrue, np.log10(Evaltrue), Psievalreco, np.log10(Evalreco),  indexing='ij'))
        else:
            Resp_interpolated[nu] = EqualGridInterpolator((psitrue, Etrue, psireco, np.log10(Ereco)), Resp[nu], order=1)(np.meshgrid(Psievaltrue, Evaltrue, Psievalreco, np.log10(Evalreco),  indexing='ij'))
    
    return Resp_interpolated


def DataHist(Bin, sample='burnsample'):
    if sample=='burnsample':
        dat_dir = "/data/user/niovine/projects/DarkMatter_OscNext/Samples/OscNext/L7/Burnsample/"
        input_files = []
        # Take all burnsample:
        for year in range(2012, 2021):
            infile = dat_dir + "OscNext_Level7_v02.00_burnsample_{}_pass2_variables_NoCut.pkl".format(year)
            print('Loading file: ')
            print(infile)
            print('')
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
    H, v0_edges, v1_edges = np.histogram2d(array_recopsi, array_recoE,
                            bins = (Psireco_edges, Ereco_edges))

    return H





#############################################################################
# Set of functions for evt by evt reweight
#############################################################################




#---------------------------------------------------------------------
#Define cut on weight
#---------------------------------------------------------------------
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

# Compute weights and extract other informations used for evt-by-evt reweight:
def ComputeWeight(MCdict, SpectraPath, JfactorPath, channel, mass, process="ann", maxE=2000, weight_cut=True):
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
        dNdE = Interpolate_Spectra(SpectraPath, true_E, channel, mass, process)

        ##Jfactor interpolation##
        #NOTE: input psi in deg!
        true_psi = MCdict["psi_true"][loc]
        Jpsi = Interpolate_Jfactor(JfactorPath, true_psi)

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

def KDE_evtbyevt(MCdict, SpectraPath, JfactorPath, channel, mass, bw_method, Bin, Scramble=False, weight_cut=True, mirror=True, process='ann'):
    array_PID, array_recopsi, array_recoE, signal_w, array_recoRA, array_recoDec = ComputeWeight(MCdict, SpectraPath, JfactorPath, channel, mass, weight_cut=weight_cut, process=process)
    # Define PID cut:
    # PID = [[0.,0.5],[0.5, 0.85],[0.85, 1]]
    PID = [[0, 1]]


    #Evaluate points:
    print("Preparing evaluation grid") 
    ##Equal spacing in the final variables: reco Psi & log10(E), true psi and true E
    maxE = 2000.
    trueEeval, recoEeval, truePsieval, recoPsieval = Extend_EvalPoints(Bin["true_energy_center"], Bin["reco_energy_center"], maxE, maxE, Bin["true_psi_center"], Bin["reco_psi_center"])  

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


class RecoRate:
    """docstring for RecoRate."""
    def __init__(
            self,
            channel, 
            mass, 
            profile, 
            bin, 
            process='ann', 
            type='Resp', 
            PreCompResp=True,
            spectra='Charon', 
            set='0000', 
            Scramble=False
            ):
        self.channel = channel
        self.mass = mass
        self.profile = profile
        self.bin = bin
        self.process = process
        self.type = type
        self.PreCompResp = PreCompResp
        self.spectra = spectra
        self.set = set
        self.Scramble = Scramble
        self.MCdict = None
        self.hist = dict()
        # self.hist['Spectra'] = None
        # self.hist['Jfactor'] = None
        # self.hist['TrueRate'] = None
        # self.hist['Resp'] = None
        # self.hist['RecoRate'] = None

    def ComputeTrueRate(self):
        print("*"*20)
        print("Computing true rate with {} spectra".format(self.spectra))
        print("channel: {} || mass: {} || profile: {} || process: {}\n".format(self.channel, self.mass, self.profile, self.process))


        # Precomputed Jfactor:
        pathJfactor="/data/user/tchau/Sandbox/GC_OscNext/Spectra/PreComp/JFactor_{}.pkl".format(self.profile)

        # Extract true rate:
        # Jfactor:
        self.hist['Jfactor'] = Interpolate_Jfactor(pathJfactor, self.bin['true_psi_center'])
        # Spectra:
        self.hist['Spectra'] = Interpolate_Spectra(self.spectra, self.bin['true_energy_center'], self.channel, self.mass, process=self.process)

        # Compute the rate as Spectra x Jfactor
        self.hist['TrueRate'] = TrueRate(self.hist['Spectra'], self.hist['Jfactor'])
        return self.hist['TrueRate']

    def GetMC(self):
        print("*"*20)
        print("Accessing MC set {}".format(self.set))

        self.MCdict = ExtractMC(['14'+self.set, '12'+self.set, '16'+self.set])
        return self.MCdict

    def ComputeResp(self):
        print("*"*20)
        print("Computing Response Matrix")

        if self.PreCompResp:
            self.hist['Resp'] = RespMatrix_Interpolated(self.set, self.bin, Scramble=self.Scramble)
        else:
            print('Compute Response Matrix from scratch (will take ~6-7 minutes!)')
            MC = self.GetMC()
            self.hist['Resp'] = KDE_RespMatrix(MC, self.bin, 'ISJ', maxEtrue=self.mass*1.25, maxEreco=1000, Scramble=self.Scramble)

        # Renormalize to the total weight 
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

        return self.hist['Resp']

    def ComputeRecoRate(self):
        print("*"*20)
        print("Buiding final reco rate using {} method".format(self.type))

        if self.type=='evtbyevt':
            MC = self.GetMC()
            pathJfactor="/data/user/tchau/Sandbox/GC_OscNext/Spectra/PreComp/JFactor_{}.pkl".format(self.profile)
            self.hist['RecoRate']= KDE_evtbyevt(MC, self.spectra, pathJfactor, self.channel, self.mass, 'ISJ', self.bin, Scramble=self.Scramble)
        
        elif self.type=='Resp':
            Rate = self.ComputeTrueRate()
            Resp = self.ComputeResp()                        
            self.hist['RecoRate'] = np.zeros((len(self.bin['reco_psi_center']), len(self.bin['reco_energy_center'])))
            for nutype in ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]:
                self.hist['RecoRate'] += np.tensordot(Resp[nutype], Rate[nutype], axes=([0,1], [0,1]))
            self.hist['RecoRate'] *= (1./(2 * 4*math.pi * self.mass**2))
    
        else:
            print("ERROR: Choose self.type among evtbyevt and Resp")
            sys.exit(1)
    
        return self.hist['RecoRate']