"""
author : N. Chau
Creating signal event distribution + pdfs with a prebuilt detector response
"""
import os
import math
import pickle as pkl
import numpy as np
import scipy
from scipy import interpolate
from scipy.interpolate import pchip_interpolate

# cut low values to zero
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


def Interpolate_Spectra(inpath, Eval, channel, mass):
    spectra_file = pkl.load(open(inpath,"rb"))

    #Different treatment of nu and anti-nu spectra
    if "PPPC4" in inpath:
        #No distinction is made between neutrinos and anti-neutrinos in PPPC4 tables
        nu_types = ["nu_e", "nu_mu", "nu_tau"]
        # pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16}
    elif "Charon" in inpath:
        #Charon can compute separately nu and anitnu but they seems to be the same in case of Galactic Center
        nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
        # pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}


    #Define array holding interpolated values of spectra
    interp_dNdE = dict()

    for nu_type in nu_types:
        #Energy
        spectra_E = np.array(spectra_file[channel][str(mass)][nu_type]["E"])
        #Spectra
        spectra_dNdE = np.array(spectra_file[channel][str(mass)][nu_type]["dNdE"])
        #Define fct from which interpolate from
        # y_interp = scipy.interpolate.splrep(spectra_E, spectra_dNdE)
        # interp_dNdE[nu_type] = scipy.interpolate.splev(Eval, y_interp, der=0)
        interp_dNdE[nu_type] = pchip_interpolate(spectra_E, spectra_dNdE, Eval)

        # bb and nunu channel of mass below 100GeV gives weird features due to lack of stat in pythia table of charon
        # set values of spectra below 1e-5 to zero and only interpolate the part >1e-5
        if mass < 100 and ("Charon" in inpath) and (channel=="bb" or channel=="nunu" or channel=="nuenue"
                        or channel=="numunumu" or channel=="nutaunutau"):
            # zeros = np.zeros(len(Eval))
            #Actually interpolate the spectra for our energy array
            #Only interpolate for the proper nu_type & above the energy threshold of the spectra
            neg_v = np.where(interp_dNdE[nu_type]<1e-5)[0]
            interp_dNdE[nu_type][neg_v] = 0.
        # put to zero for energy > mass
        loc = np.where(Eval>mass)
        interp_dNdE[nu_type][loc] = 0.

    if "PPPC4" in inpath: # in PPPC4 nu and nubar is the same so we just duplicate here
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



##---------------------------------------------##
##Import a precomputed detector response
##Required:
##  -  Precomputed detector response file
##
##Output:
##  -  detector response dictionary
##---------------------------------------------##
def ImportDetResponse(inpath):
    DetResponse = pkl.load(open(inpath, "rb"))
    return DetResponse

##---------------------------------------------##
##Compute detected rate by using the response matrix
##Required:
##  -  Expected rate in true psi, energy
##  -  Detector response for each channel: nue, nue bar, numu, numu bar, nutau, nutaubar (CC + NC)
##     + Effective area (true psi, true energy)
##     + Energy resolution (true psi, true E, reco E)
##     + psi resolution (true E, true psi, reco psi)
##     + PID prob (true E, true psi)
##Output:
##  -  Detected rate: dictionary of [PID][reco psi][reco E]
##---------------------------------------------##
def DetectedRate_withRespMatrix(TrueRate, RespMatrix):
    # TrueRate[psi][E] * ResponseMatrix[truePsi][trueE][PID][recoPsi][recoE]] sum on true E and psi
    # -> output: DetectedRate 
    DetectedRate_bynutype = dict()
    for nu_type in RespMatrix.keys():
        if nu_type == 'Bin':
            continue
        DetectedRate_bynutype[nu_type] = np.tensordot(TrueRate[nu_type], RespMatrix[nu_type], 
                                                    axes=([0,1], [0,1]))
        # -> output as Rate[nu_type][PID][recoPsi][recoE]

    # Sum the contribution of each nutype to a PID bin
    DetectedRate_byPID = dict()
    N_PID = len(RespMatrix['Bin']['PID_center'])
    shapeReco = DetectedRate_bynutype['nu_e'][0].shape

    DetectedRate_byPID = np.zeros((N_PID, shapeReco[0], shapeReco[1]))
    for i in range(N_PID):
        for nu_type in DetectedRate_bynutype.keys():
            DetectedRate_byPID[i] = np.add(DetectedRate_byPID[i], DetectedRate_bynutype[nu_type][i])


    Ntot = np.sum(DetectedRate_byPID)
    return DetectedRate_byPID/Ntot





##---------------------------------------------##
##Compute detected rate with event by event reweight
##Required:
##  -  MC events stored in dictionary (psi in deg)
##  -  Path to Spectra and Jfactor precomputed file
##Output:
##  -  Detected rate: dictionary of [PID][reco psi][reco E]
##---------------------------------------------##
def DetectedRate_evtbyevt(MCdict, SpectraPath, JfactorPath, channel, mass, bin):

    nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}

    #PDF_variables
    array_recopsi = np.array([])
    array_recoE = np.array([])
    array_PID = np.array([])
    signal_w = np.array([])

    for nu_type in nu_types:
        loc = np.where( (MCdict["E_true"]<=mass) & (MCdict["nutype"]==pdg_encoding[nu_type]) )
        if len(loc[0])==0:
            continue
        ##Sort all variables by increasing true_E values##
        ##NOTE: this is required for spectra interpolation
        # sort = MCdict["E_true"][loc].argsort()

        ##Simulation weight##
        genie_w = MCdict["w"][loc]

        ##Spectra interpolation##
        true_E = MCdict["E_true"][loc]
        dNdE = Interpolate_Spectra(SpectraPath, true_E, channel, mass)
        # print ("True_E first & last elements:", true_E[0], true_E[-1])
        # print ("True_E min & max:", min(true_E), max(true_E))
        # print ("Negative dNdE values:", np.where(dNdE<0)[0])

        ##Jfactor interpolation##
        #NOTE: input psi in deg!
        true_psi = MCdict["psi_true"][loc]
        Jpsi = Interpolate_Jfactor(JfactorPath, true_psi)
        # print ("True_psi min & max:", min(true_psi), max(true_psi))

        ##Signal weight##
        weight = (1./(2 * 4*math.pi * mass**2)) * genie_w * dNdE[nu_type] * Jpsi
        # print ("Len(weight):",len(weight))

        ##Reco variables:
        reco_psi = MCdict["psi_reco"][loc]
        reco_E = MCdict["E_reco"][loc]
        PID = MCdict["PID"][loc]

        ##group all nutype:
        array_recopsi = np.append(array_recopsi, reco_psi)
        array_recoE = np.append(array_recoE, reco_E)
        array_PID = np.append(array_PID, PID)
        signal_w = np.append(signal_w, weight)

    ##put into histogram:
    H = np.histogramdd((array_PID, array_recopsi, array_recoE), 
                            bins=(bin['PID_edges'], bin['reco_psi_edges'], bin['reco_energy_edges']),
                            weights=signal_w)
    sum_w = np.sum(signal_w)
    return H[0]/sum_w


def DetectedRate_evtbyevt_nopid(MCdict, SpectraPath, JfactorPath, channel, mass, bin):

    nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}

    #PDF_variables
    array_recopsi = np.array([])
    array_recoE = np.array([])
    signal_w = np.array([])

    for nu_type in nu_types:
        loc = np.where( (MCdict["E_true"]<=mass) & (MCdict["nutype"]==pdg_encoding[nu_type]) )
        if len(loc[0])==0:
            continue
        ##Sort all variables by increasing true_E values##
        ##NOTE: this is required for spectra interpolation
        # sort = MCdict["E_true"][loc].argsort()

        ##Simulation weight##
        genie_w = MCdict["w"][loc]

        ##Spectra interpolation##
        true_E = MCdict["E_true"][loc]
        dNdE = Interpolate_Spectra(SpectraPath, true_E, channel, mass)
        # print ("True_E first & last elements:", true_E[0], true_E[-1])
        # print ("True_E min & max:", min(true_E), max(true_E))
        # print ("Negative dNdE values:", np.where(dNdE<0)[0])

        ##Jfactor interpolation##
        #NOTE: input psi in deg!
        true_psi = MCdict["psi_true"][loc]
        Jpsi = Interpolate_Jfactor(JfactorPath, true_psi)
        # print ("True_psi min & max:", min(true_psi), max(true_psi))

        ##Signal weight##
        weight = (1./(2 * 4*math.pi * mass**2)) * genie_w * dNdE[nu_type] * Jpsi
        # print ("Len(weight):",len(weight))

        ##Reco variables:
        reco_psi = MCdict["psi_reco"][loc]
        reco_E = MCdict["E_reco"][loc]
        PID = MCdict["PID"][loc]

        ##group all nutype:
        array_recopsi = np.append(array_recopsi, reco_psi)
        array_recoE = np.append(array_recoE, reco_E)
        signal_w = np.append(signal_w, weight)

    ##put into histogram:
    H = np.histogramdd((array_recopsi, array_recoE), 
                            bins=(bin['reco_psi_edges'], bin['reco_energy_edges']),
                            weights=signal_w)
    sum_w = np.sum(signal_w)
    return H[0]/sum_w




def DetectedRate_allstep(PathSpectra, PathJfactor, channel, mass):
    # Extracting response matrix:
    PathResp = "../DetResponse/PreComp/ResponseMatrix_mass_{}.pkl".format(mass)
    # print(PathResp)
    Resp = pkl.load(open(PathResp, "rb"))

    # Compute the Spectra:
    Etrue_val = Resp['Bin']['true_energy_center']
    Psitrue_val = Resp['Bin']['true_psi_center']
    Spectra = Interpolate_Spectra(PathSpectra, Etrue_val, channel, mass)
    Jfactor = Interpolate_Jfactor(PathJfactor, Psitrue_val)
    evtrate = TrueRate(Spectra, Jfactor)

    # Compute detected rate as TrueRate x Response Matrix
    PDF = DetectedRate_withRespMatrix(evtrate, Resp)

    return PDF



##---------------------------------------------##
##Interpolate the Spectra at the energy values used in response functions
##Required:
##  -  Precomputed Spectra file (may be change to directly computed for Charon?)
##  -  energy values for interpolation
##Output:
##  -  Interpolated spectra
##---------------------------------------------##
# def Interpolate_Spectra(inpath, Eval, channel, mass):
#     spectra_file = pickle.load(open(inpath,"rb"))

#     #Different treatment of nu and anti-nu spectra
#     if "PPPC4" in inpath:
#         #No distinction is made between neutrinos and anti-neutrinos in PPPC4 tables
#         nu_types = ["nu_e", "nu_mu", "nu_tau"]
#         pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16}
#     elif "Charon" in inpath:
#         #Charon can compute separately nu and anitnu but they seems to be the same in case of Galactic Center
#         nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
#         pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}


#     #Define array holding interpolated values of spectra
#     interp_dNdE = dict()

#     for nu_type in nu_types:

#         #Energy
#         spectra_E = np.array(spectra_file[channel][str(mass)][nu_type]["E"])
#         #Spectra
#         spectra_dNdE = np.array(spectra_file[channel][str(mass)][nu_type]["dNdE"])

#         #print ("Energy to interpolate:", spectra_E)
#         #print ("Spectra to interpolate:", min(spectra_dNdE), max(spectra_dNdE))

#         #Define low energy and high energy cut for interpolation
#         #Check if there is zero in spectra (for nunu) and define interpolation cuts accordingly
#         HE_cut = mass
#         zeroes = np.where(spectra_dNdE == 0.)
#         if zeroes != np.array([]):
#             LE_cut = min(spectra_E[np.where(spectra_dNdE!=0.)])+(0.01*HE_cut)
#         else:
#             LE_cut = min(spectra_E)
#         LE_cut = 0.
#         print("Energy cut for {}".format(nu_type))
#         print(LE_cut)
#         print(HE_cut)
#         #print ( "Energy range for interpolation:[{},{}]".format(str(LE_cut),str(HE_cut)) )
#         #print ( "Real interpolation range: [{},{}]".format(str(min(spectra_E)),str(max(spectra_E))) )

#         #Define fct from which interpolate from
#         y_interp = scipy.interpolate.splrep(spectra_E, spectra_dNdE)

#         zeros = np.zeros(len(Eval))
#         #Actually interpolate the spectra for our energy array
#         #Only interpolate for the proper nu_type & above the energy threshold of the spectra
#         interp_dNdE[nu_type] = np.where((Eval>=LE_cut) & (Eval<=HE_cut),
#                                scipy.interpolate.splev(Eval, y_interp, der=0), zeros)


#         #Get rid of negative spectra values due to poor interpolation
#         while len(Eval[np.where(interp_dNdE[nu_type]<0)]) != 0:
#             neg_v = np.where(interp_dNdE[nu_type]<0)[0]
#             interp_dNdE[nu_type][neg_v] = interp_dNdE[nu_type][neg_v-1]

#         print ("Negative values in spectra:", np.where(interp_dNdE[nu_type]<0)[0])

#     return interp_dNdE