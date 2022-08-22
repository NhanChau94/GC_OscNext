"""
author : N. Chau
Making the detector response functions (eff area, resolution function and PID prob)
"""
import numpy as np
import pickle as pkl

##---------------------------------------------##
##Load the MC dictionnary and applied cut
##Required:
##  -  MC dictionary
##Output:
##  -  array of useful variables for making response functions
##  -  open angle also transfered to degree
##---------------------------------------------##

def ApplyCut(inMC, cut="OscNext"):

    # Applying cuts
    if cut == "Default":
        loc = np.where((inMC["L7OscNext_bool"]==1) &
                       (inMC["true_Energy"]<=HE_cut))
    else:
        loc = np.where((inMC["L7muon_classifier_up"]>0.4) &
                       (inMC["L4noise_classifier"]>0.95) &
                       (inMC["L7reco_vertex_z"]>-500.) &
                       (inMC["L7reco_vertex_z"]<-200.) &
                       (inMC["L7reco_vertex_rho36"]<300.) &
                       (inMC["L5nHit_DOMs"]>2.5) &
                       (inMC["L7_ntop15"]<2.5) &
                       (inMC["L7_nouter"]<7.5) &
                       (inMC["L7reco_time"]<14500.) 
                        # &(inMC["true_Energy"]<=HE_cut)
                        )

    # Output: samples of particle types, true E,  true psi; PID, reco_E, reco_psi and weights
    output_dict = dict()
    output_dict["nutype"] = inMC["PDG_encoding"][loc]
    output_dict["E_true"] = inMC["true_Energy"][loc]
    output_dict["E_reco"] = inMC["reco_TotalEnergy"][loc]
    output_dict["psi_true"] = np.rad2deg(inMC["true_psi"][loc])
    output_dict["psi_reco"] = np.rad2deg(inMC["reco_psi"][loc])
    output_dict["PID"] = inMC["PID"][loc]

    # weight
    OW = inMC["OneWeight"][loc]
    NEvents = inMC["NEvents"][loc]
    ratio = inMC["gen_ratio"][loc]
    NFiles = inMC["NFiles"]
    genie_w = OW * (1./ratio) * (1./(NEvents*NFiles))

    output_dict["w"] = genie_w

    return output_dict



def ExtractMC():
    # Extract Simulation file:
    Sim12 = pkl.load(open("../Sample/Simulation/OscNext_Level7_v02.00_120000_pass2_variables_NoCut.pkl", "rb"))
    Sim14 = pkl.load(open("../Sample/Simulation/OscNext_Level7_v02.00_140000_pass2_variables_NoCut.pkl", "rb"))
    Sim16 = pkl.load(open("../Sample/Simulation/OscNext_Level7_v02.00_160000_pass2_variables_NoCut.pkl", "rb"))
    # Sim = [Sim12['120000'], Sim14['140000'], Sim16['160000']]

    Cut = [ApplyCut(Sim12['120000']), ApplyCut(Sim14['140000']), ApplyCut(Sim16['160000'])]
    MCcut = dict()
    for key in Cut[0].keys():
        MCcut[key] = np.array([])
        for c in Cut:
            MCcut[key] = np.concatenate((MCcut[key], c[key]), axis=None) 

    return MCcut



##---------------------------------------------##
##Create binning
##Required:
##  - array of bin edges and bin center
##
##Output:
##  - dictionary of bin edges and center
##---------------------------------------------##
def GroupBinning(true_energy_edges, true_psi_edges, true_energy_center, true_psi_center,
                 reco_energy_edges, reco_psi_edges, reco_energy_center, reco_psi_center, PID_edges, PID_center):

    Bin = dict()
    Bin['true_energy_edges'] = true_energy_edges
    Bin['true_psi_edges'] = true_psi_edges
    Bin['true_energy_center'] = true_energy_center
    Bin['true_psi_center'] = true_psi_center

    Bin['reco_energy_edges'] = reco_energy_edges
    Bin['reco_psi_edges'] = reco_psi_edges
    Bin['reco_energy_center'] = reco_energy_center
    Bin['reco_psi_center'] = reco_psi_center

    Bin['PID_edges'] = PID_edges
    Bin['PID_center'] = PID_center

    return Bin


##---------------------------------------------##
##Compute effective area
##Required:
##  - MC dictionary
##  - bin edges [true psi, true energy]
##  - KDE: if using KDE to smooth the function
##Output:
##  - histogram of eff area: [i][j]: eff values at bin i, j of true psi and true E
##---------------------------------------------##
def EffectiveArea(MCdict, bin, KDE=False):

    psi_edges = bin['true_psi_edges']
    energy_edges = bin['true_energy_edges']

    psi = MCdict["psi_true"]
    E = MCdict["E_true"]
    weight = MCdict["w"]
    H, v0_edges, v1_edges = np.histogram2d(psi, E,
                                               bins = (psi_edges, energy_edges),
                                               weights=weight)
    Aeff = H

    return Aeff

##---------------------------------------------##
##Energy Resolution
##Required:
##  - MC dictionary
##  - bin edges [true psi, true energy, reco energy]
##  - KDE: if using KDE to smooth the function
##Output:
##  - histogram of resolution: [i][j][k]: eff values at bin i, j, k of true psi, true E and reco E
##---------------------------------------------##

def EnergyResolution(MCdict, bin, KDE=False):
    # nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    # pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}

    truepsi_edges = bin['true_psi_edges']
    trueenergy_edges = bin['true_energy_edges']
    recoenergy_edges = bin['reco_energy_edges']

        # loc = np.where(MCdict["nutype"]==pdg_encoding[nu_type])
    psitrue = MCdict["psi_true"]
    Etrue = MCdict["E_true"]
    Ereco = MCdict["E_reco"]
    Ntot = len(Etrue)
    H, v0_edges, v1_edges, v2_edges = np.histogramdd((psitrue, Etrue, Ereco),
                                           bins = (truepsi_edges, trueenergy_edges, recoenergy_edges))
    Resolution = H/Ntot
    return Resolution


##---------------------------------------------##
##Psi Resolution
##Required:
##  - MC dictionary
##  - bin edges [true psi, true energy, reco energy]
##  - KDE: if using KDE to smooth the function
##Output:
##  - histogram of resolution: [i][j][k]: eff values at bin i, j, k of true E, true psi and reco psi
##---------------------------------------------##

def PsiResolution(MCdict, bin, KDE=False):
    # nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    # pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}

    truepsi_edges = bin['true_psi_edges']
    trueenergy_edges = bin['true_energy_edges']
    recopsi_edges = bin['reco_psi_edges']

    psitrue = MCdict["psi_true"]
    Etrue = MCdict["E_true"]
    psireco = MCdict["psi_reco"]
    Ntot = len(Etrue)
    H, v0_edges, v1_edges, v2_edges = np.histogramdd((Etrue, psitrue, psireco),
                                           bins = (trueenergy_edges, truepsi_edges, recopsi_edges))
    Resolution = H/Ntot
    return Resolution


##---------------------------------------------##
##PID probability
##Required:
##  - MC dictionary
##  - bin edges [true psi, true energy, reco energy]
##  - KDE: if using KDE to smooth the function
##Output:
##  - histogram of PID prob: [i][j][k]: values at bin i, j ,k as: PID, true psi and true E
##---------------------------------------------##

def PIDprob(MCdict, bin, KDE=False):
    psi_edges = bin['true_psi_edges']
    energy_edges = bin['true_energy_edges']
    PID_edges = bin['PID_edges']

    psi = MCdict["psi_true"]
    E = MCdict["E_true"]
    PID = MCdict["PID"]
    Ntot = len(E)
    PIDprob, v0_edges, v1_edges = np.histogram2d(PID, psi, E,
                                            bins = (PID_edges, psi_edges, energy_edges))
    return PIDprob/Ntot


##---------------------------------------------##
##Make detector response functions
##Required:
##  - MC dictionary
##  - bin edges [true psi, true energy, reco energy]
##  - KDE: if using KDE to smooth the function
##Output:
##  - dictionary of response function: [nutype][functions]
##---------------------------------------------##


def MakeResponseFunctions(MCdict, bin, outfile, KDE=False):
    # Apply cuts on MC:
    MCcut = ApplyCut(MCdict)

    # Doing response functions for each neutrino and interaction type:
    nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}

    Resp = dict()
    Resp['Bin'] = bin

    for nu_type in nu_types:
        loc = np.where(MCdict["nutype"]==pdg_encoding[nu_type])
        MC_nutype = MCdict[loc]
        Resp[nu_type]['Aeff'] = EffectiveArea(MC_nutype, bin, KDE)
        Resp[nu_type]['Eres'] = EnergyResolution(MC_nutype, bin, KDE)
        Resp[nu_type]['Psires'] = PsiResolution(MC_nutype, bin, KDE)
        Resp[nu_type]['PIDprob'] = PIDprob(MC_nutype, bin, KDE)

    pkl.dump(Resp,open(outfile ,"wb"))
    return Resp

##---------------------------------------------##
##Make one correlated response matrix
##Required:
##  - MC dictionary
##  - binning
##  - KDE: if using KDE to smooth the function
##Output:
##  - dictionary of response matrix as: [nutype][truePsi][trueE][PID][recoPsi][recoE]
##---------------------------------------------##
def MakeResponseMatrix(MCcut, bin, outfile, KDE=False):
    # Apply cuts on MC:
    # MCcut = ApplyCut(MCdict)

    # Binning:
    truepsi_edges = bin['true_psi_edges']
    trueenergy_edges = bin['true_energy_edges']

    recopsi_edges = bin['reco_psi_edges']
    recoenergy_edges = bin['reco_energy_edges']
    PIDedges = bin['PID_edges']


    # Doing response functions for each neutrino and interaction type:
    nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}
    RespMatrix = dict()
    RespMatrix['Bin'] = bin
    for nu_type in nu_types:
        loc = np.where(MCcut["nutype"]==pdg_encoding[nu_type])
        # MC_nutype = MCcut[loc]

        psitrue = MCcut["psi_true"][loc]
        Etrue = MCcut["E_true"][loc]
        psireco = MCcut["psi_reco"][loc]
        Ereco = MCcut["E_reco"][loc]
        PIDscore = MCcut["PID"][loc]
        w = MCcut["w"][loc]
        # Response matrix for each nutype:
        Resp = np.histogramdd((psitrue, Etrue, PIDscore, psireco, Ereco),
                                            bins = (truepsi_edges, trueenergy_edges, 
                                                    PIDedges, recopsi_edges, recoenergy_edges), 
                                            weights=w)
        RespMatrix[nu_type] = np.array(Resp[0])                                    
    pkl.dump(RespMatrix, open(outfile, "wb"))                                        
    return RespMatrix
