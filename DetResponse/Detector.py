"""
author : N. Chau
Making the detector response functions (eff area, resolution function and PID prob)
"""
import numpy as np
import pickle as pkl
import sys
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Utils/")
from Utils import *
##---------------------------------------------##
##Load the MC dictionnary and applied cut
##Required:
##  -  MC dictionary
##Output:
##  -  array of useful variables for making response functions
##  -  open angle also transfered to degree
##---------------------------------------------##

def ApplyCut(inMC, cut="OscNext"):
    HE_cut = 9000.
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
                        &(inMC["true_Energy"]<=HE_cut)
                        )

    # Output: samples of: particle types, true E,  true psi; PID, reco_E, reco_psi, RA, DEC, Solid Angle and weights
    output_dict = dict()
    output_dict["nutype"] = inMC["PDG_encoding"][loc]
    output_dict["E_true"] = inMC["true_Energy"][loc]
    output_dict["E_reco"] = inMC["reco_TotalEnergy"][loc]
    output_dict["psi_true"] = np.rad2deg(inMC["true_psi"][loc])
    output_dict["psi_reco"] = np.rad2deg(inMC["reco_psi"][loc])
    output_dict["PID"] = inMC["PID"][loc]
    output_dict["Dec_true"] = inMC["true_Dec"][loc]
    output_dict["RA_true"] = inMC["true_RA"][loc]
    output_dict["Dec_reco"] = inMC["reco_Dec"][loc]
    output_dict["RA_reco"] = inMC["reco_RA"][loc]
    # output_dict["SA_true"] = 2* np.pi *(1-np.cos(np.deg2rad["psi_true"]))
    # output_dict["SA_reco"] = 2* np.pi *(1-np.cos(np.deg2rad["psi_reco"]))



    # weight
    OW = inMC["OneWeight"][loc]
    NEvents = inMC["NEvents"][loc]
    ratio = inMC["gen_ratio"][loc]
    NFiles = inMC["NFiles"]
    genie_w = OW * (1./ratio) * (1./(NEvents*NFiles))

    output_dict["w"] = genie_w
    output_dict["oneweight"] = inMC["OneWeight"][loc]
    return output_dict


def ExtractMC(sampleid):
    Simdir="/data/user/tchau/DarkMatter_OscNext/Sample/Simulation"
    # Extract Simulation file:
    Cut = dict()
    for sample in sampleid:
        Sim = pkl.load(open("{}/OscNext_Level7_v02.00_{}_pass2_variables_NoCut.pkl".format(Simdir, sample), "rb"))
        Cut[sample] = ApplyCut(Sim[sample])
    MCcut = dict()
    for key in Cut[sample].keys():
        MCcut[key] = np.array([])
        for sample in sampleid:
            MCcut[key] = np.concatenate((MCcut[key], Cut[sample][key]), axis=None) 

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

def Std_Binning(mass, N_Etrue = 100, N_psitrue = 50, N_Ereco=50, N_psireco = 18):

    # Binning:
    # E true
    
    Etrue_center = np.array(np.linspace(1., mass, N_Etrue))
    Ewidth = (mass-1.)/(N_Etrue-1.)
    Etrue_edges = np.array([E - Ewidth/2. for E in Etrue_center])
    Etrue_edges = np.append(Etrue_edges, Etrue_center[-1] + Ewidth/2.)

    # Psi true

    Psitrue_edges = np.linspace(0., 180., N_psitrue+1)
    Psitrue_center = np.array( [(Psitrue_edges[i]+Psitrue_edges[i+1])/2. for i in range(len(Psitrue_edges)-1)] )

    # E reco
    Ereco_edges = pow(10., np.linspace(np.log10(1.), np.log10(1e3), N_Ereco+1))
    Ereco_center = np.array([np.sqrt(Ereco_edges[i]*Ereco_edges[i+1]) for i in range(len(Ereco_edges) - 1)])


    # Psi reco
    Psireco_edges = np.linspace(0., 180., N_psireco+1)
    Psireco_center = np.array( [(Psireco_edges[i]+Psireco_edges[i+1])/2. for i in range(len(Psireco_edges)-1)] )
    # Psireco_center = np.exp(np.linspace(np.log(0.005), np.log(180), 3* N_psireco))


    # PID
    PID_edges = np.array([0.,0.5,0.85,1.])
    PID_center = np.array( [(PID_edges[i]+PID_edges[i+1])/2. for i in range(len(PID_edges)-1)] )

    Bin = GroupBinning(Etrue_edges, Psitrue_edges, Etrue_center, Psitrue_center,
                    Ereco_edges, Psireco_edges, Ereco_center, Psireco_center, PID_edges, PID_center)

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
    H = np.histogramdd((psitrue, Etrue, Ereco),
                                           bins = (truepsi_edges, trueenergy_edges, recoenergy_edges))
    Resolution = H[0]/Ntot
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


##---------------------------------------------##
##Interpolate a precomputed detector response
##Required:
##  - Response matrix + input grid
##  - Evaluation point
##Output:
##  - Interpolated response matrix
##---------------------------------------------##
def InterpolateResponseMatrix(Resp, grid, points):
    # Input grid (bin center):
    Psitrue_center=grid['true_psi_center']
    Etrue_center=grid['true_energy_center']
    Psireco_center=grid['reco_psi_center']
    Ereco_center=grid['reco_energy_center']

    # Evaluation points:
    Psitrue_interp = points['true_psi_center']
    Etrue_interp = points['true_energy_center']
    Psireco_interp = points['reco_psi_center']
    Ereco_interp = points['reco_energy_center']

    # Interpolate the response matrix on desired evaluation points:
    Resp_interp = RegularGrid_4D((Psitrue_center, Etrue_center, Psireco_center, Ereco_center), Resp, 
                    (Psitrue_interp, Etrue_interp, Psireco_interp, Ereco_interp))

    return Resp_interp
    