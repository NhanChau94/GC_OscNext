"""
author : N. Chau
Collective tools and functions for generating DM interaction (annihilation, decay) rate from GC
note: part of code comes from N. Iovine's analysis on DM search from Galactic Center with OscNext
"""

import os,time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import pickle as pkl

#Charon
import sys
# sys.path.append("/data/user/niovine/software/charon/charon")
from charon import propa
import charon.physicsconstants as PC
pc = PC.PhysicsConstants()


def open_PPPC4tables(filename, channel_pos, mass):
    PPPC4 = dict()
    f = open(filename).readlines()
    energy = []
    spectrum = []
    n=0
    for line in f :
        #Get the content of each line in a list
        line_split = line.split()
        if line_split[0] == "mDM":
            continue
        else :
            if line_split[0] == str(mass):
                #Convert tables format in E and dNdE
                E_log = float(line_split[1])+math.log10(mass)
                energy.append(pow(10,E_log))
                dNdE = float(line_split[channel_pos])*0.05/((pow(10,0.05)-1)*pow(10,E_log))
                spectrum.append(dNdE)
            else:
                continue
    PPPC4["E"] = np.array(energy)
    PPPC4["dNdE"] = np.array(spectrum)
    return PPPC4



#Retrieve PPPC4 Spectra (Cirelli) for all masses and channels considered
def extract_PPPC4tables(channels, masses):
    
    ###  Cirelli tables ###
    PPPC4_values = dict()
    for channel in channels:
        PPPC4_values[channel] = dict()
        
        for mass in masses:
            PPPC4_values[channel][str(mass)] = dict()
    
            #The columns are: mDM, Log10x, dN/dLog10x (for 28 primary channels)
            #The ones we will be considering are mumu, tautau, bb, WW, ZZ, gg, nunu
            EW_list = ["mDM", "E", "eLeL","eReR","ee","muLmuL","muRmuR","mumu","tauLtauL","tauRtauR","tautau", "qq", "cc", "bb", "tt",
                       "WLWL","WTWT","WW", "ZLZL","ZTZT","ZZ", "gg", "gammagamma", "hh", "nuenue","numunumu","nutaunutau","VVe", "VVmu","VV_tau"]
            
            #Nunu channel 
            #The contribution of the 3 flavours at production are considered as 3 differents annihilation channels
            #We regroup annihilation through nuenue, numunumu and nutaunutau as nunu
            if channel == "nunu":
                #Define Dictionnary for each neutrino flavour
                PPPC4_values[channel][str(mass)]['nu_e'] = dict()
                PPPC4_values[channel][str(mass)]['nu_mu'] = dict()
                PPPC4_values[channel][str(mass)]['nu_tau'] = dict()
                
                #Temporary dictionnaries
                nu_e = dict()
                nu_mu = dict()
                nu_tau = dict()
                for tmp_channel in ["nuenue", "numunumu", "nutaunutau"]:
                    channel_pos = EW_list.index(tmp_channel)
                    nu_e[tmp_channel] = open_PPPC4tables("./PPPC4_table/AtProduction_neutrinos_e.dat", channel_pos, mass)
                    nu_mu[tmp_channel] = open_PPPC4tables("./PPPC4_table/AtProduction_neutrinos_mu.dat", channel_pos, mass)
                    nu_tau[tmp_channel] = open_PPPC4tables("./PPPC4_table/AtProduction_neutrinos_tau.dat", channel_pos, mass)
                
                #Sum the 3 contributions and devided by 3 in order to get a single nunu channel   
                PPPC4_values[channel][str(mass)]['nu_e']["dNdE"] = np.add(np.add(nu_e["nuenue"]["dNdE"],nu_e["numunumu"]["dNdE"]),nu_e["nutaunutau"]["dNdE"])/3.
                PPPC4_values[channel][str(mass)]['nu_e']["E"] = nu_e["nuenue"]["E"]
                PPPC4_values[channel][str(mass)]['nu_mu']["dNdE"] = np.add(np.add(nu_mu["nuenue"]["dNdE"],nu_mu["numunumu"]["dNdE"]),nu_mu["nutaunutau"]["dNdE"])/3.
                PPPC4_values[channel][str(mass)]['nu_mu']["E"] = nu_mu["nuenue"]["E"]
                PPPC4_values[channel][str(mass)]['nu_tau']["dNdE"] = np.add(np.add(nu_tau["nuenue"]["dNdE"],nu_tau["numunumu"]["dNdE"]),nu_tau["nutaunutau"]["dNdE"])/3.
                PPPC4_values[channel][str(mass)]['nu_tau']["E"] = nu_tau["nuenue"]["E"]

            else :
                channel_pos = EW_list.index(channel)
                PPPC4_values[channel][str(mass)]['nu_e'] = open_PPPC4tables("./PPPC4_table/AtProduction_neutrinos_e.dat", channel_pos, mass)
                PPPC4_values[channel][str(mass)]['nu_mu'] = open_PPPC4tables("./PPPC4_table/AtProduction_neutrinos_mu.dat", channel_pos, mass)
                PPPC4_values[channel][str(mass)]['nu_tau'] = open_PPPC4tables("./PPPC4_table/AtProduction_neutrinos_tau.dat", channel_pos, mass)
        
    pkl.dump(PPPC4_values, open("./Spectra_ann_PPPC4_atSource.pkl", "wb"))
    return PPPC4_values


def oscillate_PPPC4(channels, masses, nu_types):
    
    oscillated_PPPC4 = dict()
    
    #Get Cirelli Spectra
    PPPC4_spectra = pkl.load(open("./Spectra_ann_PPPC4_atSource.pkl","rb"))
    
    for channel in channels:
        print ("Doing channel:", channel)
        
        oscillated_PPPC4[channel] = dict()

        for mass in masses:
            
            oscillated_PPPC4[channel][str(mass)] = dict()

            #Oscillate neutrinos
            osc_nu = oscillate_spectra(PPPC4_spectra[channel][str(mass)], ["nu_mu","nu_e", "nu_tau"])
            
            #Spectra after oscillation
            for nu_type in nu_types:
                oscillated_PPPC4[channel][str(mass)][nu_type] = dict()
                oscillated_PPPC4[channel][str(mass)][nu_type]["E"] = PPPC4_spectra[channel][str(mass)][nu_type]["E"]
                oscillated_PPPC4[channel][str(mass)][nu_type]["dNdE"] = osc_nu["dNdE_"+nu_type+"_osc"]
    
    #Will be saved as 
    pklfile = "./Spectra_ann_PPPC4_atEarth.pkl"
    pkl.dump(oscillated_PPPC4, open(pklfile, "wb"))
    return oscillated_PPPC4




#---------------------------------------------------------------------
##Oscillate spectra##
#---------------------------------------------------------------------
#Computing the PNMS matrices Uij for the oscillation matrix
def PMNS_matrix(t12, t13, t23, d):
    s12 = np.sin(t12)
    c12 = np.cos(t12)
    s23 = np.sin(t23)
    c23 = np.cos(t23)
    s13 = np.sin(t13)
    c13 = np.cos(t13)
    cp  = np.exp(1j*d)
    
    return np.array([[ c12*c13, s12*c13, s13*np.conj(cp)],
                  [-s12*c23 - c12*s23*s13*cp, c12*c23 - s12*s23*s13*cp, s23*c13],
                  [ s12*s23 - c12*s23*s13*cp,-c12*s23 - s12*c23*s13*cp, c23*c13]])

#Probability of flavor to change when L->inf
def prob(a, b, U):
    #Gives the oscillation probability for nu(a) -> nu(b)
    #for PMNS matrix U, and L in km and E in GeV
    s = 0
    for i in range(3):
            s += (np.conj(U[a,i])*U[b,i]*U[a,i]*np.conj(U[b,i])).real
    return s

#Define Oscillation Matrix
def osc_matrix(U):
    return np.array([[prob(0, 0, U), prob(0, 1, U), prob(0,2,U)],
                     [prob(1, 0, U), prob(1, 1, U), prob(1,2,U)],
                     [prob(2, 0, U), prob(2, 1, U), prob(2,2,U)]])

#Oscillate the spectra
def oscillate_spectra(spectra, nutypes):
    
    oscillated = dict()  
    
    #Define mixing angles theta
    t12 = 0.57596 # Old value: 0.5934
    t13 = 0.1296  # Old value: 0.1514
    t23 = 0.8552  # Old value: 0.785398
    U = PMNS_matrix(t12, t13, t23, 0)
    P = osc_matrix(U)
    
    #print ("Oscillation parameters: theta12={}, theta13={}, theta23={}".format(str(t12), str(t13), str(t23)))
    #print ("Oscillation matrix: ", P)

    dNdE_nue_osc = []
    dNdE_numu_osc = []
    dNdE_nutau_osc = []

    #Apply Oscillation Matrix
    for i in range(len(spectra[nutypes[0]]["dNdE"])):
        dNdE_nue_osc.append(np.dot(P,np.array([spectra[nutypes[0]]["dNdE"][i], 
                                               spectra[nutypes[1]]["dNdE"][i], 
                                               spectra[nutypes[2]]["dNdE"][i]]))[0])
        dNdE_numu_osc.append(np.dot(P,np.array([spectra[nutypes[0]]["dNdE"][i],
                                                spectra[nutypes[1]]["dNdE"][i], 
                                                spectra[nutypes[2]]["dNdE"][i]]))[1])
        dNdE_nutau_osc.append(np.dot(P,np.array([spectra[nutypes[0]]["dNdE"][i], 
                                                 spectra[nutypes[1]]["dNdE"][i], 
                                                 spectra[nutypes[2]]["dNdE"][i]]))[2])

    #Spectra after oscillation
    oscillated["dNdE_"+nutypes[0]+"_osc"] = dNdE_nue_osc
    oscillated["dNdE_"+nutypes[1]+"_osc"] = dNdE_numu_osc
    oscillated["dNdE_"+nutypes[2]+"_osc"] = dNdE_nutau_osc
    
    return oscillated
