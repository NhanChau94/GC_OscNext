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
# import charon.physicsconstants as PC
# pc = PC.PhysicsConstants()
curdir=os.path.dirname(os.path.realpath(__file__))

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

#Retrieve PPPC4 Spectra (Cirelli) for a particular mass and channel considered

def PPPC4_flux(mass, channel):
    EW_list = ["mDM", "E", "eLeL","eReR","ee","muLmuL","muRmuR","mumu","tauLtauL","tauRtauR","tautau", "qq", "cc", "bb", "tt",
               "WLWL","WTWT","WW", "ZLZL","ZTZT","ZZ", "gg", "gammagamma", "hh", "nuenue","numunumu","nutaunutau","VVe", "VVmu","VV_tau"]

    PPPC4_values=dict()
    PPPC4_values['nu_e']=dict()
    PPPC4_values['nu_mu']=dict()
    PPPC4_values['nu_tau']=dict()

    if channel not in EW_list:
        NameError("not correct channel")
    if channel == "nunu":
        nu_e = dict()
        nu_mu = dict()
        nu_tau = dict()
        for tmp_channel in ["nuenue", "numunumu", "nutaunutau"]:
            channel_pos = EW_list.index(tmp_channel)
            nu_e[tmp_channel] = open_PPPC4tables("{}/PPPC4_table/AtProduction_neutrinos_e.dat".format(curdir), channel_pos, mass)
            nu_mu[tmp_channel] = open_PPPC4tables("{}/PPPC4_table/AtProduction_neutrinos_mu.dat".format(curdir), channel_pos, mass)
            nu_tau[tmp_channel] = open_PPPC4tables("{}/PPPC4_table/AtProduction_neutrinos_tau.dat".format(curdir), channel_pos, mass)

        PPPC4_values['nu_e']["dNdE"] = sum(nu_e[ch]['dNdE'] for ch in ["nuenue", "numunumu", "nutaunutau"])/3.
        PPPC4_values['nu_e']["E"] = nu_e["nuenue"]["E"]
        PPPC4_values['nu_mu']["dNdE"] = sum(nu_mu[ch]['dNdE'] for ch in ["nuenue", "numunumu", "nutaunutau"])/3.
        PPPC4_values['nu_mu']["E"] = nu_mu["numunumu"]["E"]
        PPPC4_values['nu_tau']["dNdE"] = sum(nu_tau[ch]['dNdE'] for ch in ["nuenue", "numunumu", "nutaunutau"])/3.
        PPPC4_values['nu_tau']["E"] = nu_tau["nutaunutau"]["E"]

    else:
        channel_pos = EW_list.index(channel)
        PPPC4_values['nu_e'] = open_PPPC4tables("{}/PPPC4_table/AtProduction_neutrinos_e.dat".format(curdir), channel_pos, mass)
        PPPC4_values['nu_mu'] = open_PPPC4tables("{}/PPPC4_table/AtProduction_neutrinos_mu.dat".format(curdir), channel_pos, mass)
        PPPC4_values['nu_tau'] = open_PPPC4tables("{}/PPPC4_table/AtProduction_neutrinos_tau.dat".format(curdir), channel_pos, mass)

    return PPPC4_values

#---------------------------------------------------------
# compute averaged oscillation spectra


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



def oscillate_spectra(spectra, nutypes, th12, th13, th23, delta):

    oscillated = dict()

    U = PMNS_matrix(th12, th13, th23, delta)
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




class NuRate:
    """docstring for NuRate."""

    def __init__(
            self,
            mass=100,
            channel="bb",
            process="ann",
            theta12=33.82,
            theta13=8.60,
            theta23=48.6,
            dm21=7.39e-5,
            dm31=2.528e-3,
            delta=0.0,
            nodes=100,
            bins=300,
            Emin=1.,
            Emax=None,
            logscale=False,
            interactions=True,
            energy_vec=None
            ):
        self.mass=mass
        self.channel=channel
        self.process=process
        self.theta12=theta12
        self.theta13=theta13
        self.theta23=theta23
        self.dm21=dm21
        self.dm31=dm31
        self.delta=delta
        self.nodes=nodes
        self.bins=bins
        self.Emin=Emin
        if Emax==None:
            Emax=mass
        self.Emax=Emax
        self.logscale=logscale
        self.energy_vec=energy_vec
        self.interactions=interactions
        # self.fluxCharon=propa.NuFlux(self.channel, self.mass, self.nodes, self.Emin, self.Emax, self.bins,
        #                 self.process, self.logscale, self.interactions,
        #                 self.theta12, self.theta13, self.theta23,
        #                 self.dm21, self.dm31, self.delta)

        self.nurate=dict()

    # Dedicated function for setting nodes and binnings -> need for Charon:
    def SetNodesandBins(self, Emin, Emax, nodes, bins, logscale):
        self.Emin=Emin
        self.Emax=Emax
        self.nodes=nodes
        self.bins=bins
        self.logscale=logscale

    # Function for compute rate with PPPC4 or Charon
    # Computed flux in the format of dictionary: {produced flavour:{"E":energy value array, "dN/dE": spectra value array}}

    def NuRatePPPC4(self):
        return PPPC4_flux(self.mass, self.channel)

    def NuRateCharon(self):
        # just create channel WW then change it later!
        Flux = propa.NuFlux("WW", self.mass, self.nodes, Emin=self.Emin, Emax=self.Emax, bins=self.bins,
                        process=self.process, logscale=self.logscale, interactions=self.interactions,
                        theta_12=self.theta12, theta_13=self.theta13, theta_23=self.theta23,
                        delta_m_12=self.dm21, delta_m_13=self.dm31, delta=self.delta)

        flux_at_Source=dict()
        E=Flux.iniE()

        if self.channel=="nunu":
            nu_tmp = dict()
            for tmp_channel in ["nuenue", "numunumu", "nutaunutau"]:
                print("compute: {}".format(tmp_channel))
                Flux.ch=tmp_channel
                nu_tmp[tmp_channel] = Flux.iniFlux("Halo")


            for flavour in ["nu_mu","nu_e", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]:
                flux_at_Source[flavour]=dict()
                flux_at_Source[flavour]["E"] = E
                flux_at_Source[flavour]["dNdE"] = sum(nu_tmp[ch][flavour] for ch in ["nuenue", "numunumu", "nutaunutau"])/(3.*float(self.mass))
        else:
            Flux.ch = self.channel
            NuCharon=Flux.iniFlux("Halo")
            for flavour in ["nu_mu","nu_e", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]:
                flux_at_Source[flavour]=dict()
                flux_at_Source[flavour]["E"] = E
                flux_at_Source[flavour]["dNdE"] = NuCharon[flavour]/float(self.mass)
        return flux_at_Source

# Averaged Oscillate PPPC4 flux:
    def NuRatePPPC4_AvgOsc(self):
        PPPC4_ini = self.NuRatePPPC4()
        nu_types = ["nu_e","nu_mu", "nu_tau"]
        PPPC4_osc = dict()
        osc_spectra = oscillate_spectra(PPPC4_ini, nu_types, self.theta12, self.theta13, self.theta23, self.delta)

        for flv in nu_types:
            PPPC4_osc[flv] = dict()
            PPPC4_osc[flv]["E"] = PPPC4_ini[flv]["E"]
            PPPC4_osc[flv]["dNdE"] = osc_spectra["dNdE_"+flv+"_osc"]
        return PPPC4_osc

# Averaged Oscillate Charon flux:
    def NuRateCharon_AvgOsc(self):
        Charon_ini = self.NuRateCharon()
        # for neutrino:
        nu_types = ["nu_e","nu_mu", "nu_tau"]
        Charon_osc = dict()
        osc_spectra = oscillate_spectra(Charon_ini, nu_types, self.theta12, self.theta13, self.theta23, self.delta)
        for flv in nu_types:
            Charon_osc[flv] = dict()
            Charon_osc[flv]["E"] = Charon_ini[flv]["E"]
            Charon_osc[flv]["dNdE"] = osc_spectra["dNdE_"+flv+"_osc"]

        #for anti neutrino:
        nu_types = ["nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
        osc_spectra_nubar = oscillate_spectra(Charon_ini, nu_types, self.theta12, self.theta13, self.theta23, self.delta)
        for flv in nu_types:
            Charon_osc[flv] = dict()
            Charon_osc[flv]["E"] = Charon_ini[flv]["E"]
            Charon_osc[flv]["dNdE"] = osc_spectra_nubar["dNdE_"+flv+"_osc"]
        return Charon_osc

# Charon flux at Earth propagate with nuSQuIDS:
    def NuRateCharon_nuSQUIDS(self):
        #Define zenith of the Galactic Centre
        GC_zen = np.deg2rad(-29.00781+90)
        Flux = propa.NuFlux("WW", self.mass, self.nodes, Emin=self.Emin, Emax=self.Emax, bins=self.bins,
                        process=self.process, logscale=self.logscale, interactions=self.interactions,
                        theta_12=self.theta12, theta_13=self.theta13, theta_23=self.theta23,
                        delta_m_12=self.dm21, delta_m_13=self.dm31, delta=self.delta)
        flux_at_Earth = dict()
        if self.channel=="nunu":
            nu_tmp = dict()
            for tmp_channel in ["nuenue", "numunumu", "nutaunutau"]:
                # print("compute: {}".format(tmp_channel))
                Flux.ch = tmp_channel
                nu_tmp[tmp_channel] = Flux.Halo('detector', zenith=GC_zen)


            for flavour in ["nu_e","nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]:
                flux_at_Earth[flavour]=dict()
                flux_at_Earth[flavour]["E"] = nu_tmp[tmp_channel]['Energy']
                flux_at_Earth[flavour]["dNdE"] = sum(nu_tmp[ch][flavour] for ch in ["nuenue", "numunumu", "nutaunutau"])/(3.*float(self.mass))
        else:
            Flux.ch = self.channel
            NuCharon=Flux.Halo('detector', zenith=GC_zen)
            for flavour in ["nu_e","nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]:
                flux_at_Earth[flavour]=dict()
                flux_at_Earth[flavour]["E"] = NuCharon['Energy']
                flux_at_Earth[flavour]["dNdE"] = NuCharon[flavour]/float(self.mass)
        return flux_at_Earth




# Code to test function

# f=NuRate(1000, 'nuenue', 'ann')
# style={"nu_mu":"-", "nu_mu_bar":"-."}
#
# rate=f.NuRateCharon_AverageOscillated()
# rate_source=f.NuRateCharon()
# for flv in ["nu_mu"]:
#     plt.plot(rate[flv]["E"], rate[flv]["dNdE"], linestyle=style[flv])
#     plt.plot(rate_source[flv]["E"], rate_source[flv]["dNdE"], linestyle="-.")
#     plt.semilogy()
# plt.show()
