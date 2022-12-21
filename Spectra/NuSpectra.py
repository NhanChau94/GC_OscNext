"""
author : N. Chau
Collective tools and functions for generating DM interaction (annihilation, decay) rate from GC
part of code borrowed from N. Iovine's analysis on DM search from Galactic Center with OscNext
"""

import os
import numpy as np
import math
import pickle as pkl

#Charon
from charon import propa
curdir=os.path.dirname(os.path.realpath(__file__))


# *********************************************************
# Set of functions for extracting PPPC4 spectra and oscillate it

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
def ExtractPPPC4(mass, channel):
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

    s12 = np.sin(np.deg2rad(t12))
    c12 = np.cos(np.deg2rad(t12))
    s23 = np.sin(np.deg2rad(t23))
    c23 = np.cos(np.deg2rad(t23))
    s13 = np.sin(np.deg2rad(t13))
    c13 = np.cos(np.deg2rad(t13))
    cp  = np.exp(1j*np.deg2rad(d))

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
    oscillated["dNdE_"+nutypes[0]+"_osc"] = np.array(dNdE_nue_osc)
    oscillated["dNdE_"+nutypes[1]+"_osc"] = np.array(dNdE_numu_osc)
    oscillated["dNdE_"+nutypes[2]+"_osc"] = np.array(dNdE_nutau_osc)

    return oscillated




class NuSpectra:
    """docstring for NuSpectra."""

    def __init__(
            self,
            mass=100,
            channel="bb",
            process="ann",
            theta12=33.44,
            theta13=8.57,
            theta23=49.2,
            dm21=7.42e-5,
            dm31=2.515e-3,
            delta=194.,
            nodes=100,
            bins=300,
            Emin=1.,
            Emax=None,
            logscale=False,
            interactions=True,
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
            if process=='ann':
                Emax=mass
            if process=='decay':
                Emax=mass/2.    
        self.Emax=Emax
        self.logscale=logscale
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

    def SpectraPPPC4(self):
        return ExtractPPPC4(self.mass, self.channel)

    def SpectraCharon(self):
        if self.process == "ann":
            factor = 1.0
        elif self.process == "decay":
            factor = 2.0
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
                flux_at_Source[flavour]["E"] = np.array(E)
                flux_at_Source[flavour]["dNdE"] = np.array(sum(nu_tmp[ch][flavour] for ch in ["nuenue", "numunumu", "nutaunutau"])/(3.*float(self.mass)/factor))
        else:
            Flux.ch = self.channel
            NuCharon=Flux.iniFlux("Halo")
            for flavour in ["nu_mu","nu_e", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]:
                flux_at_Source[flavour]=dict()
                flux_at_Source[flavour]["E"] = np.array(E)
                flux_at_Source[flavour]["dNdE"] = np.array(NuCharon[flavour]/(float(self.mass)/factor))
        return flux_at_Source

# Averaged Oscillate PPPC4 flux:
    def SpectraPPPC4_AvgOsc(self):
        PPPC4_ini = self.SpectraPPPC4()
        nu_types = ["nu_e","nu_mu", "nu_tau"]
        PPPC4_osc = dict()
        osc_spectra = oscillate_spectra(PPPC4_ini, nu_types, self.theta12, self.theta13, self.theta23, self.delta)

        for flv in nu_types:
            PPPC4_osc[flv] = dict()
            PPPC4_osc[flv]["E"] = PPPC4_ini[flv]["E"]
            PPPC4_osc[flv]["dNdE"] = osc_spectra["dNdE_"+flv+"_osc"]
        return PPPC4_osc

# Averaged Oscillate Charon flux:
    def SpectraCharon_AvgOsc(self):
        Charon_ini = self.SpectraCharon()
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
    def SpectraCharon_nuSQUIDS(self, GC_zen=np.deg2rad(-29.00781+90)):
        if self.process == "ann":
            factor = 1.0
        elif self.process == "decay":
            factor = 2.0
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
                flux_at_Earth[flavour]["dNdE"] = sum(nu_tmp[ch][flavour] for ch in ["nuenue", "numunumu", "nutaunutau"])/(3.*float(self.mass)/factor)
        else:
            Flux.ch = self.channel
            NuCharon=Flux.Halo('detector', zenith=GC_zen)
            for flavour in ["nu_e","nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]:
                flux_at_Earth[flavour]=dict()
                flux_at_Earth[flavour]["E"] = NuCharon['Energy']
                flux_at_Earth[flavour]["dNdE"] = NuCharon[flavour]/(float(self.mass)/factor)
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
