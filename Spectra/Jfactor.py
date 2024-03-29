"""
author : N. Chau
Collective tools and functions for computing Jfactor used for DM search from Galactic Center
note: part of code comes from N. Iovine's analysis on DM search from Galactic Center with OscNext
"""
import os, sys
import numpy as np
from charon import profile

curdir=os.path.dirname(os.path.realpath(__file__))


def extract_values(filename, pos1, pos2):
    value_file = open(filename).readlines()
    val1 = np.array([])
    val2 = np.array([])
    for l in value_file:
        line = l.split()
        # print (line)
        try:
            float(line[pos1])
            float(line[pos2])
        except ValueError:
            continue
        val1 = np.append(val1, np.float64(line[pos1].strip()))
        val2 = np.append(val2, np.float64(line[pos2].strip()))
    return val1, val2



class Jf(object):
    """Class for extracting density and Jfactor profiles.

    Extracting density and Jfactor either from Clumpy pre-computed files in the
    "HaloModel" folder or from methods implemented by Charon.
    Currently supported density profiles in Clumpy: NFW and Burkert.
    Supported density profiles in Charon: NFW, Burkert, Einasto, Zhao, and Isothermal.

    Parameters:
    profile (str): Density profile to use. Must be one of "NFW" or "Burkert" for Clumpy,
                   or one of "NFW", "Burkert", "Einasto", "Zhao", or "Isothermal" for Charon.
                   Default is "NFW".
    process (str): Annihilation or decay process. Must be either "ann" or "decay".
                   Default is "ann".

    """

    def __init__(
        self,
        profile="NFW",
        process='ann'
        ):
        self.profile = profile
        self.process = process

    # profile extraction: dictionary of {r:[values of r in kpc], rho:[values of rho in GeV/cm^3]}

    # Clumpy: contain subhalo
    def profile_Clumpy(self, errors=None):
        profile = self.profile
        if errors is None:
            clumpyfile = f"{curdir}/HaloModels/ClumpyOutput/{profile}/Density_rhor_GeV_cm3_{profile}_NestiSalucci.output"
        else:
            clumpyfile = f"{curdir}/HaloModels/ClumpyOutput/{profile}/Density_rhor_GeV_cm3_{profile}_NestiSalucci_{errors}.output"

        r_values, rho_values = extract_values(clumpyfile, 0, 3)
        rho_dict = dict()
        rho_dict["rho"] = rho_values
        rho_dict["r"] = r_values
        return rho_dict

    # Charon's profile module as function of r (array of values in kpc)
    def profile_Charon(self, r, prof=None ,**kwargs):
        if prof==None:
            prof=self.profile
        # Create charon object
        if prof=="NFW":
            density = profile.NFW(r, **kwargs)
        elif prof=="Burkert":
            density = profile.Burkert(r, **kwargs)
        elif prof=="Einasto":
            density = profile.Einasto(r, **kwargs)   
        elif prof=="Zhao":
            density = profile.Zhao(r, **kwargs)
        elif prof=="Isothermal":
            density = profile.Isothermal(r, **kwargs)                       

        rho_dict = dict()
        rho_dict["rho"] = density
        rho_dict["r"] = r
        return rho_dict

    # PreComputed Jfactor (or D-factor) taken from Clumpy in unit:  
    # [GeV^{2} cm^{-5} sr^{-1}] for annihilation, [GeV cm^{-2} sr^{-1}] for decay

    def Jfactor_Clumpy(self, errors=None):
        profile = self.profile
        if self.process=='ann':
            nametag='Jfactor_dJdOmega_GeV2_cm5_sr'
            col=4
        elif self.process=='decay':
            nametag='Dfactor_dDdOmega_GeV_cm2_sr'
            col=3
        if errors is None:
            clumpyfile = f"{curdir}/HaloModels/ClumpyOutput/{profile}/{nametag}_{profile}_NestiSalucci.output"
        else:
            clumpyfile = f"{curdir}/HaloModels/ClumpyOutput/{profile}/{nametag}_{profile}_NestiSalucci_{errors}.output"

        psi_values, Jpsi_values = extract_values(clumpyfile, 0, col)

        JPsi_dict = dict()
        JPsi_dict["J"] = Jpsi_values
        JPsi_dict["psi"] = psi_values
        return JPsi_dict

    # Compute Jfactor with Charon as function of psi in degree
    # R  = 100.  #maximum of the line of sight in kpc
    # d  = 8     #distance from Earth to the Galactic center in kpc
    def Jfactor_Charon(self, psi, prof=None, R=100, d=8, process=None, **kwargs ):
        if process==None:
            process = self.process
        if prof==None:
            prof=self.profile
        params = {}
        if len(kwargs.items()) != 0.0:
            for key, value in kwargs.items():
                params[key] = value    
        # Charon object
        prof=eval('profile.{}'.format(prof))
        J=profile.J(prof, R, d, process, **params)
        psi_inrad = np.deg2rad(psi) # Charon takes psi in rad
        Jpsi_values = [J.Jtheta(j) for j in psi_inrad]

        JPsi_dict = dict()
        JPsi_dict["J"] = Jpsi_values
        JPsi_dict["psi"] = psi
        return JPsi_dict
