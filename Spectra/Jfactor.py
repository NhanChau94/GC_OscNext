"""
author : N. Chau
Collective tools and functions for computing Jfactor used for DM search from Galactic Center
note: part of code comes from N. Iovine's analysis on DM search from Galactic Center with OscNext
"""
import os, time, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import pickle as pkl

#Charon
# sys.path.append("/data/user/niovine/software/charon/charon")
# from charon import profile
# import charon.physicsconstants as PC
# pc = PC.PhysicsConstants()
curdir=os.path.dirname(os.path.realpath(__file__))


def extract_values(filename, pos1, pos2):
    value_file = open(filename).readlines()
    val1 = np.array([])
    val2 = np.array([])
    for l in value_file:
        line = l.split("   ")
        #print (line)
        if "#" in line[0]:
            continue
        val1 = np.append(val1, np.float64(line[pos1].strip()))
        val2 = np.append(val2, np.float64(line[pos2].strip()))
    return val1, val2



class Jf(object):
    """docstring fo Jfactor
     Extracting density and Jfactor either from Clumpy pre-computed files folder HaloModel
     or from method implemented by Charon
     2 profiles are considered: NFW and Burkert
    """

    def __init__(
        self,
        process="ann",
        profile="NFW",
        ):
        self.profile = profile
        self.process = process
# profile extraction: dictionary of {r:[values of r in kpc], rho:[values of rho in GeV/cm^3]}

    # Clumpy: contain subhalo
    def profile_Clumpy(self):
        profile = self.profile
        clumpyfile = f"{curdir}/HaloModels/ClumpyOutput/{profile}/Density_rhor_GeV_cm3_{profile}_NestiSalucci.output"
        r_values, rho_values = extract_values(clumpyfile, 0, 3)
        rho_dict = dict()
        rho_dict["rho"] = rho_values
        rho_dict["r"] = r_values
        return rho_dict

    # Charon profile as function of r (array of values in kpc): parametrized
    def profile_Charon(self, r, default=True, rs=24.42, rhos=0.184, gamma=1):
        # Create charon object
        if self.profile=="NFW":
            if default==True:
                density = profile.NFW(r)
            else:
                density = profile.NFW(r, rs, rhos, gamma)
        elif self.profile=="Burkert":
            if default==True:
                density = profile.Burkert(r)
            else:
                density = profile.Burkert(r, rs, rhos)

        rho_dict = dict()
        rho_dict["rho"] = density
        rho_dict["r"] = r
        return rho_dict

    # Compute Jfactor taken from Clumpy
    def Jfactor_Clumpy(self):
        profile = self.profile
        clumpyfile = f"{curdir}/HaloModels/ClumpyOutput/{profile}/Jfactor_dJdOmega_GeV2_cm5_sr_{profile}_NestiSalucci.output"
        psi_values, Jpsi_values = extract_values(clumpyfile, 0, 4)

        JPsi_dict = dict()
        JPsi_dict["J"] = Jpsi_values
        JPsi_dict["psi"] = psi_values
        return JPsi_dict

    # Compute Jfactor with Charon as function of psi in degree
    # R  = 100.  #maximum of the line of sight in kpc
    # d  = 8     #distance from Earth to the Galactic center in kpc
    def Jfactor_Charon(self, psi, rs=24.42, rhos=0.184, gamma=1, R=100, d=8):
        process = self.process
        if self.profile=='NFW':
            pro=profile.NFW
            J = profile.J(pro,R,d,process, rs=rs, rhos=rhos, gamma=gamma)
        elif self.profile=='Burkert':
            pro=profile.Burkert
            J = profile.J(pro,R,d,process, rs=rs, rhos=rhos)
        else:
            sys.exit("ERROR: the profile considered here are only NFW and Burkert")

        psi_inrad = np.deg2rad(psi) # Charon takes psi in rad
        Jpsi_values = [J.Jtheta(j) for j in psi_inrad]

        JPsi_dict = dict()
        JPsi_dict["J"] = Jpsi_values
        JPsi_dict["psi"] = psi
        return JPsi_dict
