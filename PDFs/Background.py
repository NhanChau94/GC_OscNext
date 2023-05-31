"""
author : N. Chau
Background estimation for GC Dark Matter Search
"""
import sys, os
import pickle as pkl
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
import scipy
from scipy import integrate

base_path=os.getenv('GC_DM_BASE')
data_path=os.getenv('GC_DM_DATA')
sys.path.append(f"{base_path}/Utils/")
sys.path.append(f"{base_path}/Spectra/")
sys.path.append(f"{base_path}/DetResponse/")
sys.path.append(f"{base_path}/PDFs/")

from KDE_implementation import *
from Utils import *
from Detector import *
from Signal import *

#######################################
# Background as RA Scramble data

def ScrambleBkg(Bin, sample='burnsample', oversample=10, kde=True, savefile='/data/user/tchau/DarkMatter_OscNext/PDFs/Background/RAScramble_burnsample_FFTkde.pkl',**kwargs):
    # For now only burn sample add a switch to full data later
    if sample=='burnsample':
        dat_dir = f"{data_path}/Sample/Burnsample/"
        input_files = []
        # Take all burnsample:
        print(f'Loading {dat_dir}')
        for year in range(2012, 2021):
            infile = dat_dir + "OscNext_Level7_v02.00_burnsample_{}_pass2_variables_NoCut.pkl".format(year)
            dat = pkl.load(open(infile, 'rb'))
            input_files = np.append(input_files, dat['burnsample'])
    
    array_PID = np.array([])
    array_recopsi = np.array([])
    array_recopsi_original = np.array([])
    array_recoE = np.array([])
    array_recoRA = np.array([])
    array_recoRA_original = np.array([])
    array_recoDec = np.array([])

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
                        (input_file["L7reco_time"]<14500.)&
                        (input_file["reco_TotalEnergy"]>=1.)&
                        (input_file["reco_TotalEnergy"]<=1000.))

        array_PID = np.append(array_PID, input_file["PID"][loc])
        array_recopsi_original = np.append(array_recopsi_original, input_file["reco_psi"][loc])
        array_recoE = np.append(array_recoE, input_file["reco_TotalEnergy"][loc])
        array_recoDec = np.append(array_recoDec, input_file["reco_Dec"][loc])
        array_recoRA_original = np.append(array_recoRA_original, input_file["reco_RA"][loc])

    #Oversample the sample:
    array_PID = np.tile(array_PID, oversample)
    array_recopsi_original = np.tile(array_recopsi_original, oversample)
    array_recoE = np.tile(array_recoE, oversample)
    array_recoDec = np.tile(array_recoDec, oversample)
    array_recoRA_original = np.tile(array_recoRA_original, oversample)

    # Create scramble RA:
    seed_value = np.random.get_state()[1][0]
    np.random.seed(seed_value)
    array_recoRA = np.random.uniform(0,2.*np.pi, size=len(array_recoRA_original))

    # Getting scramble RA psi:
    array_recopsi = np.rad2deg(psi_f(array_recoRA, array_recoDec))

    Psireco_edges = Bin["reco_psi_edges"]
    Ereco_edges = Bin["reco_energy_edges"]
    if kde:
        H = kde_reco(array_recopsi, array_recoE, Bin, **kwargs)   
    else:
        H = np.histogram2d(array_recopsi, array_recoE,
                        bins = (Psireco_edges, Ereco_edges))[0]
    if savefile!="":
        outdict = dict()
        outdict['pdf'] = H/np.sum(H)
        outdict['Ntot'] = len(array_recoRA_original)
        outdict['oversample'] = oversample
        outdict['seed'] = seed_value
        pkl.dump(outdict, open(savefile, "wb"))
    return H

###############################################################################
# Galactic Plane astro flux

def GP_Espectra_pi0(E, scale=1.):
    # Galactic Diffuse flux per neutrino flavour - pi0 model:
    # Icecube best fit give: 21.8 x 10−12 [TeV cm−2 s−1] at 100 TeV as E^2 dN/dE
    # return: GeV^{-1} cm^{-2} s^{-1} per flv
    # norm = 21.8 * 1e-12 *1e3/(pow(100.*1e3,-0.7))
    norm = 4.4*1e-19/pow(100.*1e3,-2.7)
    f_GP =  norm* pow(E, -2.7)
    return f_GP* scale
    

# Functions for decoupled KRA template into energy and spatial part (similar approach as GP paper):
# all-sky energy spectra per flavour: GeV-1 cm^-2 s^-2
def GP_Espectra_KRA(E, template='/data/user/tchau/DarkMatter_OscNext/GP_template/KRA-gamma_maps_energies.tuple.npy', scale=1.):
    KRA = np.load(template, allow_pickle=True, encoding='bytes')
    KRA_energy = np.zeros(KRA[0].shape[1])
    for i in range(KRA[0].shape[1]):
        KRA_energy[i] = np.sum( KRA[0][:,i] )

    # Energy is stored as lower bin edges -> adding final upper edges and compute the bin centr
    lastedge = pow(10., 2*np.log10(KRA[3][-1]) - np.log10(KRA[3][-2]))
    binedges = np.append(KRA[3], lastedge)
    bincenter = [np.sqrt(binedges[i]* binedges[i+1]) for i in range(len(binedges)-1)]    
    flux = KRA_energy*(4*np.pi/KRA[0].shape[0]) *1/3.

    # interpolate to get the flux at desire energy value:
    # should interpolate in log scale -> put 0 value to a very small number
    loc = np.where(flux==0)
    flux[loc] = 9e-30
    y_interp = scipy.interpolate.splrep(np.log10(bincenter), np.log10(flux) )
    f_GP = pow(10., scipy.interpolate.splev(np.log10(E), y_interp, der=0))

    return f_GP* scale


def GP_SpatialPDF(template='/data/user/tchau/Sandbox/GC_OscNext/Fermi-LAT_pi0_map.npy'):

    temp = np.load(template, allow_pickle=True, encoding='bytes')

    if 'pi0' in template:
        # for pi0 template the spatial pdf is energy dependence
        pdf = temp
    elif 'KRA-gamma' in template:
        # KRA-gamma is energy dependence so the energy is integrated here
        pdf = np.zeros(temp[0].shape[0])
        for i in range(temp[0].shape[0]):
            pdf[i] = np.sum( temp[0][i,:] )
      
    # make a normalization so that the all-sky integration is 1
    pdf = pdf/np.sum(pdf* 4*np.pi/len(pdf))
    nside = hp.npix2nside(pdf.size)
    hpix = HEALPix(nside=nside, frame='icrs')  
    c = hpix.healpix_to_skycoord(np.arange(0, pdf.size,1))
    eq = c.icrs

    return pdf, eq.ra.rad, eq.dec.rad
      


#  Compute the reconstruction rate binned in energy and open angle using evt-by-evt reweighted method
def GP_RecoRate(Bin, template='/data/user/tchau/Sandbox/GC_OscNext/Fermi-LAT_pi0_map.npy', set='1122', scale=1, scrambled=False, seed=1000, kde=False, **kwargs):

    # Access the MC:
    MCdict = ExtractMC(['14'+set, '12'+set, '16'+set])

    ##Simulation weight##
    genie_w = MCdict["w"]
    
    ##reco and true variables:
    reco_E = MCdict["E_reco"]
    true_E = MCdict["E_true"]
    reco_psi = MCdict["psi_reco"]
    # true_psi = MCdict["psi_true"]
    true_RA = MCdict["RA_true"]
    reco_RA = MCdict["RA_reco"]

    true_Dec = MCdict["Dec_true"]
    reco_Dec = MCdict["Dec_reco"]
    if scrambled==True:
        np.random.seed(seed)
        reco_RA = np.random.uniform(0,2.*np.pi, size=len(reco_RA))
        reco_psi = np.rad2deg(psi_f(reco_RA, reco_Dec))


    ## Load template
    pdf = GP_SpatialPDF(template=template)[0]
    nside = hp.npix2nside(pdf.size)
    hpix = HEALPix(nside=nside, frame='icrs')
        

    ## Compute weights of each events 
    coords_MC = SkyCoord(true_RA, true_Dec, frame='icrs', unit='rad')

    ## The flux really concentrate on the Galactic plane so it would be better to interpolate of logscale rather than linear scale
    GP_spatial = pow(10., hpix.interpolate_bilinear_skycoord(coords_MC, np.log10(pdf)))
    if 'pi0' in template:
        weights = GP_spatial* GP_Espectra_pi0(true_E, scale=scale) * genie_w *1/2. # Espectra in unit of per flavour so need factor 1/2. for each polarization (nu and nu bar)
    elif 'KRA' in template:
        weights = GP_spatial* GP_Espectra_KRA(true_E, template=template, scale=scale) * genie_w *1/2. # Espectra in unit of per flavour so need factor 1/2. for each polarization (nu and nu bar)

    if kde:
        GP_RecoRate = kde_reco(reco_psi, reco_E, Bin, weights=weights, **kwargs)   
    else:    
        GP_RecoRate = np.histogram2d(reco_psi, reco_E, bins = (Bin['reco_psi_edges'], Bin['reco_energy_edges']), weights=weights)[0]

    return GP_RecoRate


#################################################################
# Extragalactic contribution of DM signal

def ExtraGalactic_decay(mass, channel, decay_time, energy, spectra='PPPC4', nutype='nu_mu' ,zmax=1089., rho_c=0.1198, H0=67.27, Ohm_dm=0.2663, Ohm_m=0.3156, Ohm_lambda=0.6844, zlogscale=False):
    factor = Ohm_dm* rho_c /(4*np.pi* mass* decay_time *H0)
    if zlogscale==True:
        z=np.logspace(1e-8, np.log10(zmax), 10000)
    else:    
        z = np.linspace(0, zmax, 10000)

    nu_spec = NuSpectra(mass=mass, channel=channel, process='decay')
    if spectra=='PPPC4':
        E_spectra = nu_spec.SpectraPPPC4_AvgOsc()
    elif spectra=='Charon':
        E_spectra = nu_spec.SpectraCharon_AvgOsc()

    extra_spectra = np.array([])
    for E in energy:
        energy_source = E*(1+z)
        spectra_source = Interpolate_Spectra(E_spectra, energy_source, mass/2., cutlow=False)
        f_int = 1/np.sqrt(Ohm_lambda + Ohm_m*pow(1+z, 3)) * spectra_source[nutype]
        integration_redshift = integrate.trapz(f_int,z)
        extra_spectra = np.append(extra_spectra, integration_redshift* factor)

    return extra_spectra


    