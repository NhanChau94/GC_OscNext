"""
author : N. Chau
Background estimation from RA scramble data
"""
import sys
import pickle as pkl
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix

sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Utils/")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/PDFs/")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DetResponse/")

from KDE_implementation import *
from Utils import *
from Detector import *

# Background as RA Scramble data
def ScrambleBkg(Bin, sample='burnsample', kde=True, method='FFT', bw="ISJ", oversample=1, mirror=True):
    # For now only burn sample
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
    array_recoRA = np.random.uniform(0,2.*np.pi, size=len(array_recoRA_original))

    # Getting scramble RA psi:
    array_recopsi = np.rad2deg(psi_f(array_recoRA, array_recoDec))

    Psireco_edges = Bin["reco_psi_edges"]
    Ereco_edges = Bin["reco_energy_edges"]
    if kde:
        if method=='FFT':
            psiE_train = np.vstack([array_recopsi, np.log10(array_recoE)]) 
            trueEeval, recoEeval, truePsieval, recoPsieval = Extend_EvalPoints(Bin["true_energy_center"], Bin["reco_energy_center"], np.max(array_recoE), np.max(array_recoE), Bin["true_psi_center"], Bin["reco_psi_center"])  
            g_psi_reco, g_energy_reco = np.meshgrid(recoPsieval, recoEeval, indexing='ij')                      
            psi_eval_reco = g_psi_reco.flatten()
            E_eval_reco = g_energy_reco.flatten()
            psiE_eval = np.vstack([psi_eval_reco, np.log10(E_eval_reco)])


            if mirror:
                print('apply reflection at psi=0')    
                psiE_train=MirroringData(psiE_train, {0:0})
                # extend grid point to contain the mirror data
                recoPsieval_width = recoPsieval[1] - recoPsieval[0]
                while recoPsieval[0]>-180.:
                    recoPsieval=np.append(recoPsieval[0]-recoPsieval_width, recoPsieval)
                
                g_psi_reco, g_energy_reco = np.meshgrid(recoPsieval, recoEeval, indexing='ij')                      
                psi_eval_reco = g_psi_reco.flatten()
                E_eval_reco = g_energy_reco.flatten()
                psiE_eval = np.vstack([psi_eval_reco, np.log10(E_eval_reco)])   

            # if (np.max(psiE_eval[0])<np.max(psiEtrain[0])): print('psi max range not cover data')
            # if (np.min(psiE_eval[0])>np.min(psiEtrain[0])): 
            #     print('psi min range not cover data')
            #     print(np.min(psiE_eval[0]))
            #     print(np.min(psiEtrain[0]))

            # if (np.max(psiE_eval[1])<np.max(psiEtrain[1])): print('E max range not cover data')
            # if (np.min(psiE_eval[1])>np.min(psiEtrain[1])): print('E min range not cover data')

            kde_w = kde_FFT(psiE_train.T, psiE_eval.T
                        ,bandwidth=bw)
            H, v0_edges, v1_edges = np.histogram2d(psi_eval_reco, E_eval_reco,
                                                bins = (Psireco_edges, Ereco_edges),
                                                weights=kde_w)
        elif method=='sklearn':
            psiEtrain = np.vstack([np.log10(array_recopsi), np.log10(array_recoE)])
            g_psi_reco, g_energy_reco = np.meshgrid(Bin['reco_psi_center'], Bin['reco_energy_center'], indexing='ij')                      
            psi_eval_reco = g_psi_reco.flatten()
            E_eval_reco = g_energy_reco.flatten()
            psiE_eval = np.vstack([np.log10(psi_eval_reco), np.log10(E_eval_reco)])

            kde_w = kde_sklearn(psiEtrain.T, psiE_eval.T
                        ,bandwidth=bw)
            H, v0_edges, v1_edges = np.histogram2d(psi_eval_reco, E_eval_reco,
                                                bins = (Psireco_edges, Ereco_edges),
                                                weights=kde_w)
            H = H/g_psi_reco    
    else:
        H, v0_edges, v1_edges = np.histogram2d(array_recopsi, array_recoE,
                                    bins = (Psireco_edges, Ereco_edges))
    
    return H


# Galactic Plane astro flux-> taken from the best fit of icecube for pi0 model

def GC_Espectra(E_GC):
    # Galactic Diffuse flux per neutrino flavour - pi0 model:
    # Icecube best fit give: 21.8 x 10−12 [TeV cm−2 s−1] at 100 TeV as E^2 dN/dE
    # return: GeV^{-1} cm^{-2} s^{-1} per flv
    norm = 21.8 * 1e-12 *1e3/(pow(100.*1e3,-0.7))
    f_GC = norm* pow(E_GC, -2.7)
    return f_GC

def GC_SpatialPDF(psi_centers, template='/data/user/tchau/Sandbox/GC_OscNext/Fermi-LAT_pi0_map.npy'):
    # Galactic Diffuse flux spatial PDF as the function of open angle in degree - default: pi0 model.
    # reading the template: healpix in equatorial coordinate
    # return: the spatial PDF [sr-1] as the function of open angle [degree] which have the integration in all direction equal to 1

    # compute the equatorial coordinates corresponding to the pi0 template

    pi0 = np.load(template)
    nside = hp.npix2nside(pi0.size)
    hpix = HEALPix(nside=nside, frame='icrs')
    c = hpix.healpix_to_skycoord(np.arange(0, pi0.size,1))
    eq = c.icrs
    # corresponding open angle:
    CoordPsi = np.rad2deg(psi_f(eq.ra.rad, eq.dec.rad))


    # bin the [RA, Dec] distribution to the open angle histogram and normalized to get the PDF:
    #  requirement: the integration in all direction should yield 1

    # make edges assuming linear binning
    psi_edges = psi_centers + np.diff(psi_centers)/2.
    psi_edges = np.append(psi_centers[0] - np.diff(psi_centers)[0]/2., psi_edges)
    # bin the GC flux in open angle
    GCflux, edges = np.histogram(CoordPsi, psi_edges, weights=pi0)
    dpsi = np.deg2rad( np.diff(psi_edges) )
    # divide by the solid angle of each bin to obtain the differential flux
    GCflux = GCflux/(2*np.pi* np.sin(np.deg2rad( psi_centers )* dpsi))
    # make the normalization:
    GCflux /=np.sum(pi0)

    return GCflux


def GC_TrueRate(trueE, truePsi,  ESpectra = GC_Espectra, SpatialPDF=GC_SpatialPDF):
    # True Rate of GC diffuse flux per flavour as the function in open angle [deg] and energy [GeV]
    GC_E = ESpectra( trueE )
    GC_psi = SpatialPDF(truePsi)
    GC_TrueRate = np.array(GC_psi[:,None]* GC_E)

    return GC_TrueRate


def GC_RecoRate(Bin, template='/data/user/tchau/Sandbox/GC_OscNext/Fermi-LAT_pi0_map.npy', 
               ESpectra=GC_Espectra, SpatialPDF=GC_SpatialPDF, method='evtbyevt', set='1122', scrambled=False):

    if 'evtbyevt' in method:
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
        true_Dec = MCdict["Dec_true"]
        if scrambled==True:
            true_RA = np.random.uniform(0,2.*np.pi, size=len(true_RA))


        ## Load template
        pi0 = np.load(template)
        nside = hp.npix2nside(pi0.size)
        hpix = HEALPix(nside=nside, frame='icrs')

        # normalize to get the flux pdf: unit[sr^-1] and integration in all direction yields 1
        dOmg = 4*np.pi/(pi0.size)
        pi0 = pi0/(np.sum(pi0) *dOmg)
         

        ## Compute weights of each events 
        coords_MC = SkyCoord(true_RA, true_Dec, frame='icrs', unit='rad')
        GC_spatial = hpix.interpolate_bilinear_skycoord(coords_MC, pi0)
        weights = GC_spatial* GC_Espectra(true_E) * genie_w
        GC_RecoRate, edges1, edges2 = np.histogram2d(reco_psi, reco_E, bins = (Bin['reco_psi_edges'], Bin['reco_energy_edges']), weights=weights)

    elif 'resp' in method:
        # access the precomputed response matrix:
        # The detector response function
        DetResp = pkl.load(open('/data/user/tchau/Sandbox/GC_OscNext/DetResponse/PreComp/Resp_MC{}_logE.pkl'.format(set), 'rb'))
        Bin = DetResp['Bin']
        if scrambled==False:
            Resp = DetResp['Resp']
        else:
            Resp = DetResp['Resp_Scr']  
        GCtrue = GC_TrueRate(Bin['true_energy_center'], Bin['true_psi_center'],  ESpectra = ESpectra, SpatialPDF=SpatialPDF)
        GC_RecoRate = np.zeros((Bin['reco_psi_center'].size, Bin['reco_energy_center'].size))

        # Note: the precomputed rsponse matrix is the dw/dE computed in logscale and then normalized to the total weight.
        # to make the correct weight for logscale bin I transfer it to: dw/(E dE) = dw/(d(logE)) then normalized to the total weight to get the correct integration
        grid = np.meshgrid(Bin['true_psi_center'], Bin['true_energy_center'], 
                    Bin['reco_psi_center'], np.log10(Bin['reco_energy_center']), indexing='ij')


        for nutype in ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]:
            totalIntegration = np.sum(Resp[nutype])
            Resp[nutype] = Resp[nutype]* grid[1]
            Resp[nutype] = Resp[nutype]/np.sum(Resp[nutype])* totalIntegration
            GC_RecoRate += np.tensordot(Resp[nutype], GCtrue, axes=([0,1], [0,1]))

    return GC_RecoRate, Bin
