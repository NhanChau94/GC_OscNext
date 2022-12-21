"""
author : N. Chau
Background estimation from RA scramble data
"""
import sys
import math
import pickle as pkl
import numpy as np
import scipy

sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Utils/")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/PDFs/")
from KDE_implementation import *
from Utils import *


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

    #Oversample the burnsample:
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