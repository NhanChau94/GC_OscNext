#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
#METAPROJECT /data/user/niovine/software/combo_py3/build/

import numpy as np
import pickle as pkl
from icecube import dataio, dataclasses, astro
import time, sys
from os.path import isfile
import random
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Utils")
from Utils import *
import glob
import tables
from optparse import OptionParser

def extract_genie(nutype, set, output):

    OscNext_var = dict()
    
    #General infos
    event_id = np.array([])
    time_mjd = np.array([])
    NEvents = np.array([])
    pdg_encoding = np.array([])
    gen_ratio = np.array([])
    PID = np.array([])
    #True variables
    true_E = np.array([])
    true_zenith = np.array([])
    true_azimuth = np.array([])
    #Weighting
    OW = np.array([])
    atm_weight = np.array([])
    nue_spectra = np.array([])
    numu_spectra = np.array([])
    #Cuts
    L4osc_next_bool = np.array([])
    L4muon_classifier_all = np.array([])
    L4noise_classifier = np.array([])
    L5nHit_DOMs = np.array([])
    L7osc_next_bool = np.array([])
    L7muon_classifier_all = np.array([])
    L7muon_classifier_up = np.array([])
    L7reco_vertex_z = np.array([])
    L7reco_vertex_rho36 = np.array([])
    L7_ntop15 = np.array([])
    L7_nouter = np.array([])
    L7reco_time = np.array([])
    L7coincidentMuon_bool = np.array([])
    L7data_quality_cut = np.array([])
    L7containment_cut = np.array([])
    #Reco variables
    reco_zenith = np.array([])
    reco_azimuth = np.array([])
    reco_cascade_energy = np.array([])
    reco_track_energy = np.array([])
    
    NFiles = 0

    # Preparing list of files:
    sample = nutype+set
    filenamelist = glob.glob('/data/ana/LE/oscNext/pass2/genie/level7_v02.00/{0}/oscNext_genie_level7_v02.00_pass2.{0}.*.hdf5'.format(sample))


    for file in filenamelist:
        if not isfile(file):
            print("Missing: {}".format(file))
            continue
        print('Processing file:--------------------')
        print(file)
        try :
            hdf = tables.open_file(file)


            ##Cuts##
            L4osc_next_bool = np.append(L4osc_next_bool, hdf.root.L4_oscNext_bool.cols.value[:])
            L4muon_classifier_all = np.append(L4muon_classifier_all, hdf.root.L4_MuonClassifier_Data_ProbNu.cols.value[:])
            L4noise_classifier = np.append(L4noise_classifier, hdf.root.L4_NoiseClassifier_ProbNu.cols.value[:])
            L5nHit_DOMs = np.append(L5nHit_DOMs, hdf.root.L5_SANTA_DirectPulsesHitMultiplicity.cols.n_hit_doms[:])
            L7osc_next_bool = np.append(L7osc_next_bool, hdf.root.L7_oscNext_bool.cols.value[:])
            L7muon_classifier_all =np.append(L7muon_classifier_all, hdf.root.L7_MuonClassifier_FullSky_ProbNu.cols.value[:])
            L7muon_classifier_up = np.append(L7muon_classifier_up, hdf.root.L7_MuonClassifier_Upgoing_ProbNu.cols.value[:])
            L7reco_vertex_z = np.append(L7reco_vertex_z, hdf.root.L7_reconstructed_vertex_z.cols.value[:])
            L7reco_vertex_rho36 = np.append(L7reco_vertex_rho36, hdf.root.L7_reconstructed_vertex_rho36.cols.value[:])
            L7_ntop15 = np.append(L7_ntop15, hdf.root.L7_CoincidentMuon_Variables.cols.n_top15[:])
            L7_nouter = np.append(L7_nouter, hdf.root.L7_CoincidentMuon_Variables.cols.n_outer[:])
            L7reco_time = np.append(L7reco_time, hdf.root.L7_reconstructed_time.cols.value[:])
            L7coincidentMuon_bool = np.append(L7coincidentMuon_bool, hdf.root.L7_CoincidentMuon_bool.cols.value[:])
            L7data_quality_cut = np.append(L7data_quality_cut, hdf.root.L7_data_quality_cut.cols.value[:])
            L7containment_cut = np.append(L7containment_cut, hdf.root.L7_greco_containment.cols.value[:])


            ##General Informations##
            pdg_encoding = np.append(pdg_encoding, hdf.root.MCInIcePrimary.cols.pdg_encoding[:])
            PID = np.append(PID, hdf.root.L7_PIDClassifier_FullSky_ProbTrack.cols.value[:])
            time_mjd = np.append(time_mjd, hdf.root.I3EventHeader.cols.time_start_mjd[:])
            event_id = np.append(event_id, hdf.root.I3EventHeader.cols.Event[:])

            ##Weights##
            gen_ratio = np.append(gen_ratio,hdf.root.I3MCWeightDict.cols.gen_ratio[:])
            NEvents = np.append(NEvents, hdf.root.I3MCWeightDict.cols.NEvents[:])
            OW = np.append(OW, hdf.root.I3MCWeightDict.cols.OneWeight[:])
            atm_weight = np.append(atm_weight, hdf.root.I3MCWeightDict.cols.weight[:])
            nue_spectra = np.append(nue_spectra, hdf.root.I3MCWeightDict.cols.flux_e[:])
            numu_spectra = np.append(numu_spectra, hdf.root.I3MCWeightDict.cols.flux_mu[:])

            ##True Informations##
            true_E = np.append(true_E, hdf.root.MCInIcePrimary.cols.energy[:]) #Same as PrimaryNeutrinoEnergy in I3MCWeightDict
            true_zenith = np.append(true_zenith, hdf.root.MCInIcePrimary.cols.zenith[:])
            true_azimuth = np.append(true_azimuth, hdf.root.MCInIcePrimary.cols.azimuth[:])

            ##Reconstructed Informations##
            reco_zenith = np.append(reco_zenith, hdf.root.retro_crs_prefit__zenith.cols.median[:])
            reco_azimuth = np.append(reco_azimuth, hdf.root.retro_crs_prefit__azimuth.cols.median[:])
            reco_cascade_energy = np.append(reco_cascade_energy, hdf.root.retro_crs_prefit__cascade_energy.cols.median[:])
            reco_track_energy = np.append(reco_track_energy, hdf.root.retro_crs_prefit__track_energy.cols.median[:])

            NFiles+=1
            hdf.close()

        except:
            print ('WARNING: {}'.format(file))
            continue

    
    if len(reco_zenith)!=0:
        
        #Compute RA, dec and psi
        event_time = np.array([])
        for i in range(len(true_E)):
            if i % int(len(true_E)/10) == 0:
                print ("%1.f%%" %(i*10/int(len(true_E)/10)))
            #generate random time
            stime = time.mktime(time.strptime("8/10/2011/00/00/00", '%m/%d/%Y/%H/%M/%S'))
            etime = time.mktime(time.strptime("7/17/2019/00/00/00", '%m/%d/%Y/%H/%M/%S'))
            eventTime = stime + random.random() * (etime - stime)
            date = time.gmtime(eventTime)
            eventTime_jd = date_to_jd(date[0], date[1], date[2], date[3], date[4], date[5]) #%YYYY,%MM,%DD,%hh,%mm,%ss
            eventTime_mjd = jd_to_mjd(eventTime_jd)
            event_time = np.append(event_time, eventTime_mjd)
        #True
        true_RA, true_dec = astro.dir_to_equa(true_zenith, true_azimuth, event_time)
        true_psi = astro.angular_distance(true_RA, true_dec, np.radians(266.4167), np.radians(-29.0078))
        #Reco
        reco_RA, reco_dec = astro.dir_to_equa(reco_zenith, reco_azimuth, event_time)
        reco_psi = astro.angular_distance(reco_RA, reco_dec, np.radians(266.4167), np.radians(-29.0078))

        
        ##Save in dictionnary##
        OscNext_var[sample] = dict()
        #General infos
        OscNext_var[sample]["Event_ID"] = event_id
        OscNext_var[sample]["MJD_time"] = event_time
        OscNext_var[sample]["NEvents"] = NEvents
        OscNext_var[sample]["PDG_encoding"] = pdg_encoding
        OscNext_var[sample]["gen_ratio"] = gen_ratio
        OscNext_var[sample]["PID"] = PID
        OscNext_var[sample]["NFiles"] = NFiles
        #Weight
        OscNext_var[sample]["OneWeight"] = OW
        OscNext_var[sample]["AtmWeight"] = atm_weight
        OscNext_var[sample]["AtmFlux_nue"] = nue_spectra
        OscNext_var[sample]["AtmFlux_numu"] = numu_spectra
        #True variables
        OscNext_var[sample]["true_Energy"] = true_E
        OscNext_var[sample]["true_Zenith"] = true_zenith
        OscNext_var[sample]["true_Azimuth"] = true_azimuth
        OscNext_var[sample]["true_Dec"] = true_dec
        OscNext_var[sample]["true_RA"] = true_RA
        OscNext_var[sample]["true_psi"] = true_psi
        #Reco variables
        OscNext_var[sample]["reco_CascadeEnergy"] = reco_cascade_energy
        OscNext_var[sample]["reco_TrackEnergy"] = reco_track_energy
        OscNext_var[sample]["reco_TotalEnergy"] = reco_cascade_energy + reco_track_energy
        OscNext_var[sample]["reco_Zenith"] = reco_zenith
        OscNext_var[sample]["reco_Azimuth"] = reco_azimuth
        OscNext_var[sample]["reco_Dec"] = reco_dec
        OscNext_var[sample]["reco_RA"] = reco_RA
        OscNext_var[sample]["reco_psi"] = reco_psi
        #Cuts
        OscNext_var[sample]["L4OscNext_bool"] = L4osc_next_bool
        OscNext_var[sample]["L4muon_classifier_all"] = L4muon_classifier_all
        OscNext_var[sample]["L4noise_classifier"] = L4noise_classifier
        OscNext_var[sample]["L5nHit_DOMs"] = L5nHit_DOMs
        OscNext_var[sample]["L7OscNext_bool"] = L7osc_next_bool
        OscNext_var[sample]["L7muon_classifier_all"] = L7muon_classifier_all
        OscNext_var[sample]["L7muon_classifier_up"] = L7muon_classifier_up
        OscNext_var[sample]["L7reco_vertex_z"] = L7reco_vertex_z
        OscNext_var[sample]["L7reco_vertex_rho36"] = L7reco_vertex_rho36
        OscNext_var[sample]["L7_ntop15"] = L7_ntop15
        OscNext_var[sample]["L7_nouter"] = L7_nouter
        OscNext_var[sample]["L7reco_time"] = L7reco_time 
        OscNext_var[sample]["L7coincident_muon_bool"] = L7coincidentMuon_bool
        OscNext_var[sample]["L7data_quality_cut"] = L7data_quality_cut
        OscNext_var[sample]["L7containment_cut"] = L7containment_cut
        

        pkl.dump(OscNext_var, open(output, "wb"))
        return OscNext_var
    else:
        print ("Empty zenith array - no output")




# hdf = h5py.File('/data/ana/LE/oscNext/pass2/genie/level7_v02.00/121122/oscNext_genie_level7_v02.00_pass2.121122.000011.hdf5', 'r')
# filenamelist = glob.glob('/data/ana/LE/oscNext/pass2/genie/level7_v02.00/121122/oscNext_genie_level7_v02.00_pass2.121122.*.hdf5')

# print(filenamelist)
extract_genie('12','1122', './test.pkl')
#----------------------------------------------------------------------------------------------------------------------
#Define parameters needed
#----------------------------------------------------------------------------------------------------------------------

parser = OptionParser()
# i/o options
parser.add_option("-s", "--set", type = "string", action = "store", default = "1122", metavar  = "<set>", help = "set",)
parser.add_option("-n", "--nutype", type = "string", action = "store", default = "12", metavar  = "<nutype>", help = "nu: 12,14,16",)


(options, args) = parser.parse_args()
set = options.set
nutype = options.nutype
sample = nutype+set
output = '/data/user/tchau/DarkMatter_OscNext/Sample/Simulation/OscNext_Level7_v02.00_{}_pass2_variables_NoCut.pkl'.format(sample)

extract_genie(nutype, set, output)
