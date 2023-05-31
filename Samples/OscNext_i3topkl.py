#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/icetray-start
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/metaprojects/icetray/v1.5.1/

from I3Tray import *
from icecube import icetray, dataio, dataclasses, simclasses, recclasses, astro

import pickle as pkl
import numpy as np
import time, sys, os
from os.path import isfile
import random
from optparse import OptionParser
import glob

base_path=os.getenv('GC_DM_BASE')
sys.path.append(f"{base_path}/Utils")
from Utils import *


# output variables : flags to access the variables in I3 files,
Vars = { ##Cuts##
            'L4OscNext_bool':['L4_oscNext_bool'],
            'L4muon_classifier_all':['L4_MuonClassifier_Data_ProbNu'],
            'L4noise_classifier':['L4_NoiseClassifier_ProbNu'],
            'L5nHit_DOMs':['L5_SANTA_DirectPulsesHitMultiplicity', 'n_hit_doms'],

            'L7OscNext_bool':['L7_oscNext_bool'],
            'L7muon_classifier_all':['L7_MuonClassifier_FullSky_ProbNu'],
            'L7muon_classifier_up':['L7_MuonClassifier_Upgoing_ProbNu'],
            'L7reco_vertex_z':['L7_reconstructed_vertex_z'],
            'L7reco_vertex_rho36':['L7_reconstructed_vertex_rho36'],
            'L7_ntop15':['L7_CoincidentMuon_Variables', 'n_top15'],
            'L7_nouter':['L7_CoincidentMuon_Variables', 'n_outer'],
            'L7reco_time':['L7_reconstructed_time'],
            'L7coincident_muon_bool':['L7_CoincidentMuon_bool'],
            'L7data_quality_cut':['L7_data_quality_cut'],
            'L7containment_cut':['L7_greco_containment'],

            ##General Informations##
            'PDG_encoding':['MCInIcePrimary', 'pdg_encoding'],
            'PID':['L7_PIDClassifier_FullSky_ProbTrack'],
            # ['I3EventHeader, time_start_mjd']
            'Event_ID':['I3EventHeader', 'event_id'],

            ##Weights##
            'gen_ratio':['I3MCWeightDict', 'gen_ratio'],
            'NEvents':['I3MCWeightDict', 'NEvents'],
            'OneWeight':['I3MCWeightDict', 'OneWeight'],
            'AtmWeight':['I3MCWeightDict', 'weight'],
            'AtmFlux_nue':['I3MCWeightDict', 'flux_e'],
            'AtmFlux_numu':['I3MCWeightDict', 'flux_mu'],

            ##True Informations##
            'true_Energy':['MCInIcePrimary', 'energy'],
            'true_Zenith':['MCInIcePrimary', 'dir', 'zenith'],
            'true_Azimuth':['MCInIcePrimary', 'dir', 'azimuth'],

            ##Genie's interaction information
            'cc':['I3GENIEResultDict', 'cc'],
            'dis':['I3GENIEResultDict', 'dis'],
            'xsec':['I3GENIEResultDict', 'xsec'],                        
            'difxsec':['I3GENIEResultDict', 'diffxsec'],
            'y':['I3GENIEResultDict', 'y'],

            ##Reconstructed Informations##
            'reco_Zenith':['retro_crs_prefit__zenith', 'median'],
            'reco_Azimuth':['retro_crs_prefit__azimuth', 'median'],
            'reco_CascadeEnergy':['retro_crs_prefit__cascade_energy', 'median'],
            'reco_TrackEnergy':['retro_crs_prefit__track_energy', 'median'],
            

}

## Variables that need computation:
Vars2 = [
    'true_Dec',
    'true_RA',
    'true_psi',
    'reco_TotalEnergy',
    'reco_Dec',
    'reco_RA',
    'reco_psi',    
    'MJD_time',
    'NFiles',
]


def files(args, include_frames=[]):
    """A frame generator that can continue over multiple files"""
    if not isinstance(args, list):
        args = [args]

    for a in args:
        try:
            with dataio.I3File(a) as i3file:
                for frame in i3file:
                    if len(include_frames) and not frame.Stop.id in include_frames:
                        continue
                    yield frame
        except RuntimeError:
            print(a)
            pass



# path = "/data/ana/LE/oscNext/pass2/genie/level7_v02.03/121122/oscNext_genie_level7_v02.03_pass2.121122.000329.i3.zst"

def extract_i3(nutype, set, output, listf=None):
    if listf==None:
        sample = nutype+set
        filenamelist = glob.glob('/data/ana/LE/oscNext/pass2/genie/level7_v02.00/{0}/oscNext_genie_level7_v02.00_pass2.{0}.*.i3.zst'.format(sample))
    else:
        text_file = open(listf, "r")
        filenamelist = text_file.read().split('\n')
    outdict = dict()
    listchecks = np.array([])
    for key in Vars.keys():
        outdict[key] = np.array([])
        listchecks = np.append(listchecks, Vars[key][0])
    print(listchecks)

    Nfiles=0
    for path in filenamelist:
        if not isfile(path):
            print("Missing: {}".format(path))
            continue
        print('Processing file:--------------------')
        Nfiles+=1
        print(path)

        for frame in files(path):
            # total+=1
            fr = frame

            # Only take frame whcih has all the required fields
            checks = True
            for k in listchecks:
                checks *= fr.Has(k)
            if checks:
                for outflag in Vars.keys():
                    # print(10*'-')
                    I3value = fr[Vars[outflag][0]]
                    # print(Vars[outflag][0])
                    # print(value)
                    # Go in to subfields to get values if needed
                    for i in range(1, len(Vars[outflag])):
                        # print('---->' + Vars[outflag][i])
                        if hasattr(I3value, "__getitem__"):
                            I3value = I3value[Vars[outflag][i]]
                        else:
                            # print(dir(value))
                            I3value = getattr(I3value, Vars[outflag][i])

            
                    if isinstance(I3value, (float, int, str, list, dict, tuple, bool)):        
                        outdict[outflag] = np.append(outdict[outflag], I3value)
                    else:
                        outdict[outflag] = np.append(outdict[outflag], I3value.value)


    # output variables that need computation:
    reco_zenith=outdict['reco_Zenith']
    true_zenith=outdict['true_Zenith']
    true_azimuth=outdict['true_Azimuth']
    reco_azimuth=outdict['reco_Azimuth']

    if len(reco_zenith)!=0:    
        #Compute RA, dec and psi
        event_time = np.array([])
        for i in range(len(true_zenith)):
            if i % int(len(true_zenith)/10) == 0:
                print ("%1.f%%" %(i*10/int(len(true_zenith)/10)))
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
        true_psi = astro.angular_distance(true_RA, true_dec, np.radians(GCRA), np.radians(GCDec))
        #Reco
        reco_RA, reco_dec = astro.dir_to_equa(reco_zenith, reco_azimuth, event_time)
        reco_psi = astro.angular_distance(reco_RA, reco_dec, np.radians(GCRA), np.radians(GCDec))

    outdict['MJD_time'] = event_time
    outdict['true_Dec'] = true_dec
    outdict['true_RA'] = true_RA
    outdict['true_psi'] = true_psi
    outdict['reco_Dec'] = reco_dec
    outdict['reco_RA'] = reco_RA
    outdict['reco_psi'] = reco_psi
    outdict['reco_TotalEnergy'] = outdict['reco_TrackEnergy'] + outdict['reco_CascadeEnergy']
    outdict['NFiles'] = Nfiles

    # true_Dec
    # true_RA
    # true_psi
    # reco_TotalEnergy
    # reco_Dec
    # reco_RA
    # reco_psi
    # MJD_time
    # NFiles



    pkl.dump(outdict, open(output, "wb"))


#----------------------------------------------------------------------------------------------------------------------
#Define parameters parsing
#----------------------------------------------------------------------------------------------------------------------

parser = OptionParser()
# i/o options
parser.add_option("-s", "--set", type = "string", action = "store", default = "1122", metavar  = "<set>", help = "set",)
parser.add_option("-n", "--nutype", type = "string", action = "store", default = "12", metavar  = "<nutype>", help = "nu: 12,14,16",)
parser.add_option("--listf", type = "string", action = "store", default = None, metavar  = "<listf>", help = "input list of file",)


(options, args) = parser.parse_args()
set = options.set
nutype = options.nutype
listf = options.listf

sample = nutype+set

if listf==None:
    output = '/data/user/tchau/DarkMatter_OscNext/Sample/Simulation/OscNext_Level7_v02.00_{}_pass2_variables_NoCut_fromi3.pkl'.format(sample)
else:
    name = os.path.basename(listf)
    output = '/data/user/tchau/DarkMatter_OscNext/Sample/Simulation/{}.pkl'.format(name)

extract_i3(nutype, set, output, listf=listf)
