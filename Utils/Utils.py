import numpy as np
import pickle as pkl

def ExtractMC():
    # Extract Simulation file:
    Sim12 = pkl.load(open("../Sample/Simulation/OscNext_Level7_v02.00_120000_pass2_variables_NoCut.pkl", "rb"))
    Sim14 = pkl.load(open("../Sample/Simulation/OscNext_Level7_v02.00_140000_pass2_variables_NoCut.pkl", "rb"))
    Sim16 = pkl.load(open("../Sample/Simulation/OscNext_Level7_v02.00_160000_pass2_variables_NoCut.pkl", "rb"))
    # Sim = [Sim12['120000'], Sim14['140000'], Sim16['160000']]

    Cut = [ApplyCut(Sim12['120000']), ApplyCut(Sim14['140000']), ApplyCut(Sim16['160000'])]
    MCcut = dict()
    for key in Cut[0].keys():
        MCcut[key] = np.array([])
        for c in Cut:
            MCcut[key] = np.concatenate((MCcut[key], c[key]), axis=None) 

    return MCcut