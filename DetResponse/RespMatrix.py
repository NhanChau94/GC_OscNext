#!/usr/bin/env /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python
import pickle as pkl
import sys
import pickle as pkl
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/DetResponse/")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/Utils/")
sys.path.append("/data/user/tchau/Sandbox/GC_OscNext/PDFs/")
from Detector import *
from Signal import *


def Extend_grid(E_true, E_reco, maxEtrue, maxEreco, psi_true, psi_reco):
    logEtrue_width = np.log10(E_true[1]) - np.log10(E_true[0])
    while E_true[-1]<maxEtrue:
        E_true = np.append(E_true, pow(10, np.log10(E_true[-1])+logEtrue_width))
    while E_true[0]>0.95:    
        E_true = np.append(pow(10, np.log10(E_true[0])-logEtrue_width), E_true)
        

    logEreco_width = np.log10(E_reco[1]) - np.log10(E_reco[0])
    while E_reco[-1]<maxEreco:
        E_reco = np.append(E_reco, pow(10, np.log10(E_reco[-1])+logEreco_width))
    while E_reco[0]>1.:    
        E_reco = np.append(pow(10, np.log10(E_reco[0])-logEreco_width), E_reco)
    

    psitrue_width = psi_true[1] - psi_true[0]
    psi_true = np.append(psi_true[0]-psitrue_width, psi_true)
    psi_true = np.append(psi_true, psi_true[-1]+psitrue_width)

    psireco_width = psi_reco[1] - psi_reco[0]
    psi_reco = np.append(psi_reco[0]-psireco_width, psi_reco)
    psi_reco = np.append(psi_reco, psi_reco[-1]+psireco_width)

    return E_true, E_reco, psi_true, psi_reco


def RespMatrix_FFTkde(MCcut, Bin, mirror=True, Scramble=False, weight_cut=True):

    grid = np.meshgrid(Bin['true_psi_center'], Bin['true_energy_center'], 
            Bin['reco_psi_center'], np.log10(Bin['reco_energy_center']), indexing='ij')

    trueEeval, recoEeval, truePsieval, recoPsieval = Extend_grid(Bin["true_energy_center"], Bin["reco_energy_center"], 3100, 1000, Bin["true_psi_center"], Bin["reco_psi_center"])  
    # print('Etrue: {}'.format(trueEeval))
    # print('Ereco: {}'.format(recoEeval))
    # print('Psitrue: {}'.format(truePsieval))
    # print('Psireco: {}'.format(recoPsieval))


    g_psi_true, g_energy_true, g_psi_reco, g_energy_reco = np.meshgrid(truePsieval, trueEeval,
                                                            recoPsieval, recoEeval, indexing='ij')                      
    psi_eval_true = g_psi_true.flatten()
    E_eval_true = g_energy_true.flatten()
    psi_eval_reco = g_psi_reco.flatten()
    E_eval_reco = g_energy_reco.flatten()

    psiE_eval = np.vstack([psi_eval_true, np.log10(E_eval_true), 
            psi_eval_reco, np.log10(E_eval_reco)])



    # Separate MC by each channel nutype->PID:
    nu_types = ["nu_e", "nu_mu", "nu_tau", "nu_e_bar", "nu_mu_bar", "nu_tau_bar"]
    # nu_types = ["nu_mu"]

    pdg_encoding = {"nu_e":12, "nu_mu":14, "nu_tau":16, "nu_e_bar":-12, "nu_mu_bar":-14, "nu_tau_bar":-16}
    # PID = [[0.,0.5],[0.5, 0.85],[0.85, 1]]
    PID = [[0.,1.]]
    Resp = dict()
    pidbin = 0
    for pid in PID:
        print("Computing {} PID bin".format(pid))
        # Resp[pidbin] = dict()
        # pidbin += 1
        for nu_type in nu_types:
            print("----{}".format(nu_type))

            psireco=MCcut["psi_reco"]
            if Scramble:
                print("Scrambled Response matrix")
                RAreco = MCcut["RA_reco"]
                Decreco = MCcut["Dec_reco"]
                # Create scramble RA:
                RAreco_Scr = np.random.uniform(0,2.*np.pi, size=len(RAreco))
                # Get correct psi from scramble RA and original DEC
                psireco = np.rad2deg(psi_f(RAreco_Scr, Decreco))

            loc = np.where(  (MCcut["nutype"]==pdg_encoding[nu_type]) & (MCcut["PID"]>=pid[0])
                            & (MCcut["PID"]<pid[1]) 
                            & (MCcut["E_reco"] < 1000)
                            & (MCcut["E_reco"] > np.min(Bin["reco_energy_center"]))
                            & (MCcut["E_true"] < 3100)
                            & (MCcut["E_true"] > 0.95)
                            #np.min(Bin["true_energy_center"]))
                            # & (MCcut["psi_true"] < np.max(Bin["true_psi_center"]))
                            # & (MCcut["psi_true"] > np.min(Bin["true_psi_center"]))
                            # & (psireco < np.max(Bin["reco_psi_center"]))
                            # & (psireco > np.min(Bin["reco_psi_center"]))
                            )
        
            #Extract MC events: 
            #NOTE: input psi in deg!
            psitrue = MCcut["psi_true"][loc]
            Etrue = MCcut["E_true"][loc]
            psireco = psireco[loc]
            Ereco = MCcut["E_reco"][loc]
            w = MCcut["w"][loc]        

            if weight_cut:
                w_lim = define_weightcut(w, 150) #Previously 200
                print ("##Applying cut on weight##")
                print ("Weight lim:", w_lim)
                w_loc = np.where(w<=w_lim)
                #Renormalise weight for total weight to be unchanged
                w = w[w_loc] * (np.sum(w)/np.sum(w[w_loc]))
                psitrue = psitrue[w_loc]
                Etrue = Etrue[w_loc]
                psireco = psireco[w_loc]
                Ereco = Ereco[w_loc]



            print("Preparing train grid")    
            # psiE_train = np.vstack([np.log(psitrue), Etrue, np.log(psireco), np.log10(Ereco)])
            
            psiE_train = np.vstack([psitrue, np.log10(Etrue), psireco, np.log10(Ereco)])

            if mirror:
                psiE_train=MirroringData(psiE_train, {0:0, 2:0})
                w=np.concatenate((w,w))
                print("Correct bias at boundary psi=0 using mirror data (reflection)")
                # extend grid point to contain the mirror data
                recoPsieval_width = recoPsieval[1] - recoPsieval[0]
                while recoPsieval[0]>-180.:
                    recoPsieval=np.append(recoPsieval[0]-recoPsieval_width, recoPsieval)
                
                truePsieval_width = truePsieval[1] - truePsieval[0]
                while truePsieval[0]>-180.:
                    truePsieval=np.append(truePsieval[0]-truePsieval_width, truePsieval)
                
                g_psi_true, g_energy_true, g_psi_reco, g_energy_reco = np.meshgrid(truePsieval, trueEeval,
                                                            recoPsieval, recoEeval, indexing='ij')                      
                psi_eval_true = g_psi_true.flatten()
                E_eval_true = g_energy_true.flatten()
                psi_eval_reco = g_psi_reco.flatten()
                E_eval_reco = g_energy_reco.flatten()

                psiE_eval = np.vstack([psi_eval_true, np.log10(E_eval_true), 
                        psi_eval_reco, np.log10(E_eval_reco)]) 


            print("Evaluating KDE.....")    

            kde_w = kde_FFT(psiE_train.T, psiE_eval.T, bandwidth='ISJ', weights=w)
            #Needs to be divided by evaluation angle
            # kde_weight = kde_w.reshape(psi_eval_true.shape)
            # kde_weight = kde_w/(psi_eval_true * psi_eval_reco)
            kde_weight = kde_w
                                    
            # Fill into histogram:
            Psitrue_edges = Bin["true_psi_edges"]
            Etrue_edges = Bin["true_energy_edges"]
            Psireco_edges = Bin["reco_psi_edges"]
            Ereco_edges = Bin["reco_energy_edges"]

            H, edges = np.histogramdd((psi_eval_true, E_eval_true, psi_eval_reco, E_eval_reco),
                                bins = (Psitrue_edges, Etrue_edges, Psireco_edges, Ereco_edges),
                                weights=kde_weight)

            # N = np.histogramdd((psi_eval_true, E_eval_true, psi_eval_reco, E_eval_reco),
            #                     bins = (Psitrue_edges, Etrue_edges, Psireco_edges, Ereco_edges))
            # if np.min(N)==0:
            #     return 'There is an empty bin!!'                    
            # H = H/N[0]
            H = H/grid[1]
            if mirror:
                norm = np.sum(w)/(2*np.sum(H))
            else:
                norm = np.sum(w)/np.sum(H) 
            Resp[nu_type] = H*norm
    return Resp


from optparse import OptionParser
parser = OptionParser()
# i/o options
parser.add_option("-s", "--set", type = "string", action = "store", default = "0000", metavar  = "<set>", help = "MC set",)

(options, args) = parser.parse_args()

set = options.set

MCcut = ExtractMC(['14'+set, '12'+set, '16'+set])

Bin = Std_Binning(3100, N_Etrue=300, N_psitrue=100)
Etrue_edges = pow(10., np.linspace(np.log10(0.95), np.log10(3100), 301))
Etrue_center = np.array([np.sqrt(Etrue_edges[i]*Etrue_edges[i+1]) for i in range(len(Etrue_edges) - 1)])
Bin['true_energy_center']=Etrue_center
Bin['true_energy_edges']=Etrue_edges

Resp = RespMatrix_FFTkde(MCcut, Bin)
Resp_Scr = RespMatrix_FFTkde(MCcut, Bin, Scramble=True)
outdict = dict()
outdict['Bin'] = Bin
outdict['Resp'] = Resp
outdict['Resp_Scr'] = Resp_Scr
pkl.dump(outdict, open("/data/user/tchau/Sandbox/GC_OscNext/DetResponse/PreComp/Resp_MC{}_logE_v2.pkl".format(set), "wb"))