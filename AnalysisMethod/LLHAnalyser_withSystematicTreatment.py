from iminuit import Minuit
import numpy as np
import scipy.special as sps

from sensitivity_utils import BinomialError, inv_BinomialError



class Profile_Analyser:

    def __init__(self):
        
        self.moreOutput = False
        
        #Define options of profile analyser
        self.LLHtype = None
        self.samplingMethod = "default"

        #Background and signal PDFs
        self.nbins = 0.
        self.NEvents = 0.
        self.PDF = dict()
        self.quadPDF = dict()
        self.ntot = dict()
        
        #Defines if ready to go to likelihood computation
        self.ready = False 
        
        #Sampling
        self.observation = None
        self.systematics = False

        #Best fit and TS distribution
        self.AllowNegativeSignal = False
        self.computedBestFit = False
        self.bestFit = None
        self.TS = None
        

        
    #Define type of LLH to use
    def setLLHtype(self,type):
        availableTypes = ["Poisson", "PoissonSignalSubtraction", "Effective"]
        if type not in availableTypes:
            raise ValueError("LLH type not implemented yet. Choose amongst: "+str(availableTypes))
        else:
            self.LLHtype = type
            
    def setNEvents(self,nEvents):
        self.NEvents = nEvents
        print("NEvent:",self.NEvents)
        
    def allowNegativeSignal(self):
        self.AllowNegativeSignal = True
        print("Allowing for negative signal") 
        
    def includeSystematics(self):
        self.systematics = True
        print("Including systematics")
            
    def saveMoreOutput(self):
        self.moreOutput = True
        
        
    #------------------------------------------------------------------------------------------
    #Load required PDFs
    #Uncertainty PDFs used for Effective likelihood 
    #Requirements: Loaded PDFs must be expressed in terms of number of events
    #------------------------------------------------------------------------------------------
    
    def loadPDF(self,pdf,name):
        possiblePDFs = ["dataPDF", "backgroundPDF", "signalPDF", "scrambledsignalPDF"]
        
        if name not in possiblePDFs:
            print ("PDF name must be in {}".format(str(possiblePDFs)))
        
        self.ntot[name] = np.sum(pdf)
        self.PDF[name] = pdf.flatten()/self.ntot[name] #Normalise PDF
        
        print("Total number of events in {}:".format(name),self.ntot[name])
        print("Sum of normalised in {}:".format(name),np.sum(self.PDF[name]))
        
        #Define number of bins in BG
        if name == "backgroundPDF":
            self.nbins = len(pdf.flatten())
        
        #Check if PDF binning matches the BG binning
        if len(pdf.flatten()) == self.nbins:
            self.ready = True
        else:
            raise ValueError("Shape of {} does not match background pdf! Was the background pdf initialised first?".format(name))     
    
    def loadUncertaintyPDF(self,pdf,name):
        possiblePDFs = ["backgroundPDF_uncert2", "signalPDF_uncert2"]
        associated_ntot = {"backgroundPDF_uncert2":"backgroundPDF", "signalPDF_uncert2":"signalPDF"}
        
        if name not in possiblePDFs:
            print ("PDF name must be in {}".format(str(possiblePDFs)))
        
        self.quadPDF[name] = pdf.flatten()/np.power(self.ntot[associated_ntot[name]],2.)
        
        print("Total number of events in {}:".format(name),np.sum(pdf.flatten()))
        print("Squared number of events in {}:".format(associated_ntot[name]),np.power(self.ntot[associated_ntot[name]],2.))
        print("Sum of normalised in {}:".format(name),np.sum(self.quadPDF[name]))
        
        #Check if PDF binning matches BG binning
        if len(pdf.flatten()) == self.nbins:
            self.ready = True
        else:
            raise ValueError("Shape of {} does not match background pdf! Was the background pdf initialised first?".format(name))     
    def loadSystematicPDF(self,pdf,name):
        possiblePDFs = ["backgroundPDF_syst", "signalPDF_syst"]
        associated_ntot = {"backgroundPDF_syst":"backgroundPDF", "signalPDF_syst":"signalPDF"}
        
        if name not in possiblePDFs:
            print ("PDF name must be in {}".format(str(possiblePDFs)))
        
        self.ntot[name] = float(np.sum(pdf))
        self.PDF[name] = pdf.flatten()/self.ntot[name]
        
        print("Total number of events in {}:".format(name),np.sum(pdf.flatten()))
        print("Sum of normalised in {}:".format(name),np.sum(self.PDF[name]))
        
        #Check if PDF binning matches the BG binning
        if len(pdf.flatten()) == self.nbins:
            self.ready = True
        else:
            raise ValueError("Shape of {} does not match background pdf! Was the background pdf initialised first?".format(name))  
    
    
    #------------------------------------------------------------------------------------------
    #Likelihood Method
    #Possible likelihood options: Poisson, PoissonSignalSubtraction, Effective
    #------------------------------------------------------------------------------------------
            
    def sampleObservation(self,xi=None,n=None):
        
        #Sample pseudo-experiments from either
        #  - the signal fraction: xi
        #  - the number of signal events: n

        if not self.ready:
            raise ValueError("Not all pdfs are correctly loaded!")
            
        if self.NEvents == 0.:
            raise ValueError('The number of events is not set!')
        
        #If initial information in term of number of events
        #Need to be converted to signal fraction
        if n!=None:
            if (n != 0.):
                n_poisson = np.random.poisson(n)
                xi = n_poisson/self.NEvents
            else:
                xi = 0.
        
        #Define which set to use to generate pseudo-experiments
        if self.systematics==False:
            gen_background = self.PDF["backgroundPDF"]
            gen_signal = self.PDF["signalPDF"]
        else:
            gen_background = self.PDF["backgroundPDF_syst"]
            gen_signal = self.PDF["signalPDF_syst"]
        
        
        observationPDF = self.NEvents * ((1-xi)*gen_background + xi*gen_signal)

        #Randomise observation PDF
        self.observation = np.zeros(np.shape(observationPDF))
        for i in range(len(observationPDF)):
            self.observation[i] = np.random.poisson(observationPDF[i])

        self.computedBestFit = False

        
        
    def evaluateLLH(self,xi):

        #Poisson
        if self.LLHtype == "Poisson":
            modelPDF =  self.NEvents * ((1-xi)*self.PDF["backgroundPDF"] + xi*self.PDF["signalPDF"]) 
            if np.isnan(modelPDF).any():
                print('nan in model array with xi=',xi, self.computedBestFit)
            bins_to_use = (modelPDF>0.)
            
            ##Poisson likelihood##
            #L = (lambda**k / k!) * exp(-lambda)
            #logL = sum(k * log(lambda) - lambda)
            #Evaluate in term of signal fraction
            k = self.observation[bins_to_use]
            lambd = modelPDF[bins_to_use]
            values = k * np.log(lambd) - lambd
            
        
        #Poisson signal contamination
        elif self.LLHtype == "PoissonSignalSubtraction":
            modelPDF =  self.NEvents * ((self.PDF["backgroundPDF"]-xi*self.PDF["scrambledsignalPDF"]) + xi*self.PDF["signalPDF"] )
            if np.isnan(modelPDF).any():
                print('nan in model array with xi=',xi, self.computedBestFit)
            
            bins_to_use = (modelPDF>0.)

            ##Poisson likelihood##
            #L = (lambda**k / k!) * exp(-lambda)
            #logL = sum(k * log(lambda) - lambda)
            #Evaluate in term of signal fraction
            k = self.observation[bins_to_use]
            lambd = modelPDF[bins_to_use]
            values = sk * np.log(lambd) - lambd
        
        
        #Effective
        elif self.LLHtype == "Effective":
            modelPDF =  self.NEvents * ((1-xi)*self.PDF["backgroundPDF"] +  xi*self.PDF["signalPDF"]) 
            modelPDF_uncert2 = np.power(self.NEvents,2.) * (np.power((1-xi),2.)*self.quadPDF["backgroundPDF_uncert2"] +
                                                               np.power(xi,2.)*self.quadPDF["signalPDF_uncert2"])
            
            if np.isnan(modelPDF).any():
                print('nan in model array with xi=',xi, self.computedBestFit)
            
            bins_to_use = (modelPDF>0.)&(modelPDF_uncert2>0.)
            
            #Effective likelihood from [arXiv:1901.04645] 
            #L = beta**alpha * Gamma(k+alpha)/ (k! * (1+beta)**(k+alpha) * Gamma(alpha)) - Equation 3.15
            #log L = sum(alpha * log[beta] + log[Gamma(k + alpha)] - (k + alpha)*log[1+beta] - log[Gamma(alpha)]) 
            #where alpha=(mu**2/sigma**2)+a and beta = (mu/sigma**2)+b with a=1 and b=0
            alpha = np.power(modelPDF[bins_to_use],2.)/modelPDF_uncert2[bins_to_use] + 1.
            beta  = modelPDF[bins_to_use]/modelPDF_uncert2[bins_to_use]
            k = self.observation[bins_to_use]
            values = [alpha*np.log(beta), 
                      sps.loggamma(k+alpha).real, 
                      -sps.loggamma(k+1.).real,
                      -(k+alpha)*np.log1p(beta), 
                      -sps.loggamma(alpha).real]
            
            #print ("Bins not used in likelihood:", len(np.where((modelPDF<=0.) | (modelPDF_uncert2<=0.))[0]))

        else:
            raise ValueError('No valid LLH type defined!')
            
            
        return -np.sum(values)

    
    
    def ComputeBestFit(self):
        
        if self.allowNegativeSignal:
            lower_bound = -1.
        else:
            lower_bound = 0.
        
        #Use migrad minimizer from iMinuit package
        LLHmin_DM = Minuit(self.evaluateLLH, xi=0.1, error_xi=.01, limit_xi=(lower_bound,2.), errordef=.5 ,print_level=0)   
        LLHmin_DM.migrad()
        
        self.bestFit = {}
        self.bestFit['xi'] = LLHmin_DM.fitarg['xi'] #Give current Minuit state in form of a dict
        self.bestFit['LLH'] = self.evaluateLLH(self.bestFit['xi'])
                
        self.computedBestFit = True
        
        
        
    def ComputeTestStatistics(self,xi_ref=0.):
        
        if not self.computedBestFit:
            self.ComputeBestFit()
            
        self.TS = 0.

        #Null hypothesis
        self.bestFit['LLH_ref'] = self.evaluateLLH(xi_ref)
        #As defined in the asimov paper, otherwise is 0
        if self.bestFit['xi'] > xi_ref:
            self.TS = 2*(self.bestFit['LLH_ref']-self.bestFit['LLH'])
            
        if self.TS < 0:
            self.TS = 0
            
            
    #------------------------------------------------------------------------------------------
    #Sensitivity computation using Likelihood Intervals
    #Rely on Wilk's theorem
    #Need to make sure TS distribution follows a chi2
    #------------------------------------------------------------------------------------------
            
    def CalculateUpperLimit(self,conf_level):

        nIterations = 0
        eps_TS = 0.005
        eps_param = 0.0005

        #Default C.L.
        deltaTS = 2.71
        if conf_level==90:
            deltaTS = 1.64
        elif conf_level==95:
            deltaTS = 2.71
            
        param_low = self.bestFit['xi']
        param_up = self.bestFit['xi']
        param_mean = self.bestFit['xi']
        
        dTS = 0
        cc = 1
        while((dTS<deltaTS) and (nIterations<100)):
            
            nIterations += 1 
              
            if param_up < 1e-14:
                param_up = 1e-14

            param_up=param_up+3.*np.abs(param_up)

            if param_up<0.:
                TS_fix = 0.
            else:
                TS_fix = 2*(self.bestFit['LLH_ref']-self.evaluateLLH(param_up))
                            
            dTS = self.TS - TS_fix

        nIterations = 0
        param_low = param_up/4.
        
        while((cc>0.) and (nIterations<100)):
            
            nIterations += 1
            param_mean = (param_low+param_up)/2.
            
            if param_mean <0.:
                TS_fix = 0.
            else:
                TS_fix = 2*(self.bestFit['LLH_ref']-self.evaluateLLH(param_mean))
                
            dTS = self.TS - TS_fix
            
            if(dTS<deltaTS):
                param_low = param_mean
                delta_param = (param_up-param_low)/(param_up)
              
                if((dTS>deltaTS-eps_TS) and (delta_param < eps_param)):
                    cc = 0
                    
            if(dTS>deltaTS):
                param_up = param_mean
                delta_param = (param_up-param_low)/(param_up)
                
                if((dTS<deltaTS+eps_TS) and (delta_param < eps_param)):
                    cc = 0
                    
        return param_up
    
    
    
    def CalculateSensitivity_LikelihoodIntervals(self, nTrials, conf_level):

        if self.LLHtype == None:
            raise ValueError('LLH type not defined yet!')

        TS = []
        upperlimits = []
        if self.moreOutput:
            fits = []
        
        #Sample pseudo-experiments from BG only
        for i in range(nTrials):
            self.sampleObservation(xi=0.)
            self.ComputeTestStatistics()
            TS.append(self.TS)
            
            #Compute upper limit for each pseudo-experiment
            ul = self.CalculateUpperLimit(conf_level)
            if np.isnan(ul):
                print("Warning: NaN upper limit at trial {i}.\nRepeating trial.".format(i=i))
                i-=1
                continue
            upperlimits.append(ul)
            
            if self.moreOutput:
                fits.append(self.bestFit)
        
        #Sensitivity
        #Take median value of upper limits
        p_median = np.percentile(upperlimits, 50)
        #Compute 1 and 2 sigma bands
        p_95_low = np.percentile(upperlimits, 2.5)
        p_95_high = np.percentile(upperlimits, 97.5)
        p_68_low = np.percentile(upperlimits, 16.)
        p_68_high = np.percentile(upperlimits, 84.)

        dic_brazilian = {}
        dic_brazilian['TS_dist'] = TS
        dic_brazilian['error_68_low'] = p_68_low
        dic_brazilian['error_68_high'] = p_68_high
        dic_brazilian['error_95_low'] = p_95_low
        dic_brazilian['error_95_high'] = p_95_high   
        dic_brazilian['median'] = p_median
        if self.moreOutput:
            dic_brazilian['upperlimits'] = upperlimits
            dic_brazilian['bestFits'] = fits
            
        return dic_brazilian
    
    
    
    #------------------------------------------------------------------------------------------
    #Sensitivity computation using frequentist approach
    #Require more computation resources
    #------------------------------------------------------------------------------------------
    
    def DoScan(self, ni_min, ni_max, nstep, ts, conf_level, precision):
        
        print(" Scanning injected fraction of events range [%i, %i]" % (ni_min, ni_max))
      
        results = []
        step = 0
        frac = 0.
        tolerance = 0.1
        
        ni_mean = 0
        frac_error = 0 
        ni_prev = 0
        frac_prev = 0 
        
        nTrials = inv_BinomialError(precision, conf_level)
        
        print (" Doing %i trials for %i +/- %.1f C.L." %(nTrials, conf_level, precision))
        
        p = conf_level/100.
        
        ni = ni_min
        while (ni <= ni_max and ni >= ni_min):
            dic_results = {}
            
            TS = []
            for i in range(nTrials):
                self.sampleObservation(n=ni)
                self.ComputeTestStatistics()
                TS.append(self.TS)

            TS = np.array(TS)
            n = TS[np.where(TS > ts)].size
            ntot = TS.size
            frac = n/ntot
            frac_error = BinomialError(ntot,n)/ntot

            print(" [%2d] ni %i, [ni_min %i, ni_max %i], n %4d, ntot %4d, C.L %.2f +/- %.2f" %
                (step, ni, ni_min, ni_max, n, ntot, frac * 100, frac_error * 100))
            
            
            dic_results['ni'] = ni
            dic_results['xi'] = ni/self.NEvents

            dic_results['n'] = n

            dic_results['ntrials'] = ntot
            dic_results['TS_dis'] = TS
            step += 1    
            results.append(dic_results)
            
            if (np.abs(frac - p) < frac_error):
                print ("Finishing...")
                print (ni, self.NEvents)
                break
            
            if frac > p:
                ni = ni - 1                
            else:
                ni = ni + 20        

        return ni/self.NEvents, results
    
    
    def CalculateSensitivity_Frequentist(self,  conf_level, precision=0.5, factor = 3., first_guess = None):
        
        #First compute sensitivity using likelihood intervals
        #Considering 100 trials seem enough
        sens = self.CalculateSensitivity_LikelihoodIntervals(1000, conf_level)
        
        #Test statistics
        median_ts = np.percentile(sens['TS_dist'], 50)
        ts_95_low = np.percentile(sens['TS_dist'], 2.5)
        ts_95_high = np.percentile(sens['TS_dist'], 97.5)
        ts_68_low = np.percentile(sens['TS_dist'], 16.)
        ts_68_high = np.percentile(sens['TS_dist'], 84.)
        
        #If first_guess not given as a parameter
        if (first_guess is None): 
            print ("No first guess given... guessing it from Likelihood Intervals")    
            first_guess = int(self.NEvents * sens['median'])
        
        print (" Median TS: %.2f" %median_ts)
        print (" First guess: %.4f" %first_guess)
        
        p = conf_level/100.
        
        print ("p:", p)
        
        #Define range for number of injected signal events
        ni_min = first_guess / factor * (1 - p)
        ni_max = first_guess * factor / p
        
        return self.DoScan(ni_min, ni_max, 30, median_ts,  conf_level, precision)
    
    def CalculateErrors_Frequentit(self, conf_level, precision):
        
        sens = self.CalculateSensitivity(100, conf_level)
        
        ts_95_low = np.percentile(sens['TS_dist'], 2.5)
        ts_95_high = np.percentile(sens['TS_dist'], 97.5)
        ts_68_low = np.percentile(sens['TS_dist'], 16.)
        ts_68_high = np.percentile(sens['TS_dist'], 84.)
        
        first_guess = sens['median']
        p = conf_level/100.
        factor = 3.
        
        ni_min = first_guess / factor * (1 - p)
        ni_max = first_guess * factor / p
        
        errors = {}
        print (" Median TS 68% low: %.2f" %ts_68_low)
        errors["error_68_low"] = self.DoScan(ni_min, ni_max, 30, ts_68_low,  conf_level, precision)
        print (" Median TS 68% high: %.2f" %ts_68_high)
        errors["error_68_high"] = self.DoScan(ni_min, ni_max, 30, ts_68_high,  conf_level, precision)
        print (" Median TS 95% low: %.2f" %ts_95_low)
        errors["error_95_low"] = self.DoScan(ni_min, ni_max, 30, ts_95_low,  conf_level, precision)
        print (" Median TS 95% high: %.2f" %ts_95_high)
        errors["error_95_high"] = self.DoScan(ni_min, ni_max, 30, ts_95_high,  conf_level, precision)
        return errors    
    
    
    def CalculateDiscoveryPotential(self, significance):

        signalStrength = 0.
        TS = 0.
       
        while TS<np.power(significance,2.):
            self.sampleObservation(xi=signalStrength)
            self.ComputeTestStatistics()
            TS = self.TS
            signalStrength += 0.1*self.bestFit['xi']
            
        return signalStrength