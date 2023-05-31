# GC_OscNext
Galactic Center DM search with IceCube using OscNext event selection.

[Wikipage](https://wiki.icecube.wisc.edu/index.php/Data_Driven_Galactic_Centre_Dark_Matter_Search_with_OscNext) 

## Prerequisites
- [`KDEpy`](https://github.com/tommyod/KDEpy): for performing FFTKDE - a very fast convolution-based kde estimation
- [`χarον (charon)`](https://github.com/icecube/charon): an IceCube package to organize calculations of neutrinos from dark matter annihilation/decay.

## Structure of the code
Each directories contain scripts for following purposes:
- DMfit: package to perform likelihood method for Dark Matter searches in IceCube
- Samples: scripts to process the MC/data to the format used in this analysis
- DetResponse: tools to make detector response matrix from Monte Carlo
- Spectra: codes to organize calculation of spectra from Charon/PPPC4 and extract the precomputed Jfactor from Clumpy.
- PDFs: Computation of Signal PDF (Signal.py) and Background PDF (Background.py)
- Sensitivity: Main analysis scripts using the above tools to perform signal recovery fit, test statistic distribution and upper limit calculation
- PlotScripts: notebooks to produce plots shown in the wikipage

## How to produce the analysis results
- Edit the `GC_DM_OUTPUT` in the `env.sh` to the desired path for storing results. Then `source env.sh`
- The results are produced by python scripts in `Sensitivity` directory:
  - `SignalRecovery.py`: signal recovery fit
  - `TS_distribution.py`: Test statistics distribution
  - `UpperLimit_LLH_Interval.py`: upper limit calculation using likelihood interval method
  - In case needed: `TS_submit.ipynb`, `UL_submit.ipynb`, `SignalRecovery_submit.ipynb` are used to create the necessary files for submitting jobs to Madison cluster. 
  * The results of these scripts should be store in `GC_DM_OUTPUT` and the corresponding plots can be made using `./PlotScripts/Sensitivity.ipynb`

- The scripts for showing intermidiate steps are also stored in `./PlotScripts`
  - `Spectra.ipynb`: Compute and plot neutrino spectra using functions and class in `./Spectra/NuSpectra.py`
  - `Jfactor.ipynb`: Compute and plot DM density and Jfactor using functions and class in `./Spectra/Jfactor.py`
  - `PDFs.ipynb`: Compute and plot Signal + Background PDF using functions and class in `./PDFs`
