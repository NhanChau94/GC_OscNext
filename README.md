# GC_OscNext
Galactic Center DM search with IceCube using OscNext event selection.
[Wikipage](https://wiki.icecube.wisc.edu/index.php/Data_Driven_Galactic_Centre_Dark_Matter_Search_with_OscNext) 

## Prerequisites
- [`KDEpy`](https://github.com/tommyod/KDEpy): for performing FFTKDE - a very fast convolution-based kde estimation
- [`χarον (charon)`](https://github.com/icecube/charon): an IceCube package to organize calculations of neutrinos from dark matter annihilation/decay.

## Structure of the code
- DMfit: package to perform likelihood method for Dark Matter searches in IceCube
- Samples: scripts to process the MC/data to the format used in this analysis
- DetResponse: tools to make detector response matrix from Monte Carlo
- Spectra: codes to organize calculation of spectra from Charon/PPPC4 and extract the precomputed Jfactor from Clumpy.
- PDFs: Computation of Signal PDF (Signal.py) and Background PDF (Background.py)
- Sensitivity: Main analysis scripts to perform signal recovery fit, test statistic distribution and upper limit calculation
- PlotScipts: notebooks to produce plots shown in the wikipage

## How to run
- 
