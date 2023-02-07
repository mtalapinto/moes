# moes

Ray tracing algorithm for echelle spectrographs

## Description

moes a ray tracing package written in python that calculates the path of single rays through optical surfaces based on Fermat's principle. It models the optical paths of light rays through an echelle spectrograph from the slit to the detector. It can be used to build the spectrograph's wavelength solution and to characterize the expected instrumental RV systematics arising from changes in spectrograph's environment.

In the most cases the modules that describe the optical elements of the spectrograph are: slit, FN-system (FN stands for F-number, also known as focal ratio), collimator, echelle grating, transfer mirror, cross-dispersion grism, camera and detector.

### Dependencies
**Python 3.8+ highly recommended!**

**moes** makes use of the following python libraries: numpy, astropy, scipy, simanneal, dynesty, matplotlib

### Downloading

The code is self-contained therefore it is not necessary to install anything. Just download the instrument folder and run the *optimization.py* script by doing *python3 optimization.py*.

### Running **moes**

Each instrument folder has an *optimization.py* file that contains three functions. 
The function *run_instrument_model()* creates a ray tracing model of the instrument based on a list of spectral lines provided by the **caracal** (https://github.com/caracal-pipeline/caracal) pipeline. 
The function *fit_instrument_model(type)* starts an optimization algorithm to fit a ray tracing instrument model to the calibration data. There are two types of optimizers implemented: simulated annealing (*type=simulated-annealing*) and nested sampling (*type=nested-sampling*).
Once found the best-fit instrumental parameters, one can calculate chromatic aberrations by using the function *fit_chromatic_aberrations()*.
Finally, once one has corrected for chromatic aberrations, it is possible to estimate the amount of optical aberrations using the function *fit_optical_aberrations()*.

We emphasize that these functions work as an example for running the **moes** package to create a spectrograph model and find its best fit parameters. 
In each instrument folder there is a *data* folder that contain the instrumental and aberrations parameters for the example of the basic functions shown above, the time series of the *caracal* wavelength solutions and the time series of the **moes** instrumental and aberration parameters that best fit the *caracal* data.

### Additional data
Using the results from **caracal** and **moes** it is possible to compute the temperature zero points (TZP) from the calculation of differential drifts.

Differential drifts are defined by

$\delta f_n = \frac{1}{N_n}\sum_{k=1}^{N_n} \delta f_{n, k}$

with 

$\delta f_{n, k} = (x_{{\rm A},k} - x_{{\rm B},k})_n~~$


where $x_{{\rm A},k}$ and $x_{{\rm B},k}$ are the wavelength positions for the science and calibration spectra in pixels, respectively, $\delta f_{n, k}$ denotes the difference between the x-positions of the science and calibration spectra, and $\delta f_{n}$ is the average of $\delta f_{n, k}$ for a given night $n$. The TZP are defined as 

$TZP_i = s\cdot (\delta f_{n, M} + \delta f_{i, C} (T_i) - \delta f_{n, C}(T_c))~~$

where $s$ is the pixel-to-rv scaling factor, and $T_i$ and $T_c$ correspond to the environmental vectors with temperatures and pressure measured at the observation and calibration times, respectively. The $i$-index refers to each single science observation.

The time series of the CARMENES VIS differential drifts and TZPs calculated using **moes** can be found in the *data/drifts/* and in the *data/tzp/* folders.

## Authors

Marcelo Tala Pinto 

## Version History

moes v1.0
