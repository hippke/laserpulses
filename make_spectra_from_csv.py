import numpy
import csv
import os
import pyfits
import matplotlib.pylab as mpl
from math import pow
from scipy.interpolate import interp1d
from gatspy.periodic import LombScargleFast


def find_nearest(array, value):
    idx = (numpy.abs(array - value)).argmin()
    return idx


def moving_average(a, n) :
    ret = numpy.cumsum(a, dtype = float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def nvalues(filename):
    csv = numpy.genfromtxt(filename, delimiter=";")
    wavelengths = csv[:,0]
    flux = csv[:,1]

    # Convert wavelengths to frequencies
    # wavelengths = numpy.power(10, tbdata['loglam'])  # in Angstrom; de-logarithm
    lightspeed = 299792.458  # km/s
    frequencies = numpy.divide(lightspeed * pow(10, 13), wavelengths)  # Hz

    # Interpolate linear up to 100k and down to 4k
    UpSteps = 20000
    TargetSteps = 10000
    frequencies_interpolated = interp1d(numpy.linspace(0, 1, len(frequencies)), frequencies, kind='linear') \
        (numpy.linspace(min(numpy.linspace(0, 1, len(frequencies))), max(numpy.linspace(0, 1, len(frequencies))), num=UpSteps))
    flux_interpolated = interp1d(numpy.linspace(0, 1, len(flux)), flux, kind='linear') \
        (numpy.linspace(min(numpy.linspace(0, 1, len(flux))), max(numpy.linspace(0, 1, len(flux))), num=UpSteps))
    linearized_frequencies = numpy.linspace(min(frequencies), max(frequencies), TargetSteps)
    linearized_flux = numpy.zeros(TargetSteps)
    for i in range(len(linearized_frequencies)):
        linearized_flux[i] = flux_interpolated[find_nearest(frequencies_interpolated, linearized_frequencies[i])]

    # Generate moving average array of proper size and subtract it from flux values
    windowsize = int(len(linearized_flux) / 100 * 2.5)
    mymovingaverage = moving_average(linearized_flux, windowsize)  
    positions_to_fill = len(linearized_flux) - len(mymovingaverage)  # No average possible
    positions_end_fill = int(0.5 * positions_to_fill)  # Fill with zeros instead
    positions_start_fill = positions_to_fill - positions_end_fill
    mymovingaverage = numpy.append(mymovingaverage, numpy.zeros(positions_end_fill))
    mymovingaverage = numpy.append(numpy.zeros(positions_start_fill), mymovingaverage)
    cleaned_flux = numpy.subtract(linearized_flux, mymovingaverage)

    # Alternatively, cut off the first and last 2.5 percent of values where not interpolation is available
    linearized_frequencies = linearized_frequencies[positions_start_fill:-positions_end_fill]
    cleaned_flux = cleaned_flux[positions_start_fill:-positions_end_fill]

    steprate = ((1 / linearized_frequencies[-1]) + (1 / linearized_frequencies[0])) / 2

    middle = int(len(linearized_frequencies) / 2)
    steprate = 1 / linearized_frequencies[middle]

    z = 0.0581  # To mark claimed signal(s) in the figure
    rate = 2.1064e-15
    manual_n = 52
    manual_peak = (manual_n - 1) * rate 
    peak1 = (51.75 + z * 51.75) * rate
    peak2 = (48.75 + z * 48.75) * rate
    fmin = 0.5e-13
    fmax = 2e-13 

    fmin = rate
    fmax = 2000 * rate

    N = 10000
    df = (fmax - fmin) / N
    model = LombScargleFast().fit(linearized_frequencies, cleaned_flux)
    power = model.score_frequency_grid(fmin, df, N) 
    continuum = numpy.std(power)
    power = numpy.divide(power, continuum)  # Normalize to S/N

    freqs = fmin + df * numpy.arange(N)
    print('Highest peak at [sec]', freqs[numpy.argmax(power)], 'with S/N', numpy.max(power))

    # Make figure
    fig = mpl.figure()
    ax = fig.add_subplot(111)
    mpl.plot([peak1, peak1], [0, max(power) * 1.1], color='b', linestyle='--', linewidth=0.5)
    mpl.plot([peak2, peak2], [0, max(power) * 1.1], color='b', linestyle='--', linewidth=0.5)
    mpl.ylim([0, max(power)])
    mpl.xlim([fmin, fmax])
    mpl.plot(freqs, power, color='k', linestyle='-', linewidth=1)
    ax.set_title('Periodogram / '+ myfilename)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Normalized power')
    mpl.savefig(myfilename + '.pdf')
    mpl.savefig(myfilename + '.png')
    mpl.show()



myfilename = 'CGCG 390-073-SDSS'
nvalues(myfilename + '.csv')

myfilename = 'CGCG 390-073-NED'
nvalues(myfilename + '.csv')