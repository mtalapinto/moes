import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


def get_snr(det, sn):
    stardata = pd.read_csv('stellar_template/stellar_template_v2.tsv', sep=' ')
    wini = 3800
    wend = 6800
    if sn == 0:
        basedir = '/melian/moes/platospec/data/pix_exp/'+'ccd_'+str(det)+'/0/'
        n = 12
    elif sn == 1 or sn == 5 or sn == 10:
        basedir = '/melian/moes/platospec/data/pix_exp/sn_'+str(sn)+'/' + 'ccd_' + str(det) + '/0/'
        n = 10
    else:
        basedir = 'carlos'

    snrout = []
    if os.path.exists(basedir):
        orderfiles = glob.glob(basedir + "*moes*")
        # print(orderfiles)
        for order in orderfiles:
            print(order)
            orderdata = pd.read_csv(order, sep=',')
            flux = orderdata['flux'].values
            snrout.append(DER_SNR(flux))
            print(DER_SNR(flux))

    else:
        print('No fun at all...')

    snrout = np.array(snrout)
    snr_final = np.mean(snrout)
    error = np.std(snrout)
    print(snr_final)
    return snr_final


# =====================================================================================

def DER_SNR(flux):
    # =====================================================================================
    """
    DESCRIPTION This function computes the signal to noise ratio DER_SNR following the
                definition set forth by the Spectral Container Working Group of ST-ECF,
            MAST and CADC.

                signal = median(flux)
                noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
            snr    = signal / noise
                values with padded zeros are skipped

    USAGE       snr = DER_SNR(flux)
    PARAMETERS  none
    INPUT       flux (the computation is unit independent)
    OUTPUT      the estimated signal-to-noise ratio [dimensionless]
    USES        numpy
    NOTES       The DER_SNR algorithm is an unbiased estimator describing the spectrum
            as a whole as long as
                * the noise is uncorrelated in wavelength bins spaced two pixels apart
                * the noise is Normal distributed
                * for large wavelength regions, the signal over the scale of 5 or
              more pixels can be approximated by a straight line

                For most spectra, these conditions are met.

    REFERENCES  * ST-ECF Newsletter, Issue #42:
                www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
                * Software:
            www.stecf.org/software/ASTROsoft/DER_SNR/
    AUTHOR      Felix Stoehr, ST-ECF
                24.05.2007, fst, initial import
                01.01.2007, fst, added more help text
                28.04.2010, fst, return value is a float now instead of a numpy.float64
    """
    from numpy import array, where, median, abs

    flux = array(flux)

    # Values that are exactly zero (padded) are skipped
    flux = array(flux[where(flux != 0.0)])
    n = len(flux)

    # For spectra shorter than this, no value can be returned
    if (n > 4):
        signal = median(flux)

        noise = 0.6052697 * median(abs(2.0 * flux[2:n - 2] - flux[0:n - 4] - flux[4:n]))

        return float(signal / noise)

    else:

        return 0.0


# end DER_SNR -------------------------------------------------------------------------


if __name__ == '__main__':
    det = 0
    snr = get_snr(1, 0)
    print(snr)
    snr = get_snr(2, 0)
    print(snr)
    '''
    data = pd.read_csv(basedir+'ccd_'+str(det)+'/'+str(rv)+'/'+str(order)+'_2D_moes.tsv', sep=',')
    wini = min(data['wave'].values)
    wend = max(data['wave'].values)
    stardata = stardata.loc[stardata['WAVE'] <= wend*1e4]
    stardata = stardata.loc[stardata['WAVE'] >= wini*1e4]
    snr_temp = DER_SNR(stardata['FLUX'])
    print(snr_temp)
    # we add noise of 5%
    snr_i = DER_SNR(data['flux'].values)
    print(snr_i)
    ns = np.random.normal(0, 1*0.1, len(data))
    data['flux_ns'] = data['flux'] + ns
    snr_f = DER_SNR(data['flux_ns'].values)
    print(snr_f)
    plt.figure(figsize=[8, 3])
    plt.plot(data['wave'], data['flux'], 'k-', alpha=0.5)
    plt.plot(data['wave'], data['flux_ns'], 'b-', alpha=0.5)
    plt.show()
    # signal = np.median(data['flux'].values)
    # noise = np.sqrt(signal)
    # snr_ori = signal/noise
    # signal_new = np.median(data['flux_ns'].values)
    # noise_new = np.sqrt(signal_new)
    # snr_new = signal_new / noise_new
    # print(snr_ori, snr_new)
    '''
