__author__ = 'akaminsk'

import argparse
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.stats import median_absolute_deviation
from astropy.time import Time
from scipy.optimize import curve_fit
import time


def nanwmean(array, w=None):
    return np.nansum(array * w) / np.nansum(w)

def nanwstd(array, w=None):
    demeaned = array - nanwmean(array, w)
    return np.sqrt(nanwmean(demeaned**2, w))

def linfit_lev(xv, yv, dyv):
    # remove NaN values
    ind = np.where(np.isfinite(xv))
    x = xv[ind] * 1.
    y = yv[ind] * 1.
    dy = dyv[ind] * 1.

    # Fit line analytically:
    matrix = np.asarray([[np.sum(x**2./dy**2.), np.sum(x/dy**2.)], [np.sum(x/dy**2.), np.sum(1./dy**2.)]])

    y_sigma = y / dy**2.
    free_vector = np.asarray([np.sum(x*y_sigma), np.sum(1.*y_sigma)]).T
    solution = np.linalg.lstsq(matrix, free_vector, rcond=None)[0]
    err = np.sqrt(np.diag(np.linalg.inv(matrix)))

    return solution, err

def func_lin(x, a, b):
    return a*x + b

def fit_lin(x, y, e_y, method = "ana"):
    if method == "ana": #use analytical method like Lev
        result = linfit_lev(x, y, e_y)
    else: # do a weighted regression
        popt, pcov = curve_fit(func_lin, x, y, sigma=e_y, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        result = [popt, perr]

    return result


def find_duplicates(ar):
    # search the 1D array for dublicate entries
    # output: nested list of dublicate value indices (for example [1,2,2,2,3,4,5,6,6,7] gives [[1,2,3],[7,8]])
    dub = []
    uniquevals = np.unique(ar)
    for val in uniquevals:
        ind_val = np.where(ar == val)[0]
        if len(ind_val) > 1:
            dub.append(ind_val)
    return dub

def wmed(arr, weights):

    #normalize the weights
    w_norm = weights / np.sum(weights)
    i_sort = np.argsort(arr)

    arr_sort = arr[i_sort]
    w_norm = w_norm[i_sort]

    cdf = np.array([np.sum(w_norm[0:i+1]) for i in range(len(arr))])   # there is also np.cumsum

    i_lower = np.searchsorted(cdf, 0.5)

    if cdf[i_lower] == 0.5:
        wmed = (arr_sort[i_lower] + arr_sort[i_lower+1]) / 2
    elif cdf[i_lower] > 0.5:
        wmed = arr_sort[i_lower]
    else:
        wmed = arr_sort[i_lower+1]
    return wmed

def savedat(filename, *cols, **kwargs):
    with open(filename, "w") as f:
        for line in zip(*cols):
            print(*line, file=f, **kwargs)


def rvc2avc(filename, suf=''):
    '''
    Apply NZP correction to rvc data
    '''
    global Nnodrift, k, Nrv, rv_std_vec_corr, rv_med_err_corr

    M = np.loadtxt(filename).T
    M = M[:,np.argsort(M[0])]   # sort by time

    # Empty files could be caused by not-drift-corrected measurement:
    if not M.size:
        print(starname+".dat should be an empty file.")
        return

    # read matrix into vectors:
    ind = M[0] >= jd_min
    M = M[:,ind]   # remove time stamps before jd_min
    bjd, rvc, e_rvc, drift, e_drift, rv, e_rv, brv, sa = M[0:9]

    #     M(isnan(drift_tmp),:)=[]; % 20170606 LT: DO NOT Remove measurements with no drift correction, only from obj.avc.dat files!
    flag_rvc = 1 * np.isnan(drift)
    Nnodrift[k] = flag_rvc.sum()
    good_rvc = np.where(flag_rvc==0)[0]
    Nrv[k] = len(good_rvc)

    # Subtract previously calculated weighted mean (zero-point of the star):
    rvc = rvc - rvc_mean_vec[k]   # stellar zero-point subtracted RVs  # MZ: Needed? AVC mean will be subtracted

    #==========================================================
    # subtract the nightly mean and co-add its error:
    nights = np.floor(bjd).astype(int)
    columns = nights - jd_min   # index list n
    nzp = nights_mean[columns]
    e_nzp = e_nights_mean[columns]
    e_corr = e_nzp * 1.
    corr_t_onlydrift = np.zeros_like(bjd)
    e_corr_onlydrift = np.zeros_like(bjd) + 0.000001 # MZ: default for no drift correction?

    if drift_corr:
        premap = nights < drift_jd_max
        modbjd_t = np.mod(bjd[premap],1) - 0.5
        corr_t_onlydrift[premap] = b1 + modbjd_t*a1
        e_corr_onlydrift[premap] = np.sqrt(db1**2. + (modbjd_t*da1)**2.)
        e_corr[premap] = np.sqrt(e_nzp[premap]**2. + db1**2. + (modbjd_t*da1)**2.)

    # Apply correction
    corr_t = nzp + corr_t_onlydrift
    avc = rvc - corr_t
    e_avc = np.sqrt(e_rvc**2. + e_corr**2.)
    flag_avc = 1*flag_nights[columns] | 2*flag_rvc   # 1: NZP is based on RVs from adjacent nights.

    #==========================================================

    # Subtract again the weighted mean (zero-point of the star), but do not change uncertainties:
    print("Nrv =", Nrv[k])
    if Nrv[k]:
        weights = e_avc**-2
        avc_mean = nanwmean(avc, w=weights)
        print(avc_mean)
        avc -= avc_mean    # stellar zero-point subtracted RVs

    # Find outliers per star (do not rely on RV errors - they do not account for systematics yet):
    avc_median = np.nanmedian(avc[good_rvc])
    bias = 1-1/4/len(good_rvc) if len(good_rvc) else np.inf
    rv_std = 1.48 * median_absolute_deviation(avc[good_rvc], ignore_nan=True) / bias
    rv_std_vec_corr[k] = rv_std
    rv_med_err_corr[k] = np.nanmedian(e_avc[good_rvc])

    Mavc = bjd, avc, e_avc, drift, e_drift, rv, e_rv, brv, sa, corr_t, e_corr, nzp, e_nzp, corr_t_onlydrift, e_corr_onlydrift, flag_avc

    if savedata:
        # Write an entry to .orb file (only drift+NZP corrected RVs, NO outliers removed):
        fid.write(f"STAR: {starname}\n")
        for j,line in enumerate(zip(bjd, 0.001*rvc, 0.001*e_rvc)):
            if np.isfinite(drift[j]):
                print(*line, file=fid)
        fid.write("END\n")

        # Write a .avc.dat file (only drift+NZP corrected RVs, NO outliers removed):
        j = flag_avc < 2
        savedat(outdir+f"/avc{suf}/{starname}.avc.dat", bjd[j], avc[j], e_avc[j])

        # Write a .avcn.dat file (All RVs, NO outliers removed):
        savedat(outdir+f"/avcn{suf}/{starname}.avcn.dat", *Mavc[:11], flag_avc)

        # Write a .avcn_drift.dat file with two extra column where only drift is written to (All RVs, NO outliers removed):
        savedat(outdir+f"/avcn{suf}/{starname}.avcn_drift.dat", *Mavc)

    return Mavc


def robust_nzp():
    '''
    Loop trough the nights and calculate in each night a robust mean.
    '''

    global f_mat, jd_mat, rv_mat, err_mat
    global nights_mean, nights_median, e_nights_mean, nights_mean_std, Nrv_night, Nout_night

    # Remove hidden-variable stars from list
    if remove_var:
        stars_flag[[(star in variable) for star in stars]] |= 4

    var_rv = np.where(stars_flag)

    # initialize some vectors:
    nights_mean = np.zeros(N_n) * np.nan
    nights_median = nights_mean * 1.0
    e_nights_mean = nights_mean * 1.0
    nights_mean_std = nights_mean * 1.0
    Nrv_night = nights_mean * 1.0
    Nrv_night_ini = nights_mean * 1.0
    Nout_night = nights_mean * 1.0

    # use the results of stage 1 (the sparse RV-night matrices):
    jd_mat = jd_mat_orig * 1.0
    rv_mat = rv_mat_orig * 1.0
    err_mat = err_mat_orig * 1.0

    # Remove the RVs of targets with less than Nrv_min drift-corrected RVs (only for the purpose of calculating the NZPs):
    jd_mat[Nrv<Nrv_min,:] = np.nan
    rv_mat[Nrv<Nrv_min,:] = np.nan
    err_mat[Nrv<Nrv_min,:] = np.nan

    # Remove the RVs of variable targets (only for the purpose of calculating the NZPs):
    jd_mat[var_rv,:] = np.nan
    rv_mat[var_rv,:] = np.nan
    err_mat[var_rv,:] = np.nan

    f_mat = 1 * np.isnan(rv_mat)

    for n in range(N_n):
        night_jds = jd_mat[:,n] * 1.0
        night_rvs = rv_mat[:,n] * 1.0
        night_err = err_mat[:,n] * 1.0

        # remove outliers per night (but plot them) only if there are enough RVs:
        good_rvs_tmp = np.where(np.isfinite(night_rvs))[0]
        print(good_rvs_tmp)
        Nrv_night_tmp = len(good_rvs_tmp)
        if Nrv_night_tmp > Nrv_min_night:
            rv_median_night = np.nanmedian(night_rvs)
            bias = 1. - 1./4./len(night_rvs[np.isfinite(night_rvs)])
            rv_std_night = 1.48 * median_absolute_deviation(night_rvs, ignore_nan=True) / bias
            bad_rv_night = np.where((np.abs(night_rvs-rv_median_night) > sigma_outlier_night*rv_std_night) & (np.abs(night_rvs-rv_median_night) > init_rv_var))[0]
            #print(len(bad_rv_night))

            night_jds[bad_rv_night] = np.nan
            night_rvs[bad_rv_night] = np.nan
            night_err[bad_rv_night] = np.nan
            Nout_night[n] = len(bad_rv_night)
            f_mat[bad_rv_night,n] |= 2
        #==========================================================
        # Calculate the nightly zero-point RV using only good RVs:
        good_rvs = np.where(np.isfinite(night_rvs))[0]
        Nrv_night[n] = len(good_rvs)
        print('n value = ', Nrv_night[n])
        bias = 1. - 1./4./Nrv_night[n]
        weights = night_err**-2
        if np.isfinite(weights).any():
            nights_mean[n] = nanwmean(night_rvs[good_rvs], weights[good_rvs])
            e_nights_mean[n] = 1. / np.sqrt(np.sum(weights[good_rvs])) / bias
            # DEBUG: 2016 LT: a more robust way?:
            nights_median[n] = wmed(night_rvs[good_rvs], weights[good_rvs])
            nights_mean_std[n] = nanwstd(night_rvs[good_rvs], w=weights[good_rvs]) / np.sqrt(Nrv_night[n]) / bias
            # 20170601 MZ: We should make a maximum likelihood estimation of
            #              the NZPs and the stellar ZPs simultaneously!
            # 20170118 LT: Take the max of e_nights_mean and nights_mean_std:
            e_nights_mean[n] = np.nanmax([e_nights_mean[n],nights_mean_std[n]])

    global flag_nights, bad_mean, good_mean, median_nights_mean, std_nights_mean, nights_vec

    # ==========================================================================
    # find nights where NZP correction could not be applied and mark them:
    flag_nights = Nrv_night < Nrv_min_night   # flag 1 for too few RV points
    bad_mean = np.where(flag_nights)[0]
    good_mean = np.where(Nrv_night>0)[0]
    median_nights_mean = np.nanmedian(nights_mean[~flag_nights])
    std_nights_mean = 1.48 * median_absolute_deviation(nights_mean[~flag_nights])
    nights_vec = np.arange(N_n)

    # display some statistics:
    print("median(NZP) =", median_nights_mean)
    print("std(NZP) =", std_nights_mean)

    # save nzps before smoothing:
    nzp_good = nights_mean * 1.0
    e_nzp_good = e_nights_mean * 1.0

    #plt.plot(xplot[bad_mean], xplot[bad_mean]*0.0+1.5, "rx", label="bad mean")
    # correct for no-mean nights by averaging adjascent nights (like smoothing)
    for y in bad_mean:
        val_before = nights_mean[y]
        n_win = np.arange(y-average_window, y+average_window+1)
        # now remove indices <0 and > N_n
        n_win = n_win[(n_win>=0) & (n_win<N_n)]
        # n_win = n_win[np.isfinite(nights_mean[n_win]) & np.isfinite(e_nights_mean[n_win])] # does not give the exact same output!?

        nights_mean[y] = nanwmean(nights_mean[n_win][np.isfinite(nights_mean[n_win])], w=e_nights_mean[n_win][np.isfinite(e_nights_mean[n_win])]**-2)
        val_after = nights_mean[y]
        # print("Averaging index ", y, "before - after= ", val_before, " - ", val_after)
        e_nights_mean[y] = nanwmean(e_nights_mean[n_win][np.isfinite(e_nights_mean[n_win])], w=e_nights_mean[n_win][np.isfinite(e_nights_mean[n_win])]**-2)
        nights_mean_std[y] = nanwstd(nights_mean[n_win][np.isfinite(nights_mean[n_win])], w=e_nights_mean[n_win][np.isfinite(e_nights_mean[n_win])]**-2)
        e_nights_mean[y] = np.nanmax([e_nights_mean[y], nights_mean_std[y]])


    global jd, rva,err
    # calculate the global nightly drift
    rva = rv_mat - nights_mean

    ind = np.where(~(np.isnan(jd_mat) | np.isnan(rva)))
    jd = jd_mat[ind]
    rva = rva[ind]
    err = err_mat[ind]

    if pdfplot:
        jd_plot = jd * 1.
        rva_plot = rva * 1.

    global a1, b1, da1, db1, modbjd, popt, pcov

    if drift_corr:
        i_pre = jd <= drift_jd_max
        jd = jd[i_pre]
        rva = rva[i_pre]
        err = err[i_pre]
        modbjd = np.mod(jd, 1) - 0.5

        if 0:
            # try here to sort by nights and to center differently (-t(first observation))
            modbjd_new = jd * 1.
            for mdb in range(len(jd)):
                # find the reference here
                tmp = jd - np.floor(jd[mdb])
                tmp[tmp<0] = np.nanmax(tmp)
                indmin = np.where(tmp == np.nanmin(tmp))[0][0]
                modbjd_new[mdb] = modbjd_new[mdb] - jd[indmin]

        popt, pcov = fit_lin(modbjd, rva, err, method="ana")
        print("Fit result: ", popt)
        a1, b1 = popt
        print("Driftcorr: ", len(modbjd), len(rva))
        da1, db1 = pcov
        print(a1/24., b1)

# Goes through the GTO-RV rvc.dat files, creates (n_stars,m_nights) matrixes of JDs, RVs, and errors,
# calculates the nightly offsets (NZPs) and their errors, corrects the original (drift-corrected) RVs,
# writes obj.avc.dat and obj.avcn.dat files, makes some plots, and saves some outptut files (.orb, .mat, and .txt).
#
# Input:
# arm = 'vis' or 'nir' (default = 'vis').
# fname = full path to the file that contains full pathes to the GTO RV
#         files (default = ask the user).
# jd_min - first night of real survey (default = 2457390)
# Nrv_min - minimum RVs/star to be included in the analysis (default = 5).
# Nrv_min_night - minimum RVs/night to correct for its average (default = 3).
# sigma_outlier_star - Nsigma for outlier rejection per star (default = 10).
# sigma_outlier_night - Nsigma for outlier rejection per night (default = 4).
# init_rv_var - initial minimum std(RV) of an RV-loud star (m/s, default = 10 for the 'vis' and 100 for the 'nir').
# final_rv_var - final minimum std(RV) of an RV-loud star (m/s, default = 10 for the 'vis' and 100 for the 'nir').
# init_rv_systematics - initial guess of the systematic RV uncertainty
#                       (added to RV uncertainties, default = 2.5 m/s for the 'vis' and 13.1 for the 'nir').
# variables - cell array of hidden variable stars and/or RV-loud stars.
# remove_var - flag to remove hidden variable stars from NZP caclulation (default = 0 for the 'vis' and 1 for the 'nir').

#----------------------------------------------------------

# Output:
# stars - cell vector with the names of the stars
# Nrv - number of drift-corrected measurements per star
# rvc_mean_vec - vector of mean RVC per star
# rvc_mean_vec_err - vector of the error of the mean RVC per star
# Nout_star - Number of outliers per-star
# Nout_night - Number of outliers per-night
# ind_same_vec - number of nights in which the star was observed more than once
# rv_std_vec - std(RV) per star (before correction)
# rv_med_err - median RV error per star (before correction)
# nights_mean - nightly means
# e_nights_mean - nightly-mean error
# nights_mean_std - nightly-mean scatter divided by sqrt(Nrv_night)
# Nrv_night - number of good RVs per night
# rv_std_vec_corr - std(RV) per star (after correction)
# rv_med_err_corr - median RV error per star (after correction)
# jd_mat - [Nstars,Nrv] matrix of exposure times (BJD)
# rv_mat - [Nstars,Nrv] matrix of RVs
# err_mat - [Nstars,Nrv] matrix of RV errors
starttime = time.time()
now = Time.now().datetime

default = " (default: %(default)s)."
parser = argparse.ArgumentParser(add_help=False)
argopt = parser.add_argument   # function short cut
argopt('inst', nargs='?', help='instrument'+default, default='CARM_VIS', choices=['CARM_VIS', 'CARM_NIR'])
argopt('path', nargs='?', help='source directory'+default, default='../serval/{inst}/')
argopt('outdir', nargs='?', help='Output directory'+default, default='./{inst}/')
argopt('-date', help='Day string [YYYYMMDD] (default: today).', dest='daystr', default=f"{now.year}{now.month:02d}{now.day:02d}")
argopt('-?', '-h', '-help', '--help', help='show this help message and exit', action='help')
args = parser.parse_args()
globals().update(vars(args))
outdir = 'caracal_rvs/CARM_VIS_AVC_last' #outdir.format(inst=inst)
basedir = 'caracal_rvs/CARM_VIS_RVC_tzp'  # path.format(inst=inst)
#date = daystr[:4]+'-'+daystr[4:6]+'-'+daystr[6:]
date = '2019-10-22'
daystr = '2019-10-22'
arm = {'CARM_VIS':'vis', 'CARM_NIR':'nir'}[inst]


print(f"Processing data for {inst} channel.")
print('inst:   ', inst)
print('basedir:', basedir)
print('outdir:', outdir)
print('daystr: ', daystr)


if arm == "vis":
    jd_min = 2457390
    init_rv_var = 10.
    final_rv_var = 10.
    init_rv_systematics = 1.0   # initial jitter
    remove_var = 0 # 0= calculate internally
    drift_corr = 1 # flag to correct intra-night drift
    drift_jd_max = 2458047 # first night of FP always on (Oct 20, 2017)
elif arm == "nir":
    jd_min = 2457390
    init_rv_var = 30.
    final_rv_var = 30.
    init_rv_systematics = 3.0
    remove_var = 1 # 1 = read from a list
    drift_corr = 0 # flag to correct intra-night drift
    drift_jd_max = 2458047 # first night of FP always on (Oct 20, 2017)
    print("You are running NZP correction for the nir arm. Make sure the list of RV-loud stars is up to date!")


unselfbias = 2   # 0 means selfbiased; 1 means not self biased; 2 means both

savedata = 1
makeplots = 1
savefigs = 1   # leave this and makeplots on 1, if the plots should be saved to the disk; if you want only pop up plots then put this on 0 but leave makeplots on 1
pdfplot = 0

Nrv_min = 5
Nrv_min_night = 3
sigma_outlier_star = 10
sigma_outlier_night = 5
average_window = 6

# check if necessary directoties are there, if not create them
os.makedirs(outdir+"/avc", exist_ok=True)
os.makedirs(outdir+"/avc_selfbiased", exist_ok=True)
os.makedirs(outdir+"/avcn", exist_ok=True)
os.makedirs(outdir+"/avcn_selfbiased", exist_ok=True)
os.makedirs(outdir+"/results/mat_files", exist_ok=True)
os.makedirs(outdir+"/results/orb_files", exist_ok=True)
os.makedirs(outdir+"/results/txt_files", exist_ok=True)

dirname = outdir+'/results/mat_files/'

#----------------------------------------------------------------------------------

# collect all useful rvc dat files
objects = [x for x in os.listdir(basedir) if os.path.isdir(basedir+"/"+x) and x.startswith('J')]
objects.sort()
#print(objects)
targets = [] # list of all rvc-files
stars = [] # list of stars with rvcs
for obj in objects:
    temp = np.loadtxt(basedir+f"/{obj}/{obj}.rvc.dat")
    if len(temp):
        targets.append(basedir+f"/{obj}/{obj}.rvc.dat")
        stars.append(obj)

variable = ['J19255+096', 'J06594+193', 'J14307-086', 'J06024+498', 'J05532+242', 'J16554-083N', 'J17578+046',
            'J04167-120', 'J03531+625', 'J17033+514', 'J05127+196', 'J18409-133', 'J14082+805', 'J05084-210',
            'J23340+001', 'J11026+219', 'J06011+595', 'J11474+667', 'J20450+444', 'J05421+124', 'J19251+283',
            'J08526+283', 'J22021+014', 'J22096-046', 'J10584-107', 'J22137-176', 'J16167+672N', 'J09428+700',
            'J01125-169', 'J18498-238', 'J11054+435', 'J01026+623', 'J21466-001', 'J04290+219', 'J12373-208',
            'J09161+018', 'J19422-207', 'J09307+003', 'J12100-150', 'J11306-080', 'J18480-145', 'J10350-094',
            'J02070+496', 'J23492+024', 'J03181+382', 'J02222+478', 'J15598-082', 'J21221+229', 'J17052-050',
            'J05365+113', 'J21164+025', 'J19216+208', 'J02530+168', 'J04219+213', 'J04429+214', 'J07446+035',
            'J01019+541', 'J02362+068', 'J17578+465', 'J06105-218', 'J21152+257', 'J21019-063', 'J17355+616',
            'J11511+352', 'J13591-198', 'J08023+033', 'J21348+515', 'J04198+425', 'J13229+244', 'J00183+440',
            'J17303+055', 'J02123+035', 'J05314-036', 'J10416+376', 'J18427+596S', 'J09144+526', 'J05415+534',
            'J05019+011', 'J05348+138', 'J07403-174', 'J06371+175', 'J02442+255', 'J18022+642', 'J06103+821',
            'J09028+680', 'J18346+401', 'J03463+262', 'J11476+786', 'J04225+105', 'J23216+172', 'J11000+228',
            'J14257+236E', 'J18198-019', 'J07001-190', 'J06548+332', 'J07393+021', 'J08119+087', 'J13299+102',
            'J05033-173', 'J10564+070', 'J11201-104', 'J06318+414', 'J11055+435', 'J08161+013', 'J13457+148',
            'J17115+384', 'J04376-110', 'J05337+019', 'J18224+620', 'J03217-066', 'J13005+056', 'J12248-182',
            'J18482+076', 'J12479+097', 'J20305+654', 'J01025+716', 'J08409-234', 'J18427+596N', 'J05019-069',
            'J07274+052', 'J02002+130', 'J01433+043', 'J23381-162', 'J13427+332', 'J05360-076', 'J22559+178',
            'J07361-031', 'J20525-169', 'J17378+185', 'J21463+382', 'J07558+833', 'J05062+046', 'J16167+672S',
            'J05394+406', 'J11110+304W', 'J09360-216', 'J18131+260', 'J18363+136', 'J00162+198E', 'J15474-108',
            'J12230+640', 'J16555-083', 'J12312+086', 'J18174+483', 'J09411+132', 'J08358+680', 'J18319+406',
            'J22565+165', 'J11126+189', 'J22115+184', 'J07319+362N', 'J16303-126', 'J07287-032', 'J20336+617',
            'J22125+085', 'J19072+208', 'J14257+236W', 'J04376+528', 'J10482-113', 'J11417+427', 'J12123+544S',
            'J13450+176', 'J00067-075', 'J15194-077', 'J11033+359', 'J15305+094', 'J00389+306', 'J00286-066',
            'J10196+198', 'J14321+081', 'J19169+051N', 'J19098+176', 'J19084+322', 'J22532-142', 'J23419+441',
            'J13102+477', 'J02336+249', 'J17198+417', 'J23431+365', 'J15412+759', 'J20556-140N', 'J10289+008',
            'J10396-069', 'J18165+048', 'J21466+668', 'J23556-061', 'J10122-037', 'J01518+644', 'J22252+594',
            'J08293+039', 'J13536+776', 'J08536-034', 'J19070+208', 'J07582+413', 'J17364+683', 'J09143+526',
            'J00051+457', 'J04588+498', 'J23351-023', 'J20567-104', 'J23505-095', 'J14342-125', 'J22468+443',
            'J08413+594', 'J09140+196', 'J02358+202', 'J00184+440', 'J22114+409', 'J06421+035', 'J22503-070',
            'J22330+093', 'J11289+101', 'J20260+585', 'J18051-030', 'J09447-182', 'J22057+656', 'J10023+480',
            'J03213+799', 'J13458-179', 'J18419+318', 'J20533+621', 'J14544+355', 'J22298+414', 'J23585+076',
            'J05366+112', 'J09003+218', 'J03133+047', 'J10360+051', 'J20556-140S', 'J09425+700', 'J19346+045',
            'J08315+730', 'J02015+637', 'J09511-123', 'J12111-199', 'J10508+068', 'J01048-181', 'J08298+267',
            'J11477+008', 'J04538-177', 'J11467-140', 'J01339-176', 'J09561+627', 'J10504+331', 'J19169+051S',
            'J08126-215', 'J20451-313', 'J15218+209', 'J20405+154', 'J18027+375', 'J23245+578']


#===========================================================================
# STARTING POINT OF CALCULATING THE SPARSE MATRIX OF RV-NIGHT
#===========================================================================
N_targs = len(targets)
rv_std_vec = np.zeros(N_targs) * np.nan
Nrv = rv_std_vec*1.
rvc_mean_vec = Nrv*1.
rvc_mean_vec_err = Nrv*1.
rv_med_err = Nrv*1.
ind_same_vec = Nrv*1.
Nnodrift = Nrv*1.
Nout_star = Nrv*1.

jd_now = Time.now().jd
N_n = int(np.floor(jd_now)-jd_min+1)   # number of nights since beginning

jd_mat = np.zeros((N_targs, N_n), dtype=float) * np.nan
rv_mat = jd_mat * 1.
err_mat = jd_mat * 1.

if savedata:
    fid = open(outdir + f"/results/orb_files/gto_{arm}_rvc{daystr}.orb", "w")

for k in range(N_targs):
    print(k, stars[k])
    M = np.loadtxt(targets[k]).T
    starname = stars[k]
    # remove time stamps before jd_min
    ind = M[0] >= jd_min
    # remove time stamps with drift = NaN
    ind &= np.isfinite(M[3])

    bjd, rvc, e_rvc = M[0:3,ind]

    rvc_median = np.nanmedian(rvc)
    rv_med_err[k] = np.nanmedian(e_rvc)

    # find sigma_outliers
    bias = 1.-1./4./len(rvc[np.isfinite(rvc)]) if len(rvc[np.isfinite(rvc)]) else np.inf
    rv_std_vec[k] = rv_std = 1.48 * median_absolute_deviation(rvc) / bias
    bad_rv_ind = np.where((np.abs(rvc - rvc_median) > sigma_outlier_star * rv_std) & (np.abs(rvc - rvc_median) > init_rv_var))[0]
    Nrv[k] = len(rvc)
    Nout_star[k] = len(bad_rv_ind)

    # Write an entry to .orb file (drift-corrected RVs only, including the outliers)
    if savedata:
        fid.write(f"STAR: {starname}\n")
        for i in range(len(bjd)):
            fid.write(f"{bjd[i]} {0.001*rvc[i]} {0.001*e_rvc[i]}\n")
        fid.write("END\n")

    # Remove stellar outliers (only for the purpose of calculating the NZPs)
    bjd = np.delete(bjd, bad_rv_ind)
    rvc = np.delete(rvc, bad_rv_ind)
    e_rvc = np.delete(e_rvc, bad_rv_ind)

    # For nights with multiple exposures, take the mean RVC in that night, with its scatter as an error
    nights = np.floor(bjd).astype(int)
    columns = nights - jd_min
    ind_same_tmp = 0

    dub_nights_ind = find_duplicates(nights)
    if dub_nights_ind:
        ind_delete = [] # indices to remove afterwards
        for indx in dub_nights_ind:
            # indx is array of indices of same values
            bjd_same = nanwmean(bjd[indx], w=1./e_rvc[indx]**2.)
            rvc_same = nanwmean(rvc[indx], w=1./e_rvc[indx]**2.)
            e_rvc_same = np.nanmax([np.nanmin(e_rvc[indx]), nanwstd(rvc[indx], w=1./e_rvc[indx]**2.)])
            # now store indices to delete and save the new values at the first index of the multiples
            ind_delete.append(indx[1:])
            bjd[indx[0]] = bjd_same
            rvc[indx[0]] = rvc_same
            e_rvc[indx[0]] = e_rvc_same
            ind_same_tmp += len(indx)
        # now remove the duplicates
        ind_delete = [val for sublist in ind_delete for val in sublist]
        bjd = np.delete(bjd,ind_delete)
        rvc = np.delete(rvc,ind_delete)
        e_rvc = np.delete(e_rvc,ind_delete)
        columns = np.delete(columns,ind_delete)
        nights = np.delete(nights,ind_delete)
    ind_same_vec[k] = ind_same_tmp

    # calculate the weighted mean (zero-point of the star) and error
    #Nrv[k] = len(rvc)
    if Nrv[k]:
        bias = 1.-1./4./len(rvc[np.isfinite(rvc)]) if len(rvc[np.isfinite(rvc)]) else np.inf
        e_rvc = np.sqrt(e_rvc**2. + init_rv_systematics**2.)
        weights = e_rvc**-2.
        rvc_mean = nanwmean(rvc, w=weights)
        rvc_mean_vec[k] = rvc_mean
        rvc_mean_err = 1. / np.sqrt(sum(weights))
        rvc_mean_std = nanwstd(rvc, w=weights) / np.sqrt(Nrv[k]) / bias
        rvc_mean_err = np.nanmax([rvc_mean_err, rvc_mean_std])
        rvc_mean_vec_err[k] = rvc_mean_err

        # Write into the matrices:
        jd_mat[k,columns] = bjd
        rv_mat[k,columns] = rvc - rvc_mean # stellar zero-point subtracted RVs
        err_mat[k,columns] = np.sqrt(e_rvc**2.+rvc_mean_err**2.) # the error on the stellar zero-point is co-added to the RV error,
                                                                 # so that variables and/or faint stars get lower weight in NZP calculation.
    else:
        print(starname, "has no good RVs.")

# save the matrices
np.savetxt(dirname+"jd_mat.dat", jd_mat)
np.savetxt(dirname+"rv_mat.dat", rv_mat)
np.savetxt(dirname+"err_mat.dat", err_mat)
jd_mat_orig = jd_mat * 1.0
rv_mat_orig = rv_mat * 1.0
err_mat_orig = err_mat * 1.0

if savedata:
    fid.close()

# plot std(RV) histogram
mpl.rc('font', size=16)
n_bins = 50
if makeplots:
    outfile = outdir + f"/hist_stars_std_{daystr}.png"
    print("plotting ", outfile)
    params = {
          'xtick.direction':'in', 'ytick.direction':'in',
          'xtick.top':True, 'ytick.right':True,
          'xtick.major.size':8, 'ytick.major.size':8,
          'xtick.minor.visible':True, 'ytick.minor.visible':True,
          'xtick.minor.size':4, 'ytick.minor.size':4,
    }
    plt.rcParams.update(params)

    fig = plt.figure(figsize=(10, 6), dpi=200)
    histdata = rv_std_vec[Nrv > Nrv_min]
    med = np.median(rv_std_vec[rv_std_vec<init_rv_var])
    logbins = np.geomspace(1, 10.**(1.05*np.log10(np.nanmax(rv_std_vec))), n_bins)
    _, bins, patches = plt.hist(histdata, bins=logbins, histtype='step', facecolor='None', edgecolor='red', linewidth=2., label='before correction', zorder=2)
    plt.axvline(x=med, label="median RV-quite: %.2f m/s"%med, color='red', linestyle='dashed')
    plt.xscale('log')
    plt.xlabel('std(RV) [m/s]')
    plt.ylabel('Number of stars')
    plt.rc('axes', axisbelow=True)
    plt.axes().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: '%g' %x))
    plt.title(f"{inst} std before ({date})")
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(outfile, bbox_inches='tight')
    plt.clf()
    fig.clf()
    plt.close()


#===========================================================================
# STARTING POINT OF CORRECTING THE RADIAL VELOCITIES WITH SELF BIASING
#===========================================================================

if unselfbias in [0, 2]:
    novar_rv = np.where(rv_std_vec<init_rv_var)[0]
    rv_std_median = np.nanmedian(rv_std_vec[novar_rv])
    print('Before correction:')
    print("Median std(RV) of RV-quiet stars:", rv_std_median)
    print("std(RV) threshold for an RV_loud star:", init_rv_var)

    stars_flag = 1 * (rv_std_vec > init_rv_var)

    robust_nzp()

# ==========================================================================
    #xplot = np.arange(N_n)
    #plt.plot(xplot, nights_mean*0.0+1, "x", label="good nights (STD below threshold)")
    #print(len(np.where(np.isfinite(nights_mean))[0]))
    # find nights where NZP correction could not be applied and mark them:


## ==========================================================================
# make additional plots

    print(len(np.where(np.isfinite(nights_mean))[0]))
    #plt.plot(xplot, nights_mean*0.0+2, "x", label = "after 12 days averaging")
    #plt.ylim(0,10)
    #plt.show()
    if makeplots:
        fig = plt.figure(figsize=(15, 9), dpi=200)
        plt.plot(jd_mat.ravel()-jd_min, rv_mat.ravel(), ".g", alpha=0.5, label="RV$_{k,n}$")
        plt.plot(jd_mat[f_mat&2==2]-jd_min, rv_mat[f_mat&2==2], '.r', mfc='none', label="bad RV$_{k,n}$")
        # overplot NZP
        #fig = plt.figure(figsize=(20, 9), dpi=100)
        #plt.plot(night_jds-jd_min, night_rvs, ".b")
        plt.errorbar(nights_vec[good_mean]-0.55+1, nights_mean[good_mean], yerr=e_nights_mean[good_mean], fmt="ok", label="NZPs", zorder=2)
        bad = (Nrv_night<Nrv_min_night) & (Nrv_night>0)
        plt.errorbar(nights_vec[bad]-0.55+1, nights_mean[bad], yerr=e_nights_mean[bad], fmt='om', mfc='none', label="bad NZPs", zorder=3)
        plt.xlabel(f"JD - {jd_min}")
        plt.ylabel('RV [m/s]')
        plt.title(f"NZP calibration of {inst} ({date})")
        plt.xlim(-5, N_n+5)
        plt.ylim(-40, 40)
        plt.grid(True)
        plt.tight_layout()
        plt.legend(ncol=4, loc='upper left')
        textstr = f"median(NZP) = {median_nights_mean:.2f} m/s\nstd(NZP) = {std_nights_mean:.3f} m/s"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(10, 30, textstr, fontsize=14, verticalalignment='top', bbox=props)
        if savefigs:
            print(f"plotting {outdir}/night_means_{daystr}.png")
            plt.savefig(outdir+f"/night_means_{daystr}.png", bbox_inches='tight')
        else:
            plt.show()
        plt.clf()
        fig.clf()
        plt.close()


    # plot number of good RVs per night
    if makeplots:

        fig = plt.figure(figsize=(20, 9), dpi=100)
        plt.plot(nights_vec, Nrv_night, 'xb', label='number of good RVs')
        plt.plot(nights_vec, e_nights_mean, 'xk', label='nightly mean uncertainty [m/s]')
        plt.plot(nights_vec, nights_mean_std, 'xr', label='nightly mean scaled std [m/s]')
        plt.xlabel(f"Day since JD = {jd_min}")
        plt.xlim(-5, N_n+5)
        plt.title(f"Global analysis of Carmenes-gto {arm} RVs")
        plt.grid(True)
        plt.tight_layout()
        if savefigs:
            plt.savefig(outdir+f"/Nrv_night_{daystr}.png", bbox_inches='tight')
        else:
            plt.show()
        plt.clf()
        fig.clf()
        plt.close()

    if drift_corr:
        if makeplots:

            fig = plt.figure(figsize=(20, 9), dpi=100)
            plt.plot(modbjd, rva,'.b')
            plt.plot(modbjd, func_lin(modbjd, *popt), "-k")
            plt.xlabel("$t_\mathrm{mid}$ [hr]")
            plt.ylabel("NZP-corrected RV-quiet star RVs [m/s]")
            plt.xlim(np.nanmin(modbjd)-0.01, np.nanmax(modbjd)+0.01)
            plt.ylim(-20, 20)
            plt.title("Global intra-night fit")
            plt.grid(True)
            plt.tight_layout()
            textstr = f"linefit slope = {a1/24.:.3f} +/- {da1/24.:.3f} m/s/hr"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(np.nanmin(modbjd), 7, textstr, fontsize=14,
                verticalalignment='top', bbox=props)
            if savefigs:
                plt.savefig(outdir+f"/nightly_drift_{daystr}.png", bbox_inches='tight')
            else:
                plt.show()
            plt.clf()
            fig.clf()
            plt.close()

        if pdfplot:
            pp = PdfPages(outdir+f"/nightly_drift_nights_{daystr}.pdf")
            fig = plt.figure(figsize=(15, 9), dpi=200)
            mpl.rc('font', size=16)
            plt.clf()
            fig.clf()
            ax1 = plt.gca()
            for i in range(N_n):
                print(i+1, " of ", N_n)
                nightjd = i + jd_min
                if nightjd <= drift_jd_max*100:
                    jd_plot_ind = np.where(np.floor(jd_plot)==nightjd)[0]
                    print(nightjd, drift_jd_max, len(jd_plot_ind))
                    if len(jd_plot_ind):
                        modbjd_night = np.mod(jd_plot[jd_plot_ind],1) - 0.5
                        rva_night = rva_plot[jd_plot_ind]

                        ax1.plot(modbjd, rva,'.b')
                        ax1.plot(modbjd_night, rva_night,'or')
                        ax1.plot(modbjd, func_lin(modbjd, *popt), "-k")
                        #ax1.plot(modbjd, func_lin(bjd, -3.225, -0.2), "m-")
                        ax1.set_xlabel("$t_\mathrm{mid}$ [hr]")
                        ax1.set_ylabel("NZP-corrected RV-quiet star RVs [m/s]")
                        ax1.set_xlim(np.nanmin(modbjd)-0.01, np.nanmax(modbjd)+0.01)
                        ax1.set_ylim(-20, 20)
                        if nightjd <= drift_jd_max:
                            ax1.set_title(f"Global intra-night fit; Night = {nightjd}")
                        else:
                            ax1.set_title(f"Global intra-night fit; Night = {nightjd} (> drift_jd_max)")
                        plt.grid(True)
                        plt.tight_layout()
                        textstr = f"linefit slope = {a1/24.:.3f} +/- {da1/24.:.3f} m/s/hr"
                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        plt.text(np.nanmin(modbjd), 7, textstr, fontsize=14,
                            verticalalignment='top', bbox=props)
                        plt.savefig(pp, format='pdf')
                        plt.clf()
                        fig.clf()
                        ax1 = plt.gca()
            plt.close(fig)
            pp.close()



    ## apply the nightly correction, add its uncertainties, and write a new .orb,avc.dat, and avcn.dat files:
    if savedata:
        fid = open(outdir+f"/results/orb_files/gto_{arm}_selfbiased_avc_{daystr}.orb", "w")

    # initialize additional output vectors
    rv_std_vec_corr = np.zeros(N_targs) * np.nan
    rv_med_err_corr = np.zeros(N_targs) * np.nan

    # Go through the targets and apply the NZP correction:
    for k in range(N_targs):
        # Re-load all the RVs from the rvc.dat file for correction:
        starname = stars[k]
        Mavc = rvc2avc(filename=targets[k], suf='_selfbiased')
 
    if savedata:
        fid.close()

    # Find again RV-loud stars and make some display:
    rv_std_median_after = np.nanmedian(rv_std_vec_corr[novar_rv])
    print('After correction:')
    print("Median std(RV) of RV-quiet stars:", rv_std_median_after)
    print("std(RV) threshold for an RV_loud star:", final_rv_var)

    if makeplots:
        # Plot number of RVs per star versus its std(RV) after the correction:

        fig = plt.figure(figsize=(20, 9), dpi=100)
        plt.plot(Nrv, rv_std_vec_corr,'.b')
        #plt.plot(Nrv[var_rv],rv_std_vec_corr[var_rv],'or')
        plt.xlabel("$n_{RV}$")
        plt.ylabel("std(RV) [m/s]")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(3, 1000)
        plt.ylim(1, np.nanmax([1.2*np.nanmax(rv_std_vec), 10000]))
        plt.title(inst+" after correction")
        plt.grid(True)
        plt.tight_layout()
        if savefigs:
            plt.savefig(outdir+f"/RVstd_Nrv_after_{daystr}.png", bbox_inches='tight')
        else:
            plt.show()
        plt.clf()
        fig.clf()
        plt.close()

#===========================================================================
# STARTING POINT OF CORRECTING THE RADIAL VELOCITIES WITHOUT SELF BIASING
#===========================================================================
if unselfbias in [1, 2]:
    if savedata:
        fid = open(outdir+f"/results/orb_files/gto_{arm}_avc_{daystr}.orb", "w")

    # initialize additional output vectors
    rv_std_vec_corr = np.zeros(N_targs) * np.nan
    rv_med_err_corr = np.zeros(N_targs) * np.nan

    for k in range(N_targs):
        # Re-load all the RVs from the rvc.dat file for correction:
        starname = stars[k]
        print(f"\nRunning star number {k}/{N_targs}", starname)

        # Find variable stars:
        stars_flag = 1 * (rv_std_vec > init_rv_var)

        # Remove also the current star:
        stars_flag[k] |= 2

        # Average RVs from the same night using all stars observed in that night:

        #===========================================================================
        # STARTING POINT OF CALCULATING THE CORRECTION (NESTED LOOP)
        #===========================================================================
        robust_nzp()
        #=========================================================================
        # END OF NESTED LOOP
        #===========================================================================

        # apply the nightly correction, add its uncertainties, and write a new .orb,avc.dat, and avcn.dat files:
        Mavc = rvc2avc(filename=targets[k])

    if savedata:
        fid.close()

    # make some display:
    novar_rv = np.where(rv_std_vec<init_rv_var)[0]
    rv_std_median = np.nanmedian(rv_std_vec[novar_rv])
    print('Before correction:')
    print("Median std(RV) of RV-quiet stars:", rv_std_median)
    print("std(RV) threshold for an RV_loud star:", init_rv_var)
    rv_std_median_after = np.nanmedian(rv_std_vec_corr[novar_rv])
    print('After correction:')
    print("Median std(RV) of RV-quiet stars:", rv_std_median_after)
    print("std(RV) threshold for an RV_loud star:", final_rv_var)

## Plot the histogram of std(RV) after correction
if makeplots:

    print("\nplotting gtoc_hist_stars_std")
    fig = plt.figure(figsize=(10, 6), dpi=200)
    histdata = rv_std_vec_corr[Nrv > Nrv_min]
    logbins = np.geomspace(1, 10.**(1.05*np.log10(np.nanmax(rv_std_vec_corr))), n_bins)
    n, bins, patches = plt.hist(histdata, bins=logbins, facecolor='green', alpha=0.75, edgecolor='black', linewidth=0.3)
    plt.xscale('log')
    plt.xlabel('std(RV) [m/s] (after correction)')
    plt.ylabel('Number of stars')
    plt.title(f"Histograms of {inst} std(RV) per star (after correction)")
    plt.grid(True)
    plt.tight_layout()
    if unselfbias:
        print(outdir+f"/gtoc_hist_stars_std_{daystr}.png")
        plt.savefig(outdir+f"/gtoc_hist_stars_std_{daystr}.png", bbox_inches='tight')
    elif unselfbias == 0:
        plt.savefig(outdir+f"/selfbiased_gtoc_hist_stars_std_{daystr}.png", bbox_inches='tight')
    plt.clf()
    fig.clf()
    plt.close(fig)

    # comparison plot (before/after)
    fig = plt.figure(figsize=(10, 6), dpi=200)
    plt.title("Histograms of %s std(RV) per star (%s)" % (inst, date))
    histdata_old = rv_std_vec[Nrv > Nrv_min]
    histdata_corr = rv_std_vec_corr[Nrv > Nrv_min]
    logbins = np.geomspace(1, 10.**(1.05*np.log10(np.nanmax(rv_std_vec))), n_bins)

    plt.hist(histdata_old, bins=logbins, histtype='step', fill=False, label='std(RV) before correction', color='red', zorder=3)
    plt.hist(histdata_corr, bins=logbins, histtype='step', fill=False, label='std(RV) after correction', color='blue', zorder=4)
    plt.axvline(x=rv_std_median, label=f"median RV-quite: {rv_std_median:.3f} m/s", color='red', linestyle='dashed', linewidth=0.7)
    plt.axvline(x=rv_std_median_after, label=f"median RV-quite: {rv_std_median_after:.3f} m/s", color='blue', linestyle='dashed', linewidth=0.7)

    plt.xscale('log')
    plt.axes().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: '%g' %x))
    plt.xlabel('std(RV) [m/s]')
    plt.ylabel('Number of stars')

    # resort legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 0, 3, 1]
    #plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    plt.grid(True)
    plt.tight_layout()
    if savefigs:
        if unselfbias:
            print(outdir+f"/gtoc_{inst}_RVstd_hist_{daystr}.png")
            plt.savefig(outdir+f"/gtoc_{inst}_RVstd_hist_{daystr}.png", bbox_inches='tight')
        elif unselfbias == 0:
            plt.savefig(outdir+f"/selfbiased_gtoc_{arm}_RVstd_hist_{daystr}.png", bbox_inches='tight')
    else:
        plt.show()
    plt.clf()
    fig.clf()
    plt.close()

    # plot std(RV) before and after correction for RV-quiet stars:
    fig = plt.figure(figsize=(10, 6), dpi=200)
    ind_comp = np.where((rv_std_vec<init_rv_var) & (Nrv > 11))[0]
    plt.plot(rv_std_vec[ind_comp], rv_std_vec_corr[ind_comp], '.b')
    plt.xlim(0, final_rv_var)
    plt.ylim(0, final_rv_var)
    plt.plot([0, final_rv_var], [0, final_rv_var], '-k')
    plt.xlabel('std(RV) before correction [m/s]')
    plt.ylabel('std(RV) after correction [m/s]')
    plt.title(daystr+": NZP correction of carmenes-"+arm)
    plt.grid(True)
    plt.tight_layout()
    if savefigs:
        if unselfbias == 1:
            plt.savefig(outdir+f"/gto_RVstd_compare_{daystr}.png", bbox_inches='tight')
        elif unselfbias == 0:
            plt.savefig(outdir+f"/selfbiased_gto_RVstd_compare_{daystr}.png", bbox_inches='tight')
    else:
        plt.show()
    plt.clf()
    fig.clf()
    plt.close()

#===========================================================================
# STARTING POINT OF SAVING THE RESULTS
#===========================================================================
if savedata:
    # all stars
    savedat(outdir+f"/results/txt_files/gtoc_{arm}_RVstd_{daystr}.txt",
               stars, Nrv, rv_std_vec, rv_med_err, rv_std_vec_corr, rv_med_err_corr)

    # make a txt table for RV-loud stars
    k = np.where(rv_std_vec_corr>final_rv_var)[0]
    savedat(outdir+f"/results/txt_files/rv-loud_stars_{daystr}.txt",
            [stars[i] for i in k], Nrv[k], rv_std_vec_corr[k], rv_std_vec_corr[k]/np.sqrt(2.*Nrv[k]), rv_med_err_corr[k])
    savedat(outdir+f"/results/txt_files/rv-loud_stars_{daystr}.lis",
            [stars[i] for i in k])

    # make a txt table for the nightly averages
    flag_nzp = np.zeros(len(Nrv_night)) # MZ should be .astype(int)
    flag_nzp[bad_mean] = 1
    savedat(outdir+f"/results/txt_files/gtoc_{arm}_night_zero_{daystr}.txt",
               jd_min+nights_vec, nights_mean, e_nights_mean, Nrv_night, flag_nzp)

endtime = time.time()
print(f"time: {endtime-starttime:.3f} s ({(endtime-starttime)/60:.3f} min)")
print("NZPs done!")

variablemeint = ['J00162+198W',
    'J01019+541',
    'J01033+623',
    'J01056+284',
    'J01352-072',
    'J02002+130',
    'J02088+494',
    'J02519+224',
    'J03473-019',
    'J04198+425',
    'J04219+213',
    'J04472+206',
    'J05019+011',
    'J05062+046',
    'J05084-210',
    'J05337+019',
    'J05365+113',
    'J05394+406',
    'J05532+242',
    'J06000+027',
    'J06318+414',
    'J06396-210',
    'J06574+740',
    'J07001-190',
    'J07033+346',
    'J07361-031',
    'J07403-174',
    'J07446+035',
    'J07472+503',
    'J07558+833',
    'J08298+267',
    'J08409-234',
    'J08413+594',
    'J08536-034',
    'J09003+218',
    'J09033+056',
    'J09140+196',
    'J09161+018',
    'J09439+269',
    'J09449-123',
    'J10182-204',
    'J10196+198',
    'J10354+694',
    'J10360+051',
    'J10504+331',
    'J11026+219',
    'J11201-104',
    'J11417+427',
    'J11421+267',
    'J11474+667',
    'J11476+002',
    'J12156+526',
    'J12189+111',
    'J13005+056',
    'J13536+776',
    'J13591-198',
    'J14155+046',
    'J14173+454',
    'J14321+081',
    'J15194-077',
    'J15218+209',
    'J15305+094',
    'J15412+759',
    'J15474-108',
    'J15499+796',
    'J16102-193',
    'J16313+408',
    'J16555-083',
    'J16570-043',
    'J16578+473',
    'J17338+169',
    'J18022+642',
    'J18131+260',
    'J18174+483',
    'J18189+661',
    'J18356+329',
    'J18498-238',
    'J19169+051S',
    'J19255+096',
    'J19422-207',
    'J19511+464',
    'J20093-012',
    'J20198+229',
    'J20556-140N',
    'J22012+283',
    'J22468+443',
    'J22518+317',
    'J22532-142',
    'J23064-050',
    'J23113+085',
    'J23548+385',
    'J23556-061',
    'J23585+076']
