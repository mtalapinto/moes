# drift_a = data_A['posm'].values - data_A_zero['posm'].values
        # drift_norm_constant_a = 1 / np.sum(1 / (np.sqrt(data_drift_a['posme_i'] ** 2 + data_drift_a['posme_0'] ** 2)))
        # drift_norm_constant_a = 1 / np.sum(1 / (np.sqrt(data_A['posme'] ** 2 + data_A_zero['posme'] ** 2)))
        # weights_drifts_a = drift_norm_constant_a / np.sqrt(data_A['posme'] ** 2 + data_A_zero['posme'] ** 2)
        # drift_a_weighted = np.sum(weights_drifts_a * drift_a)
        # drift_a_weighted_std = np.std(weights_drifts_a * drift_a)
        # da.append(drift_a_weighted)
        # da_std.append(drift_a_weighted_std)
        # print('Weighted drift in A = ', drift_a_weighted)


'''

    rms_nzp = np.sqrt(np.sum(data['nzp']**2)/len(data))
    #print(rms_nzp)
    rms_tzp = np.sqrt(np.sum(data['tzp']**2)/len(data))
    #print(rms_tzp)

    pcoeff2, _ = spearmanr(data['nzp'], dt)
    pcoeff3, _ = pearsonr(data['nzp'], dt)
    pcoeff = np.corrcoef(data['nzp'], dt)
    #print(pcoeff, pcoeff2, pcoeff3)
    fapLevels = np.array([0.1, 0.05, 0.01])
    ls = LombScargle(data['bjd'].values, data['nzp'].values, normalization='psd')
    frequency, power = ls.autopower(minimum_frequency=0.00125, maximum_frequency=0.1, samples_per_peak=50)
    gls = pyPeriod.Gls((data['bjd'].values, data['nzp'].values), norm='ZK')
    plevels = gls.powerLevel(fapLevels)
    #print(ls)
    #print('NZP max. power Period (days) = ', np.round(1./frequency[np.argmax(power)], 2))
    ls2 = LombScargle(data['bjd'].values, data['tzp'].values, normalization='psd')
    gls2 = pyPeriod.Gls((data['bjd'].values, data['tzp'].values), norm='ZK')
    plevels2 = gls2.powerLevel(fapLevels)
    frequency2, power2 = ls2.autopower(minimum_frequency=0.00125, maximum_frequency=0.1, samples_per_peak=50)
    #print(ls2)
    #print('TZP max. power Period (days) = ', np.round(1. / frequency[np.argmax(power2)], 2))

    prob = [0.01]
    #levels = ls.false_alarm_probability(power.max(), method='bootstrap')
    #levels2 = ls2.false_alarm_probability(power2.max(), method='bootstrap')
    #print(levels)
    xtest = [0, 1e3]
    #lev = [levels, levels]
    #lev2 = [levels2, levels2]
    ls3 = LombScargle(dtarr['bjd'].values, dtarr['dt'], normalization='psd')
    frequency3, power3 = ls3.autopower(minimum_frequency=0.00125, maximum_frequency=0.1, samples_per_peak=50)
    #print(ls3)
    #print('Temp gradient max. power Period (days) = ', np.round(1. / frequency[np.argmax(power3)], 2))

    gls3 = pyPeriod.Gls((dtarr['bjd'].values, dtarr['dt'].values), norm='ZK')
    #plevels3 = gls3.powerLevel(fapLevels)
    #print(plevels)
    #print(plevels2)
    #print(plevels3)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))

    #plt.plot(xtest, lev, 'b--')
    #plt.plot(xtest, lev2, '--', color='darkorange')
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    #plt.xlim(0, 800)
    axes.plot(1/frequency, power, '-', color='blue', label='NZP')
    axes.plot(1/frequency2, power2, '-', color = 'darkorange', label='TZP')
    axes.set_xlim(0, 800)
    axes.legend(loc=1)
    axtemp2 = axes.twinx()
    axtemp2.yaxis.set_major_locator(ticker.NullLocator())
    axtemp2.set_xlim(0, 800)
    axtemp2.plot(1/frequency3, power3, '-', color = 'black', label = r'$\Delta T$')
    axtemp2.legend(loc=2)
    #for i in range(len(fapLevels)):
    #    plt.plot([min(clp.freq), max(clp.freq)], [plevels[i]]*2, '--', label='FAP = %4.1f%%' %(fapLevels[i])*100)
    #plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('plots/zps_gls.png')
    plt.clf()
    '''

'''
    for k in range(len(star_data)):
        #fddaux = (star_data['dd_c_date'].values[k] - star_data['dd_moes_obs'].values[k]) * star_data['pix2ms'].values[k]
        e_tzp_aux = np.sqrt(star_data['e_tzp'].values[k] ** 2 + (star_data['dd_m_date'].values[k]*star_data['pix2ms'].values[k]) ** 2)
        nzp.append(star_data['nzp'].values[k])
        e_nzp.append(star_data['e_nzp'].values[k])
        bjd.append(star_data['bjd'].values[k])
        #tzp.append(star_data['tzp_med'].values)
        tzp.append(tzp_test[k])
        #e_tzp.append(star_data['e_tzp'].values[k])
        rvcout.append(star_data['rvc'].values[k])
        avcout.append(star_data['avc'].values[k])
        star.append(star_data['starid'].values[k])
        sptype.append(star_data['spt'].values[k])
        t1.append(star_data['t1'].values[k])
        t2.append(star_data['t2'].values[k])
        t3.append(star_data['t3'].values[k])
        t4.append(star_data['t4'].values[k])
        t5.append(star_data['t5'].values[k])
        t6.append(star_data['t6'].values[k])
        t7.append(star_data['t7'].values[k])
        t8.append(star_data['t8'].values[k])
    #tvc0 = star_data['rvc'] + star_data['tzp0']
    #std_tvc0 = np.std(tvc0)
    #tvc1 = star_data['rvc'] + star_data['tzp1']
    #std_tvc1 = np.std(tvc1)
    #tvc2 = star_data['rvc'] + star_data['tzp2']
    #std_tvc2 = np.std(tvc2)
    #tvc3 = star_data['rvc'] + star_data['tzp3']
    #std_tvc3 = np.std(tvc3)
    '''

quiet_stars = np.array(quiet_stars)
print(quiet_stars)
# zps = zps.loc[zps.nzp > -4]
# zps = zps.loc[zps.nzp < 4]
print('Total number of stars = ', nstars)
print('Total number of observations = ', nobs)
# print('Total number of observing nights = ', ndates)

# stars = np.unique(zps['star'].values)
# quiet_stars = []
# for i in range(len(stars)):
#    zpdata = zps.loc[zps.star == stars[i]]
#    tvc2 = zpdata['rvc'] - zpdata['tzp']
#    std_tvc2 = np.std(tvc2)
##    std_avc2 = np.std(zpdata['avc'].values)
#   std_rvc2 = np.std(zpdata['rvc'].values)
# print(zpdata)
# print(stars[i], std_rvc2, std_avc2, std_tvc2)
#    quiet_stars.append(np.array([stars[i], std_rvc2, std_avc2, std_tvc2]))


# quiet_stars = np.array(quiet_stars)
# print(quiet_stars)
'''
for k in range(len(quiet_stars)):
    if quiet_stars[k][3] < quiet_stars[k][1]:
        ntvc += 1
        tvc_stars.append(quiet_stars[k])
    elif quiet_stars[k][3] < quiet_stars[k][2]:
        ntvc_plus += 1
        tvc_stars_plus.append(quiet_stars[k])
        #print(quiet_stars[k])

print('Number of TVC well corrected stars = ', ntvc)
print('Number of TVC super well corrected stars = ', ntvc_plus)
print('Percentage of TVC well corrected stars = ', ntvc / nstars * 100, '%')
print('Percentage of TVC super well corrected stars = ', ntvc_plus / nstars * 100, '%')
'''
# tvc_stars = np.array(tvc_stars)
# tvc_stars_plus = np.array(tvc_stars_plus)

# tvc_mean = np.mean(tvc_stars[:, 1].astype(np.float))
# tvc2 = np.mean(tvc_stars[:, 3].astype(np.float))
# print((tvc_mean - tvc2) * 100 / tvc_mean)
# tvc_mean_plus = np.mean(tvc_stars_plus[:, 2].astype(np.float) - tvc_stars_plus[:, 3].astype(np.float))

# tvc_stars_mean = np.mean(tvc_stars_plus[:, 1].astype(np.float))
# tvc_stars_plus_mean = np.mean(tvc_stars_plus[:, 2].astype(np.float))
# tvc_stars_plus2_mean = np.mean(tvc_stars_plus[:, 3].astype(np.float))

# print(tvc_stars_mean, tvc_stars_plus_mean, tvc_stars_plus2_mean)
print(quiet_stars)
rvc_std = quiet_stars[:, 1].astype(np.float)
avc_std = quiet_stars[:, 2].astype(np.float)
tvc_std = quiet_stars[:, 3].astype(np.float)
print(np.mean(rvc_std), np.mean(tvc_std), np.mean(avc_std))
# PLOT

fig, axes = plt.subplots(nrows=3,
                         ncols=1,
                         figsize=(14, 10),
                         gridspec_kw={'height_ratios': [1, 1, 1]},
                         sharex=True,
                         sharey=False)

# plt.figure(figsize=[10, 3])
binwidth = 0.5
bins_all = np.arange(0., 10., binwidth)

# RVC
axes[0].hist(rvc_std, bins=bins_all, color='yellow', alpha=0.8, label='Non-corrected', edgecolor='black')
axes[0].hist(avc_std, bins=bins_all, color='blue', alpha=0.5, label='NZP corrected', edgecolor='black')
axes[0].legend(fontsize=16)
hist_rvc, bins_rvc = np.histogram(rvc_std, bins=bins_all)
yrange = [0, 10, 20, 30, 40]
axes[0].set_yticklabels(yrange, fontsize=18)

bins_rvc = bins_rvc[1:] - binwidth / 2
rvc_weighted_mean = np.sum(bins_rvc * (hist_rvc / len(rvc_std)))
# AVC
axes[1].hist(rvc_std, bins=bins_all, color='yellow', alpha=0.8, label='Non-corrected', edgecolor='black')
axes[1].hist(tvc_std, bins=bins_all, color='red', alpha=0.5, label='TZP corrected', edgecolor='black')
axes[1].set_ylabel('Number of stars', fontsize=20)
axes[1].legend(fontsize=16)
axes[1].set_yticklabels(yrange, fontsize=18)
hist_avc, bins_avc = np.histogram(avc_std, bins=bins_all)
bins_avc = bins_avc[1:] - binwidth / 2
avc_weighted_mean = np.sum(bins_avc * (hist_avc / len(avc_std)))

# TVC

axes[2].hist(avc_std, bins=bins_all, color='blue', alpha=0.5, label='NZP corrected', edgecolor='black')
axes[2].hist(tvc_std, bins=bins_all, color='red', alpha=0.5, label='TZP corrected', edgecolor='black')
axes[2].legend(fontsize=16)
axes[2].set_yticklabels(yrange, fontsize=18)

plt.xticks(fontsize=18)

hist_tvc, bins_tvc = np.histogram(tvc_std, bins=bins_all)
# print(bins_tvc)
bins_tvc = bins_tvc[1:] - binwidth / 2
# print(weighted_average_m1(bins_tvc, hist_tvc))
# print(weighted_average_m1(bins_rvc, hist_rvc))
# print(weighted_average_m1(bins_avc, hist_avc))
# weighted_average_m1(hist_tvc, )
axes[2].plot(bins_tvc, hist_tvc, 'ro')
tvc_weighted_mean = np.sum(bins_tvc * (hist_tvc / len(tvc_std)))

# lineas verticales

tzp_mean = [np.mean(tvc_std), np.mean(tvc_std)]
nzp_mean = [np.mean(avc_std), np.mean(avc_std)]
nozp_mean = [np.mean(rvc_std), np.mean(rvc_std)]
yrange = [0., 100.]
# axes[2].plot(tzp_mean, yrange, '--', 'red')
# axes[2].plot(nzp_mean, yrange, '--', 'blue')


# plt.hist(quiet_stars[:, 4].astype(np.float), bins=bins_all, color='green', alpha=0.5, label='TZP corrected', edgecolor='black')
# plt.hist(quiet_stars[:, 5].astype(np.float), bins=bins_all, color='purple', alpha=0.5, label='TZP corrected', edgecolor='black')
# plt.hist(quiet_stars[:, 3].astype(np.float), bins=bins_all, color='blue', alpha=0.5, label='TZP corrected', edgecolor='black')
# plt.hist(quiet_stars[:, 7].astype(np.float), bins=bins_all, color='brown', alpha=0.5, label='TZP corrected', edgecolor='black')
# hist_tvc, bins_tvc = np.histogram(quiet_stars[:, 3].astype(np.float), bins=bins_all)
# bins_tvc = bins_tvc[1:]
# tvc_weighted_mean = np.sum(bins_tvc * (hist_tvc / len(tvc_std)))

# print(rvc_weighted_mean, avc_weighted_mean, tvc_weighted_mean)
# plt.legend()
plt.xlabel('RVs standard deviation [m/s]', fontsize=20)
# plt.ylabel('Number of RV-quiet stars')
# plt.savefig('plots/std_rvc_vs_avc.png')
# plt.savefig('plots/std_avc_vs_tvc.png')
# plt.savefig('plots/std_rvc_vs_tvc.png')
# plt.savefig('plots/std_rvc_vs_avc_vs_tvc.png',bbox_inches='tight')
plt.savefig('plots/std_rvc_vs_avc_vs_tvc_v1.png',
            bbox_inches='tight')
plt.show()
plt.clf()